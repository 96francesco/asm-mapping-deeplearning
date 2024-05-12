import torchmetrics
import segmentation_models_pytorch as smp
import torch
import pytorch_lightning as pl

from torch.optim.lr_scheduler import StepLR

class LitModelStandalone(pl.LightningModule):
    """
    A LightningModule for binary classification with Focal loss and Adam optimizer.
    Implements forward pass, training, validation, test steps, and optimizer configuration.
    
    Attributes:
        model (torch.nn.Module): The underlying model for predictions. 
            Defaults to SMP's Unet with ResNet34 backbone if None.
        pos_weight (torch.Tensor, optional): A weight of positive examples for unbalanced datasets.
        weight_decay (float): Weight decay (L2 penalty) for optimizer. Default: 1e-5.
        lr (float): Learning rate for optimizer. Default: 1e-3.
        threshold (float): Threshold for binary classification. Default: 0.5.
        optimizer (torch.optim.Optimizer): Optimizer class. Default: torch.optim.Adam.
        criterion (torch.nn.Module): Loss function. Default: smp.losses.FocalLoss.
        alpha (float): Focal loss alpha parameter. Default: 0.25.
        gamma (float): Focal loss gamma parameter. Default: 2.0.
    
    Methods:
        forward(x): Defines the forward pass of the model.
        training_step(train_batch, batch_idx): Processes a single batch during training.
        validation_step(val_batch, batch_idx): Processes a single batch during validation.
        test_step(test_batch, batch_idx): Processes a single batch during testing, 
            calculates and logs metrics.
        configure_optimizers(): Configures and returns the model's optimizers and 
            learning rate schedulers.
    """
    def __init__(self, model=None, weight_decay=1e-5,
                 lr=1e-3, threshold=0.5, in_channels=7, alpha=0.25, gamma=2.0):
        super().__init__()
        self.weight_decay = weight_decay
        self.lr = lr
        self.threshold = threshold
        self.alpha = alpha
        self.gamma = gamma
        self.optimizer = torch.optim.Adam
        self.criterion = smp.losses.FocalLoss(alpha=alpha, gamma=gamma, mode='binary')
        self.save_hyperparameters()

        if model is None:
            # initialize 'standard' unet
            # None is used when loading model from checkpoints
            self.model = smp.Unet(
                    encoder_name="resnet34",
                    decoder_use_batchnorm=True,
                    decoder_attention_type='scse',
                    encoder_weights=None,
                    in_channels=in_channels,
                    classes=1
            )
            pass
        else:
            self.model = model

        # initialize accuracy metrics
        self.accuracy = torchmetrics.Accuracy(task='binary', 
                                              average='macro',
                                              threshold=self.threshold)
        self.precision = torchmetrics.Precision(task='binary', 
                                                average='macro',
                                                threshold=self.threshold)
        self.recall = torchmetrics.Recall(task='binary', 
                                          average='macro',
                                          threshold=self.threshold)
        self.f1_score = torchmetrics.F1Score(task='binary', 
                                             average='macro',
                                             threshold=self.threshold)

    def forward(self, x):
        """
        Wrapper function of the model's forward function
        """
        return self.model.forward(x)

    def training_step(self, train_batch, batch_idx):
        """
        Perform forward and backward propagation to calculate train loss
        """
        x, y = train_batch
        outputs = self(x)
        y = y.unsqueeze(1).type_as(x) # add a channel dimension
        loss = self.criterion(outputs, y)
        self.log('train_loss', loss, prog_bar=True, on_step=False,
                on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        """
        Perform forward and backward propagation to calculate val loss
        """
        x, y = val_batch
        outputs = self(x)
        y = y.unsqueeze(1).type_as(x) # add a channel dimension
        loss = self.criterion(outputs, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False,
                on_epoch=True, sync_dist=True)
        self.log('val_f1score', self.f1_score(outputs, y),
                  prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        return loss

    def test_step(self, test_batch, batch_idx):
        """
        Perform forward and backward propagation to calculate test loss and
        accuracy metrics
        """
        x, y = test_batch
        y = y.unsqueeze(1).type_as(x) # convert int to float
        logits = self.model(x) # get raw logits
        loss = self.criterion(logits, y) # compute loss

        probs = torch.sigmoid(logits) # convert logits to probabilities
        preds = (probs > self.threshold).float() # apply threshold to probrabilities

        acc = self.accuracy(probs.squeeze(), y.squeeze().long())

        # log loss and accuracy metrics
        self.log('test_loss', loss, sync_dist=True)
        self.log('accuracy', acc, sync_dist=True)
        self.log('precision', self.precision(preds, y), sync_dist=True)
        self.log('recall', self.recall(preds, y), sync_dist=True)
        self.log('f1_score', self.f1_score(preds, y), sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer(self.model.parameters(), lr=self.lr,
                                        weight_decay=self.weight_decay)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.8)
        return [optimizer], [scheduler]
