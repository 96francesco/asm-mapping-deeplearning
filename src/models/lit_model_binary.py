import torchmetrics
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import pytorch_lightning as pl

from torch.optim.lr_scheduler import StepLR

class LitModelBinary(pl.LightningModule):
    """
    A LightningModule for binary classification with various loss functions and optimizers.
    Implements forward pass, training, validation, test steps, and optimizer configuration.
    
    Attributes:
        model (torch.nn.Module): The underlying model for predictions. 
            Defaults to SMP's Unet with ResNet34 backbone if None.
        loss (str): Loss function name ('bce', 'focal', 'iou'). Default: 'bce'.
        pos_weight (torch.Tensor, optional): A weight of positive examples for unbalanced datasets.
        weight_decay (float): Weight decay (L2 penalty) for optimizer. Default: 1e-5.
        optimizer (str): Optimizer name ('sgd', 'adam'). Default: 'sgd'.
        num_classes (int): Number of target classes. Default: 2 for binary classification.
    
    Methods:
        forward(x): Defines the forward pass of the model.
        training_step(train_batch, batch_idx): Processes a single batch during training.
        validation_step(val_batch, batch_idx): Processes a single batch during validation.
        test_step(test_batch, batch_idx): Processes a single batch during testing, 
            calculates and logs metrics.
        configure_optimizers(): Configures and returns the model's optimizers and 
            learning rate schedulers.
    """
    def __init__(self, model=None, loss='bce', pos_weight=None, weight_decay=1e-5,
                optimizer='sgd', num_classes=2):
        super().__init__()
        self.weight_decay = weight_decay
        self.num_classes = num_classes

        if optimizer == 'sgd':
            self.optimizer = torch.optim.SGD
        elif optimizer == 'adam':
            self.optimizer = torch.optim.Adam
        else:
            raise ValueError(f'Unkwnown optimizer')

        if model is None:
            # initialize 'standard' unet
            # None is used when loading model from checkpoints
            self.model = smp.Unet(
                    encoder_name="resnet34",
                    decoder_use_batchnorm=True,
                    decoder_attention_type='scse',
                    encoder_weights=None,
                    in_channels=7,
                    classes=1
            )
            pass
        else:
            self.model = model


        if loss == 'bce':
            if pos_weight is not None:
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to('cuda'))
            else:
                self.criterion = nn.BCEWithLogitsLoss()
        elif loss == 'focal':
            self.criterion = smp.losses.FocalLoss(alpha=0.25, gamma=2.0)
        elif loss == 'iou':
            self.criterion = smp.losses.JaccardLoss()
        else:
            raise ValueError(f'Unkwnon loss function: {loss}')

        # initialize accuracy metrics
        self.accuracy = torchmetrics.Accuracy(task='binary',
                                            threshold=0.5)
        self.precision = torchmetrics.Precision(task='binary',
                                                num_classes=self.num_classes,
                                                average='macro')
        self.recall = torchmetrics.Recall(task='binary',
                                        num_classes=self.num_classes,
                                        average='macro')
        self.f1_score = torchmetrics.F1Score(task='binary',
                                            num_classes=self.num_classes,
                                            average='macro')

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
                on_epoch=True)

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
                on_epoch=True)
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
        preds = (probs > 0.5).float() # apply threshold to probrabilities

        acc = self.accuracy(probs.squeeze(), y.squeeze().long())

        # log loss and accuracy metrics
        self.log('test_loss', loss)
        self.log('accuracy', acc)
        self.log('precision', self.precision(preds, y))
        self.log('recall', self.recall(preds, y))
        self.log('f1_score', self.f1_score(preds, y))

        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer(self.model.parameters(), lr=1e-3,
                                        weight_decay=self.weight_decay)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.25)
        return [optimizer], [scheduler]
