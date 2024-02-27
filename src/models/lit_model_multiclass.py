import torchmetrics
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import pytorch_lightning as pl

from torch.optim.lr_scheduler import StepLR

class LitModelMulticlass(pl.LightningModule):
    """
    A LightningModule for multiclass classification with various loss functions and optimizers.
    Implements forward pass, training, validation, test steps, and optimizer configuration.

    Attributes:
        model (torch.nn.Module): The underlying model for predictions.
            Defaults to SMP's Unet with ResNet34 backbone if None.
        loss (str): Loss function name ('ce', 'focal', 'iou'). Default: 'ce'.
        pos_weight (torch.Tensor, optional): A weight of positive examples for unbalanced datasets.
        weight_decay (float): Weight decay (L2 penalty) for optimizer. Default: 1e-5.
        lr (float): Learning rate for optimizer. Default: 1e-3.
        num_classes (int): Number of classes. Default: 11.
        ignore_index (int): Index to ignore in loss calculations. Default: -100.
        optimizer (str): Optimizer name ('sgd', 'adam'). Default: 'sgd'.

        Methods:
        forward(x): Defines the forward pass of the model.
        training_step(train_batch, batch_idx): Processes a single batch during training.
        validation_step(val_batch, batch_idx): Processes a single batch during validation.
        test_step(test_batch, batch_idx): Processes a single batch during testing,
            calculates and logs metrics.
        configure_optimizers(): Configures and returns the model's optimizers and
            learning rate schedulers.
    """
    def __init__(self, model=None, loss='ce', pos_weight=None, weight_decay=1e-5,
                 optimizer='sgd', lr=1e-3, num_classes=11, ignore_index=-100):
        super().__init__()
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.lr = lr
        self.save_hyperparameters()

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
                    classes=11
            )
            pass
        else:
            self.model = model


        if loss == 'ce':
            if pos_weight is not None:
                self.criterion = nn.CrossEntropyLoss(weight=pos_weight.to('cuda'))
            else:
                self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        elif loss == 'focal':
            self.criterion = smp.losses.FocalLoss(mode='multiclass',
                                                  alpha=0.25,
                                                  ignore_index=self.ignore_index)
        elif loss == 'iou':
            self.criterion = smp.losses.JaccardLoss(mode='multiclass')
        else:
            raise ValueError(f'Unkwnon loss function: {loss}')

        # initialize accuracy metrics
        self.accuracy = torchmetrics.Accuracy(task='multiclass',
                                                num_classes=self.num_classes,
                                                average='macro')
        self.precision = torchmetrics.Precision(task='multiclass',
                                                num_classes=self.num_classes,
                                                average='macro')
        self.recall = torchmetrics.Recall(task='multiclass',
                                        num_classes=self.num_classes,
                                        average='macro')
        self.f1_score = torchmetrics.F1Score(task='multiclass',
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
        loss = self.criterion(outputs, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False,
                 on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, test_batch, batch_idx):
        """
        Perform forward and backward propagation to calculate test loss and
        accuracy metrics
        """
        x, y = test_batch
        logits = self.model(x) # get raw logits
        loss = self.criterion(logits, y) # compute loss

        # use softmax to convert logits to probabilities for multiclass
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)  # get the predicted class for each pixel

        # calculate accuracy and other metrics
        acc = self.accuracy(preds, y)
        self.log('test_loss', loss, sync_dist=True)
        self.log('accuracy', acc, sync_dist=True)
        self.log('precision', self.precision(preds, y), sync_dist=True)
        self.log('recall', self.recall(preds, y), sync_dist=True)
        self.log('f1_score', self.f1_score(preds, y), sync_dist=True)

        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer(self.model.parameters(), lr=self.lr,
                                     weight_decay=self.weight_decay)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.25)
        return [optimizer], [scheduler]