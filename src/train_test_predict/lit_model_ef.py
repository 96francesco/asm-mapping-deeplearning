import pytorch_lightning as pl
import torch
import segmentation_models_pytorch as smp
import torchmetrics

from torch.optim.lr_scheduler import StepLR

class LitModelEarlyFusion(pl.LightningModule):
      """
      A PyTorch Lightning module for an Early Fusion U-Net. This model fuses features
      from Sentinel-1 and Planet images at the input level via a concatenation operation.

      Attributes:
            lr (float): learning rate for the optimizer.
            threshold (float): threshold for binary classification.
            weight_decay (float): weight decay for the optimizer.
            alpha (float): alpha parameter for the Focal Loss.
            gamma (float): gamma parameter for the Focal Loss.
            optimizer (torch.optim): optimizer for the model.
            criterion (smp.losses.FocalLoss): focal loss function for the model.
            combined_in_channels (int): number of input channels for the model.
            encoder_decoder (smp.Unet): U-Net model for the encoder-decoder.
            accuracy_class (torchmetrics.Accuracy): class-wise accuracy metric.
            precision_class (torchmetrics.Precision): class-wise precision metric.
            recall_class (torchmetrics.Recall): class-wise recall metric.
            f1_score_class (torchmetrics.F1Score): class-wise F1 score metric.
      
      Methods:
            forward(planet_input, s1_input): Forward pass of the model.
            training_step(batch, batch_idx): Training step for the model.
            validation_step(batch, batch_idx): Validation step for the model.
            test_step(batch, batch_idx): Test step for the model.
            configure_optimizers(): Configure the optimizer and scheduler for the model.
      """
      def __init__(self, lr=1e-3, threshold=0.5, weight_decay=1e-5, alpha=0.25, gamma=2.0):
            super().__init__()
            self.lr = lr
            self.threshold = threshold
            self.weight_decay = weight_decay
            self.alpha = alpha
            self.gamma = gamma
            self.optimizer = torch.optim.Adam
            self.criterion = smp.losses.FocalLoss(alpha=alpha, gamma=gamma, mode='binary')
            self.save_hyperparameters()
            
            # planet images have 7 channels, while Sentinel-1 images have 2 channels
            self.combined_in_channels = 9

            # unified encoder
            self.encoder_decoder = smp.Unet(
                  encoder_name='resnet34', 
                  in_channels=self.combined_in_channels, 
                  classes=1,
                  encoder_weights=None
            )

            # initialize class-wise metrics to compute
            self.accuracy_class = torchmetrics.Accuracy(task='multiclass',
                                                      average='none',
                                                      threshold=self.threshold,
                                                      num_classes=2)
            self.precision_class = torchmetrics.Precision(task='multiclass',
                                                      average='none',
                                                      threshold=self.threshold,
                                                      num_classes=2)
            self.recall_class = torchmetrics.Recall(task='multiclass',
                                                average='none',
                                                threshold=self.threshold,
                                                num_classes=2)
            self.f1_score_class = torchmetrics.F1Score(task='multiclass',
                                                average='none',
                                                threshold=self.threshold,
                                                num_classes=2)

      def forward(self, planet_input, s1_input):
            # concatenate the input images before sending them to the net
            combined_input = torch.cat([s1_input, planet_input], dim=1)
            x = self.encoder_decoder(combined_input)
            return x

      def training_step(self, batch, batch_idx):
            planet_input, s1_input, labels = batch
            labels = labels.unsqueeze(1).type_as(s1_input)
            outputs = self(planet_input, s1_input)
            loss = self.criterion(outputs, labels)
            self.log('train_loss', loss, on_step=False, on_epoch=True, 
                     prog_bar=True, sync_dist=True)
            return loss

      def validation_step(self, batch, batch_idx):
            planet_input, s1_input, labels = batch
            labels = labels.unsqueeze(1).type_as(s1_input)
            logits = self.forward(planet_input, s1_input)
            loss = self.criterion(logits, labels)
            probs = torch.sigmoid(logits) # convert logits to probabilities
            preds = (probs > self.threshold).float() # apply threshold to probrabilities
            f1_score_class = self.f1_score_class(preds.squeeze(1), labels.squeeze(1))
            self.log("val_loss", loss, prog_bar=True, on_step=False, 
                     on_epoch=True, sync_dist=True)
            self.log('val_f1score', f1_score_class[1], 
                     prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            
            return loss
      
      def test_step(self, batch, batch_idx):
            planet_input, s1_input, labels = batch
            labels = labels.unsqueeze(1).type_as(planet_input) 
            logits = self.forward(planet_input, s1_input)
            loss = self.criterion(logits, labels)
            probs = torch.sigmoid(logits) # convert logits to probabilities
            preds = (probs > self.threshold).float() # apply threshold to probrabilities
            
            # compute class-wise metrics
            accuracy_class = self.accuracy_class(preds.squeeze(1), labels.squeeze(1))
            precision_class = self.precision_class(preds.squeeze(1), labels.squeeze(1))
            recall_class = self.recall_class(preds.squeeze(1), labels.squeeze(1))
            f1_score_class = self.f1_score_class(preds.squeeze(1), labels.squeeze(1))

            # manually calculate macro metrics
            accuracy_macro_manual = accuracy_class.mean()
            precision_macro_manual = precision_class.mean()
            recall_macro_manual = recall_class.mean()
            f1_score_macro_manual = f1_score_class.mean()

            # log metrics
            self.log('test_accuracy_macro_manual', accuracy_macro_manual, sync_dist=True)
            self.log('test_accuracy_class0', accuracy_class[0], sync_dist=True)
            self.log('test_accuracy_class1', accuracy_class[1], sync_dist=True)

            self.log('test_precision_macro_manual', precision_macro_manual, sync_dist=True)
            self.log('test_precision_class0', precision_class[0], sync_dist=True)
            self.log('test_precision_class1', precision_class[1], sync_dist=True)

            self.log('test_recall_macro_manual', recall_macro_manual, sync_dist=True)
            self.log('test_recall_class0', recall_class[0], sync_dist=True)
            self.log('test_recall_class1', recall_class[1], sync_dist=True)

            self.log('test_f1score_macro_manual', f1_score_macro_manual, sync_dist=True)
            self.log('test_f1score_class0', f1_score_class[0], sync_dist=True)
            self.log('test_f1score_class1', f1_score_class[1], sync_dist=True)

            return loss

      def configure_optimizers(self):
            optimizer = self.optimizer(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            scheduler = StepLR(optimizer, step_size=10, gamma=0.8)
            return [optimizer], [scheduler]
