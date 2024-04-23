import pytorch_lightning as pl
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import torchmetrics

class LitModelLateFusion(pl.LightningModule):
      """
      A PyTorch Lightning module for late fusion of models trained on different data streams.
      Specifically designed for fusing models trained separately on Sentinel-1 and Planet satellite
      image data. Supports various loss functions and optimizers for training the fused model.

      Attributes:
            s1_model (torch.nn.Module): stream for processing Sentinel-1 satellite images.
            planet_model (torch.nn.Module): stream for processing Planet satellite images.
            fusion_conv (torch.nn.Conv2d): Convolutional layer to fuse features from s1_model and planet_model.
            s1_loss (torch.nn.modules.loss._Loss): Loss function used for training the s1_model.
            planet_loss (torch.nn.modules.loss._Loss): Loss function used for training the planet_model.
            fusion_criterion (torch.nn.modules.loss._Loss): Loss function used for the fused output.
            accuracy (torchmetrics.Accuracy): Metric for calculating the accuracy of predictions.
            precision (torchmetrics.Precision): Metric for calculating the precision of predictions.
            recall (torchmetrics.Recall): Metric for calculating the recall of predictions.
            f1_score (torchmetrics.F1Score): Metric for calculating the F1 score of predictions.
            threshold (float): Threshold value for converting logits to binary outputs.
            optimizer (torch.optim.Optimizer): Optimizer used for training the model.
      
      Methods:
            forward(planet_input, s1_input): Defines the forward pass of the module by fusing features from
                  both input streams and applying the fusion_conv layer.
            training_step(batch, batch_idx): Processes a single batch during training.
            validation_step(batch, batch_idx): Processes a single batch during validation.
            test_step(batch, batch_idx): Processes a single batch during testing, calculates and logs metrics.
            configure_optimizers(): Configures and returns the model's optimizers and learning rate schedulers.
            load_model_from_checkpoint(checkpoint_path, in_channels): Utility method for loading a model
                  from a specified checkpoint.
      """
      def __init__(self, planet_checkpoint=None, s1_checkpoint=None,
                  fusion_loss='ce', lr=1e-3, threshold=0.5, optimizer='adam',
                  s1_in_channels=2, planet_in_channels=7, weight_decay=1e-5, pos_weight=None):
            super().__init__()
            self.save_hyperparameters()
            self.threshold = threshold

            # stream-specific models
            if s1_checkpoint is not None:
                  # load Sentinel-1 model from checkpoint
                  self.s1_model = self.load_model_from_checkpoint(s1_checkpoint, 
                                                                  self.hparams.s1_in_channels)
            else:
                  # initialize a new Sentinel-1 model
                  self.s1_model = smp.Unet(
                  encoder_name="resnet34",
                  in_channels=self.hparams.s1_in_channels,
                  classes=1,
                  encoder_weights=None,
                  decoder_use_batchnorm=True,
                  decoder_attention_type='scse'
                  )
            
            if planet_checkpoint is not None:
                  self.planet_model = self.load_model_from_checkpoint(planet_checkpoint, 
                                                                      self.hparams.planet_in_channels)
            else:
                  self.planet_model = smp.Unet(
                  encoder_name="resnet34",
                  in_channels=self.hparams.planet_in_channels,
                  classes=1,
                  encoder_weights=None,
                  decoder_use_batchnorm=True,
                  decoder_attention_type='scse'
            )

            # loss for the single streams
            self.s1_loss = smp.losses.JaccardLoss(mode='binary')
            self.planet_loss = smp.losses.FocalLoss(alpha=0.25, gamma=2.0, mode='binary')

            # define fusion layer
            self.fusion_conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1)

            # define loss function
            if fusion_loss == 'ce':
                  if pos_weight is not None:
                        pos_weight_tensor = torch.tensor(pos_weight).to(self.device)
                        self.fusion_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
                  else:
                        self.fusion_criterion = nn.BCEWithLogitsLoss()
            elif fusion_loss == 'focal':
                  self.fusion_criterion = smp.losses.FocalLoss(alpha=0.25, gamma=2.0, mode='binary')
            elif fusion_loss == 'iou':
                  self.fusion_criterion = smp.losses.JaccardLoss(mode='binary')
            else:
                  raise ValueError(f'Unknown loss function: {fusion_loss}')

            # define optimizer
            if optimizer == 'adam':
                  self.optimizer = torch.optim.Adam
            elif optimizer == 'sgd':
                  self.optimizer = torch.optim.SGD
            else:
                  raise ValueError(f'Unknown optimizer: {optimizer}')

            # define metrics to compute
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

      def forward(self, planet_input, s1_input):
            s1_features = self.s1_model(s1_input)
            planet_features = self.planet_model(planet_input)
            fused_features = torch.cat((s1_features, planet_features), dim=1)
            fused_output = self.fusion_conv(fused_features)
            return fused_output

      def training_step(self, batch, batch_idx):
            planet_input, s1_input, labels = batch
            labels = labels.unsqueeze(1).type_as(s1_input)
            fused_output = self.forward(planet_input, s1_input)
            loss = self.fusion_criterion(fused_output, labels)

            self.log('train_loss', loss, on_step=False, on_epoch=True, 
                     prog_bar=True, sync_dist=True)
            return loss

      def validation_step(self, batch, batch_idx):
            planet_input, s1_input, labels = batch
            labels = labels.unsqueeze(1).type_as(s1_input)
            fused_output = self.forward(planet_input, s1_input)
            loss = self.fusion_criterion(fused_output, labels)

            self.log("val_loss", loss, prog_bar=True, on_step=False,
                on_epoch=True, sync_dist=True)
            self.log('val_f1score', self.f1_score(fused_output, labels),
                        prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
      
      def test_step(self, batch, batch_idx):
            planet_input, s1_input, labels = batch
            labels = labels.unsqueeze(1).type_as(planet_input) 
            logits = self.forward(planet_input, s1_input)
            loss = self.fusion_criterion(logits, labels)

            probs = torch.sigmoid(logits) # convert logits to probabilities
            preds = (probs > self.threshold).float() # apply threshold to probrabilities

            acc = self.accuracy(probs.squeeze(), labels.squeeze().long())

            self.log('test_accuracy', acc, sync_dist=True)
            self.log('test_precision', self.precision(preds, labels), sync_dist=True)
            self.log('test_recall', self.recall(preds, labels), sync_dist=True)
            self.log('test_f1score', self.f1_score(preds, labels), sync_dist=True)

            return

      def configure_optimizers(self):
                  params = list(self.s1_model.parameters()) + list(self.planet_model.parameters()) + list(self.fusion_conv.parameters())
                  if self.hparams.optimizer == 'adam':
                        optimizer = torch.optim.Adam(params, lr=self.hparams.lr, 
                                                     weight_decay=self.hparams.weight_decay)
                  elif self.hparams.optimizer == 'sgd':
                        optimizer = torch.optim.SGD(params, lr=self.hparams.lr, 
                                                    weight_decay=self.hparams.weight_decay)
                  else:
                        raise ValueError(f'Unknown optimizer: {self.hparams.optimizer}')
                  
                  # define scheduler
                  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.25)
                  
                  return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

      def load_model_from_checkpoint(self, checkpoint_path, in_channels):
            model = smp.Unet(encoder_name="resnet34", 
                             in_channels=in_channels, 
                             classes=1, 
                             decoder_attention_type='scse')
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            adjusted_state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}
            model.load_state_dict(adjusted_state_dict)
            return model

