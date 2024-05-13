import pytorch_lightning as pl
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import torchmetrics
import torch.nn.functional as F

from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from torch.optim.lr_scheduler import StepLR

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
      def __init__(self, pretrained_streams=False,
                  planet_checkpoint=None, s1_checkpoint=None,
                  s1_in_channels=2, planet_in_channels=7,
                  lr=1e-3, threshold=0.5, weight_decay=1e-5, 
                  alpha=0.25, gamma=2.0):
            super().__init__()
            self.lr = lr
            self.threshold = threshold
            self.weight_decay = weight_decay
            self.alpha = alpha
            self.gamma = gamma
            self.optimizer = torch.optim.Adam
            self.criterion = smp.losses.FocalLoss(alpha=alpha, gamma=gamma, mode='binary')

            self.save_hyperparameters()
            
            # initialize encoders, with pretrained streams or to be trained from scratch
            if pretrained_streams and planet_checkpoint and s1_checkpoint:
                  # load pretrained streams
                  self.s1_encoder = self._load_encoder(s1_checkpoint, s1_in_channels)
                  self.planet_encoder = self._load_encoder(planet_checkpoint, planet_in_channels)
            else:
                  # initialize from scratch
                  self.s1_encoder = smp.encoders.get_encoder('resnet34', 
                                                             in_channels=s1_in_channels)
                  self.planet_encoder = smp.encoders.get_encoder('resnet34',
                                                                  in_channels=planet_in_channels)

            # setup decoder and segmentation head
            encoder_channels = [128, 128, 256, 512, 1024]
            decoder_channels=[1024, 512, 256, 128, 128]
            self.decoder = UnetDecoder(
                  encoder_channels=encoder_channels,
                  decoder_channels=decoder_channels,
                  n_blocks=len(decoder_channels),
                  use_batchnorm=True,
                  attention_type='scse',
                  center=False
            )
            self.segmentation_head = nn.Sequential(
                  nn.Conv2d(decoder_channels[-1], 1, kernel_size=1),
                  
                  # this upsampling is crucial to match the shape of the original input
                  # this LF model is not able to upscale the input to the original size automatically
                  nn.Upsample(size=(384, 384), mode='bilinear', align_corners=True)
            )

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
            # extract features from both encoders
            s1_features = self.s1_encoder(s1_input)
            planet_features = self.planet_encoder(planet_input)

            # combine features, skipping the first feature map (initial channel size)
            combined_features = [torch.cat([s1, planet], dim=1) for s1, planet in zip(s1_features[1:], planet_features[1:])] 

            # send combined feature to decoder
            x = self.decoder(*combined_features)
            x = self.segmentation_head(x)
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
            outputs = self(planet_input, s1_input)
            loss = self.criterion(outputs, labels)
            self.log("val_loss", loss, prog_bar=True, on_step=False,
                on_epoch=True, sync_dist=True)
            self.log('val_f1score', self.f1_score(outputs, labels),
                        prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            return loss
      
      def test_step(self, batch, batch_idx):
            planet_input, s1_input, labels = batch
            labels = labels.unsqueeze(1).type_as(planet_input) 
            logits = self.forward(planet_input, s1_input)
            loss = self.criterion(logits, labels)
            probs = torch.sigmoid(logits) # convert logits to probabilities
            preds = (probs > self.threshold).float() # apply threshold to probrabilities
            
            # compute and log accuracy metrics
            acc = self.accuracy(probs.squeeze(), labels.squeeze().long())
            self.log('test_accuracy', acc, sync_dist=True)
            self.log('test_precision', self.precision(preds, labels), sync_dist=True)
            self.log('test_recall', self.recall(preds, labels), sync_dist=True)
            self.log('test_f1score', self.f1_score(preds, labels), sync_dist=True)

            return loss

      def configure_optimizers(self):
            params = list(self.s1_encoder.parameters()) + list(self.planet_encoder.parameters()) + list(self.decoder.parameters()) + list(self.segmentation_head.parameters())
            optimizer = self.optimizer(params, lr=self.lr, weight_decay=self.weight_decay)
            scheduler = StepLR(optimizer, step_size=10, gamma=0.8)
            return [optimizer], [scheduler]

      def _load_encoder(self, checkpoint_path, in_channels):
            print("Loading encoder from checkpoint:", checkpoint_path)
            model = smp.Unet(encoder_name='resnet34', in_channels=in_channels, classes=1)
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # get the state dict of the encoder from the checkpoint
            encoder_state_dict = {key.replace('model.', ''): value for key, value in checkpoint['state_dict'].items() if key.startswith('model.encoder')}
            
            # load the state dict into the encoder of the model
            model.encoder.load_state_dict(encoder_state_dict, strict=False)

            return model.encoder 

