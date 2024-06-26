import pytorch_lightning as pl
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import torchmetrics

from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from torch.optim.lr_scheduler import StepLR

class LitModelLateFusion(pl.LightningModule):
      """
      A PyTorch Lightning module for a Late Fusion U-Net. This model fuses features from 
      two input streams via a concatenation operation after the encoder stage, and then sends
      the combined features to the decoder.

      Attributes:
            lr (float): Learning rate for the optimizer.
            weight_decay (float): Weight decay for the optimizer.
            alpha (float): Alpha parameter for the Focal Loss function.
            gamma (float): Gamma parameter for the Focal Loss function.
            pretrained_streams (bool): Whether to use pretrained streams for the model.
            planet_checkpoint (str): Path to the checkpoint for the Planet stream model.
            s1_checkpoint (str): Path to the checkpoint for the Sentinel-1 stream model.
            accuracy (torchmetrics.Accuracy): Metric for calculating the accuracy of predictions.
            precision (torchmetrics.Precision): Metric for calculating the precision of predictions.
            recall (torchmetrics.Recall): Metric for calculating the recall of predictions.
            f1_score (torchmetrics.F1Score): Metric for calculating the F1 score of predictions.
            threshold (float): Threshold value for converting logits to binary outputs.
            planet_in_channels (int): Number of input channels for the Planet stream model.
            s1_in_channels (int): Number of input channels for the Sentinel-1 stream model.
      
      Methods:
            forward(planet_input, s1_input): Defines the forward pass of the module by fusing features from
                  both input streams.
            training_step(batch, batch_idx): Processes a single batch during training.
            validation_step(batch, batch_idx): Processes a single batch during validation.
            test_step(batch, batch_idx): Processes a single batch during testing, calculates and logs metrics.
            configure_optimizers(): Configures and returns the model's optimizers and learning rate schedulers.
            _load_encoder(checkpoint_path, in_channels): Loads the encoder from a given checkpoint path.
      """
      def __init__(self, pretrained_streams=False,
                  planet_checkpoint=None, s1_checkpoint=None,
                  s1_in_channels=2, planet_in_channels=7,
                  lr=1e-3, threshold=0.5, weight_decay=1e-5, 
                  alpha=0.25, gamma=2.0, is_inference=False):
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
            # encoder_channels = [64, 64, 128, 256, 512]
            # decoder_channels = [512, 256, 128, 64, 64]
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

            if is_inference:
                  size = (1024, 1024)
            else:
                  size = (384, 384)
            self.segmentation_head = nn.Sequential(
                  nn.Conv2d(decoder_channels[-1], 1, kernel_size=1),
                  
                  # this upsampling is crucial to match the shape of the original input
                  # this LF model is not able to upscale the input to the original size automatically
                  nn.Upsample(size=size, mode='bilinear', align_corners=True)
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
            # extract features from both encoders
            s1_features = self.s1_encoder(s1_input)
            planet_features = self.planet_encoder(planet_input)

            # combine features via summation, skipping the first feature map (initial channel size)
            # combined_features = [s1 + planet for s1, planet in zip(s1_features[1:], planet_features[1:])] 
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

