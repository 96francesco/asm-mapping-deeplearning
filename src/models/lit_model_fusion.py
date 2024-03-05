import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torchmetrics
import torch.nn as nn

from torch.optim.lr_scheduler import StepLR

class LitModelBinaryLateFusion(pl.LightningModule):
      def __init__(self, planet_checkpoint=None, s1_checkpoint=None,
                 loss='ce', pos_weight=None, weight_decay=1e-5,
                 lr=1e-3, threshold=0.5, optimizer='sgd',
                  s1_in_channels=2, planet_in_channels=7):
            super().__init__()
            self.weight_decay = weight_decay
            self.lr = lr
            self.threshold = threshold
            self.s1_in_channels = s1_in_channels
            self.planet_in_channels = planet_in_channels
            self.save_hyperparameters

            # define two separate U-Net models for Sentinel-1 and Planet data
            if s1_checkpoint is not None:
                  # load Sentinel-1 model from checkpoint
                  self.unet_s1 = self.load_model_from_checkpoint(s1_checkpoint)
            else:
                  # initialize a new Sentinel-1 model
                  self.unet_s1 = smp.Unet(
                                    encoder_name="resnet34",
                                    decoder_use_batchnorm=True,
                                    decoder_attention_type='scse',
                                    encoder_weights=None,
                                    in_channels=self.s1_in_channels, 
                                    classes=1
                                    )
            
            if planet_checkpoint is not None:
                  # load Planet model from checkpoint
                  self.unet_planet = self.load_model_from_checkpoint(planet_checkpoint)
            else:
                  # initialize a new Planet model
                  self.unet_planet = smp.Unet(
                                          encoder_name="resnet34",
                                          decoder_use_batchnorm=True,
                                          decoder_attention_type='scse',
                                          encoder_weights=None,
                                          in_channels=self.planet_in_channels,
                                          classes=1)

            if optimizer == 'sgd':
                  self.optimizer = torch.optim.SGD
            elif optimizer == 'adam':
                  self.optimizer = torch.optim.Adam
            else:
                  raise ValueError(f'Unkwnown optimizer')
            
            if loss == 'ce':
                  if pos_weight is not None:
                        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to('cuda'))
                  else:
                        self.criterion = nn.BCEWithLogitsLoss()
            elif loss == 'focal':
                  self.criterion = smp.losses.FocalLoss(alpha=0.25, gamma=2.0, mode='binary')
            elif loss == 'iou':
                  self.criterion = smp.losses.JaccardLoss(mode='binary')
            else:
                  raise ValueError(f'Unkwnon loss function: {loss}')
            
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

            # define a fusion layer 
            self.fusion_conv = torch.nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1)
      
      def load_model_from_checkpoint(self, checkpoint_path, in_channels):
            model = smp.Unet(
                        encoder_name="resnet34",
                        in_channels=in_channels,
                        classes=1,
                        encoder_weights=None,
                        decoder_use_batchnorm=True,
                        decoder_attention_type='scse'
                  )
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint['state_dict'])

            return model

      def forward(self, s1_input, planet_input):
            # process inputs through their respective U-Ney streams
            s1_features = self.unet_s1(s1_input)
            planet_features = self.unet_planet(planet_input)

            # concatenate features along the channel dimension and apply fusion
            fused_features = torch.cat((s1_features, planet_features), dim=1)
            fused_output = self.fusion_conv(fused_features)

            return fused_output

      def training_step(self, batch, batch_idx):
            planet_input, s1_input, labels = batch
            outputs = self.forward(s1_input, planet_input)
            labels = labels.unsqueeze(1).type_as(s1_input) # add a channel dimension
            loss = self.criterion(outputs, labels)

            self.log('train_loss', loss, prog_bar=True, on_step=False,
                on_epoch=True, sync_dist=True)

            return loss

      def validation_step(self, batch, batch_idx):
            planet_input, s1_input, labels = batch
            outputs = self.forward(s1_input, planet_input)
            labels = labels.unsqueeze(1).type_as(s1_input)
            loss = self.criterion(outputs, labels)

            self.log("val_loss", loss, prog_bar=True, on_step=False,
                on_epoch=True, sync_dist=True)
            self.log('val_f1score', self.f1_score(outputs, labels),
                  prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

            return loss

      def test_step(self, batch, batch_idx):
            planet_input, s1_input, labels = batch
            logits = self.forward(s1_input, planet_input)
            labels = labels.unsqueeze(1).type_as(s1_input)
            loss = self.criterion(logits, labels)
            
            probs = torch.sigmoid(logits) # convert logits to probabilities
            preds = (probs > self.threshold).float() # apply threshold to probrabilities

            acc = self.accuracy(probs.squeeze(), labels.squeeze().long())

            # log loss and accuracy metrics
            self.log('test_loss', loss, sync_dist=True)
            self.log('accuracy', acc, sync_dist=True)
            self.log('precision', self.precision(preds, labels), sync_dist=True)
            self.log('recall', self.recall(preds, labels), sync_dist=True)
            self.log('f1_score', self.f1_score(preds, labels), sync_dist=True)

            return loss

      def configure_optimizers(self):
            # combine parameters from both models
            params = list(self.unet_s1.parameters()) + list(self.unet_planet.parameters())

            # initialize optimizer and scheduler
            optimizer = self.optimizer(params, lr=self.lr, weight_decay=self.weight_decay)
            scheduler = StepLR(optimizer, step_size=5, gamma=0.25)

            return [optimizer], [scheduler]
