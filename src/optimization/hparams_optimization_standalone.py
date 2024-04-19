# this is the optimization script for the Late Fusion models

import optuna
import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import gc

from data.fusion_dataset import FusionDataset  
from models.lit_model_fusion import LitModelLateFusion

from data.planet_dataset_normalization import linear_norm_global_minmax as planet_norm_minmax
from data.s1_dataset_normalization import global_standardization as s1_standardization


def objective(trial):
      # hyperparameters to be optimized
      lr = trial.suggest_float('lr', 1e-5, 1e-1)
      batch_size = trial.suggest_categorical('batch_size', [8, 16])
      weight_decay = trial.suggest_float('weight_decay', 1e-10, 1e-3)
      threshold = trial.suggest_float("threshold", 0.3, 0.7, step=0.05)
      fusion_loss = trial.suggest_categorical('fusion_loss', ['ce', 'focal', 'iou'])
      
      # Dataset preparation
      dataset = FusionDataset(root_dir='/mnt/guanabana/raid/home/pasan001/thesis/dataset/asm_dataset_split_0/fusion', 
                              train=True,
                              transforms=None,
                              planet_normalization=planet_norm_minmax,
                              s1_normalization=s1_standardization) 
      
      train_size = int(0.8 * len(dataset))
      val_size = len(dataset) - train_size
      train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

      train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
      val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

      # Model initialization
      model = LitModelLateFusion(
            s1_checkpoint='models/checkpoints/s1-db-trial7-epoch=55-val_f1score=0.48.ckpt',
            planet_checkpoint='models/checkpoints/planet-binary-optimized-trial43-epoch=59-val_f1score=0.74.ckpt',
            lr=lr,
            threshold=threshold,
            weight_decay=weight_decay,
            optimizer='adam',
            fusion_loss=fusion_loss
      )

      # Trainer setup
      trainer = pl.Trainer(max_epochs=50,
                              logger=False,
                              enable_progress_bar=False,
                              enable_checkpointing=True,
                              accelerator='gpu',
                              devices=2,
                              strategy=DDPStrategy(find_unused_parameters=True),
                              accumulate_grad_batches=4,
                              precision=16
                              )

      # Model training
      trainer.fit(model, train_loader, val_loader)

      # return validation loss and F1-score
      val_loss = trainer.callback_metrics["val_loss"].item()
      val_f1score = trainer.callback_metrics["val_f1score"].item()

      return val_loss, val_f1score

# clear CUDA cache
torch.cuda.empty_cache()
gc.collect()

# setup study
storage = optuna.storages.RDBStorage("sqlite:///src/optimization/optimization.db")
pruner = MedianPruner(n_startup_trials=5,
                      n_warmup_steps=10,
                      interval_steps=1)

study = optuna.create_study(directions=['minimize', 'maximize'],
                            sampler=TPESampler(seed=42),
                            pruner=pruner,
                            storage=storage,
                            study_name='lf_pretrained_streams',
                            load_if_exists=True)

# start study
study.optimize(objective, n_trials=20, gc_after_trial=True)
