# this is the optimization script for the Late Fusion models

import optuna
import torch
import pytorch_lightning as pl
import gc

from torch.utils.data import DataLoader, random_split
from pytorch_lightning import seed_everything
from pytorch_lightning.strategies import DDPStrategy
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from data.fusion_dataset import FusionDataset  
from models.lit_model_fusion import LitModelLateFusion

from data.planet_dataset_normalization import linear_norm_global_percentile as planet_norm_percentile
from data.planet_dataset_normalization import linear_norm_global_minmax as planet_norm_minmax
from data.planet_dataset_normalization import global_standardization as planet_standardization

from data.s1_dataset_normalization import global_standardization as s1_standardization
from data.s1_dataset_normalization import linear_norm_global_minmax as s1_norm_minmax
from data.s1_dataset_normalization import linear_norm_global_percentile as s1_norm_percentile

def objective(trial):
      # set seed for reproducibility
      seed_everything(42, workers=True)

      # parameters to optimize
      weight_decay = trial.suggest_float('weight_decay', 1e-10, 1e-3, log=True)
      batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
      lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
      threshold = trial.suggest_float("threshold", 0.3, 0.7, step=0.05)
      alpha = trial.suggest_float("alpha", 0.05, 0.75, step=0.05)
      gamma = trial.suggest_float("gamma", 0, 5.0, step=1.0)
      normalization_s1 = trial.suggest_categorical('normalization_s1', 
                                                ['minmax', 'percentile', 'standardization'])
      normalization_planet = trial.suggest_categorical('normalization_planet',
                                                ['minmax', 'percentile', 'standardization'])

      # select normalization function (Sentinel-1)
      if normalization_s1 == 'minmax':
            normalization_s1_function = s1_norm_minmax
      elif normalization_s1 == 'percentile':
            normalization_s1_function = s1_norm_percentile
      elif normalization_s1 == 'standardization':    
            normalization_s1_function = s1_standardization
      
      # select normalization function (Planet)
      if normalization_planet == 'minmax':
            normalization_planet_function = planet_norm_minmax
      elif normalization_planet == 'percentile':
            normalization_planet_function = planet_norm_percentile
      elif normalization_planet == 'standardization':    
            normalization_planet_function = planet_standardization

      # initialize dataset
      dataset = FusionDataset(root_dir='/mnt/guanabana/raid/home/pasan001/asm-mapping-deeplearning/data/split_0/fusion/training_data', 
                              planet_normalization=normalization_planet_function,
                              s1_normalization=normalization_s1_function) 
      
      train_size = int(0.9 * len(dataset))
      val_size = len(dataset) - train_size
      train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

      # initialize data loaders
      train_loader = DataLoader(train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              num_workers=9,
                              persistent_workers=True, 
                              pin_memory=True,
                              prefetch_factor=2)
      val_loader = DataLoader(val_dataset, 
                              batch_size=batch_size, 
                              shuffle=False, 
                              num_workers=9,
                              persistent_workers=True, 
                              pin_memory=True,
                              prefetch_factor=2)

      # model initialization
      model = LitModelLateFusion(
            pretrained_streams=False,
            s1_checkpoint=None,
            planet_checkpoint=None,
            lr=lr,
            threshold=threshold,
            weight_decay=weight_decay,
            alpha=alpha,
            gamma=gamma
      )

      # define trainer
      trainer = pl.Trainer(max_epochs=50,
                              logger=False,
                              enable_progress_bar=False,
                              enable_checkpointing=True,
                              accelerator='gpu',
                              devices=2,
                              strategy=DDPStrategy(find_unused_parameters=True),
                              # accumulate_grad_batches=4,
                              # precision=16
                              )

      # fit model
      trainer.fit(model, train_loader, val_loader)

      # return validation loss and F1-score
      val_loss = trainer.callback_metrics["val_loss"].item()
      val_f1score = trainer.callback_metrics["val_f1score"].item()

      return val_loss, val_f1score

# clear CUDA cache
torch.cuda.empty_cache()
gc.collect()

# setup study
storage = optuna.storages.RDBStorage("sqlite:///reports/optimization.db")
pruner = MedianPruner(n_startup_trials=5,
                      n_warmup_steps=10,
                      interval_steps=1)

study = optuna.create_study(directions=['minimize', 'maximize'],
                            sampler=TPESampler(seed=42),
                            pruner=pruner,
                            storage=storage,
                            study_name='lf_from_scratch',
                            load_if_exists=True)

# start study
study.optimize(objective, n_trials=50, gc_after_trial=True)
