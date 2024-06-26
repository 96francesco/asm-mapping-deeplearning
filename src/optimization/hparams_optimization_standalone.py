# import libraries
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import optuna
import torch
import gc

from torch.utils.data import DataLoader, random_split
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning import seed_everything
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# import custom modules
from data.planet_dataset import PlanetDataset
from data.s1_dataset import Sentinel1Dataset

from data.planet_dataset_normalization import linear_norm_global_percentile as planet_norm_percentile
from data.planet_dataset_normalization import linear_norm_global_minmax as planet_norm_minmax
from data.planet_dataset_normalization import global_standardization as planet_standardization

from data.s1_dataset_normalization import global_standardization as s1_standardization
from data.s1_dataset_normalization import linear_norm_global_minmax as s1_norm_minmax
from data.s1_dataset_normalization import linear_norm_global_percentile as s1_norm_percentile

from models.lit_model_standalone import LitModelStandalone

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
        normalization = trial.suggest_categorical('normalization', 
                                                  ['minmax', 'percentile', 'standardization'])
        
        # select normalization function (Planet)
        if normalization == 'minmax':
            normalization_function = s1_norm_minmax
        elif normalization == 'percentile':
            normalization_function = s1_norm_percentile
        elif normalization == 'standardization':    
            normalization_function = s1_standardization

        # select normalization function (Sentinel-1)
        # if normalization == 'minmax':
        #     normalization_function = planet_norm_minmax
        # elif normalization == 'percentile':
        #     normalization_function = planet_norm_percentile
        # elif normalization == 'standardization':    
        #     normalization_function = planet_standardization

        # initialize sample dataset
        training_dir = '/mnt/guanabana/raid/home/pasan001/asm-mapping-deeplearning/data/split_0/s1/training_data'
        # training_dataset = PlanetDataset(training_dir,
        #                                 pad=True,
        #                                 normalization=normalization_function,
        #                                 transforms=True)
        training_dataset = Sentinel1Dataset(training_dir,
                                            pad=True,
                                            normalization=normalization_function,
                                            to_linear=False,
                                            transforms=True)

        # extract validation subset from the training set
        total_size = len(training_dataset)
        train_size = int(0.9 * total_size)
        val_size = total_size - train_size
        train_set, val_set = random_split(training_dataset, [train_size, val_size])

        # initialize data loaders
        train_loader = DataLoader(train_set, batch_size=batch_size,
                                        shuffle=True, num_workers=9,
                                        persistent_workers=True, pin_memory=True,
                                        prefetch_factor=2)
        val_loader = DataLoader(val_set, batch_size=batch_size, 
                                shuffle=False, num_workers=9, 
                                persistent_workers=True, pin_memory=True,
                                prefetch_factor=2)

        # initialize model instance
        n_channels = 2 # 7 for Planet imagery, 2 for Sentinel-1

        unet = smp.Unet(
        encoder_name="resnet34",
        decoder_use_batchnorm=True,
        decoder_attention_type='scse',
        encoder_weights=None,
        in_channels=n_channels, 
        classes=1,
        activation=None
        )

        epochs = 50
        model = LitModelStandalone(model=unet, 
                                weight_decay=weight_decay,
                                lr=lr,
                                threshold=threshold,
                                alpha=alpha,
                                gamma=gamma)

        # define trainer
        trainer = pl.Trainer(max_epochs=epochs,
                                logger=False,
                                enable_progress_bar=False,
                                enable_checkpointing=True,
                                accelerator='gpu',
                                devices='2',
                                strategy=DDPStrategy(find_unused_parameters=True),
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
                            study_name='s1_resnet34',
                            load_if_exists=True)

# start study
study.optimize(objective, n_trials=10, gc_after_trial=True)