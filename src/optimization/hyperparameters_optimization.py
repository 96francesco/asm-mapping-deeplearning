# import libraries
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import optuna
import torch

from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.strategies import DDPStrategy
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# import custom modules
from data.planet_dataset import PlanetDataset
from models.lit_model_binary import LitModelBinary
from models.lit_model_multiclass import LitModelMulticlass
from data.normalization_functions import linear_norm_global_minmax, linear_norm_global_percentile, global_standardization


def objective(trial):
        # parameters to optimize
        loss_function = trial.suggest_categorical('loss', ['iou', 'ce', 'focal'])
        weight_decay = trial.suggest_float('weight_decay', 1e-10, 1e-3)
        batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
        class_weight = trial.suggest_categorical('pos_weight', [None, 'weights'])
        optimizer = trial.suggest_categorical('optimizer', ['adam', 'sgd'])
        lr = trial.suggest_float("lr", 1e-5, 1e-1)
        threshold = trial.suggest_float("threshold", 0.3, 0.7, step=0.05)
        normalization = trial.suggest_categorical('normalization', 
                                                  ['minmax', 'percentile', 'standardization'])
        
        # create class weights
        if class_weight == 'weights':
            class_weight = torch.tensor([133822248 / 4196568])
                                   
        # select normalization function
        if normalization == 'minmax':
            normalization_function = linear_norm_global_minmax
        elif normalization == 'percentile':
            normalization_function = linear_norm_global_percentile
        elif normalization == 'standardization':    
            normalization_function = global_standardization

        # initialize sample dataset
        # training_dir = '/mnt/guanabana/raid/home/pasan001/thesis/dataset/asm_dataset_split_0/planet/multiclass/training_data'
        training_dir = '/mnt/guanabana/raid/home/pasan001/thesis/dataset/asm_dataset_split_0/planet/binary/training_data'

        training_dataset = PlanetDataset(training_dir,
                                        pad=True,
                                        normalization=normalization_function)

        # extract validation subset from the training set
        total_size = len(training_dataset)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size
        train_set, val_set = random_split(training_dataset, [train_size, val_size])

        # initialize data loaders
        train_loader = DataLoader(train_set, batch_size=batch_size,
                                        shuffle=True, num_workers=4,
                                        persistent_workers=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                                num_workers=4, persistent_workers=True)

        # initialize model instance
        unet = smp.Unet(
        encoder_name="resnet34",
        decoder_use_batchnorm=True,
        decoder_attention_type='scse',
        encoder_weights=None,
        in_channels=7,
        classes=1,
        activation=None
        )

        epochs = 50

        # early_stop_callback = EarlyStopping(
        #     monitor='val_loss',
        #     min_delta=0.00,
        #     patience=10, 
        #     verbose=False, # disable for the study
        #     mode='min')

        # model = LitModelMulticlass(model=unet, 
        #                                 loss=loss_function, 
        #                                 weight_decay=weight_decay,
        #                                 optimizer=optimizer, 
        #                                 lr=lr,
        #                                 pos_weight=None)
        
        model = LitModelBinary(model=unet, 
                                loss=loss_function, 
                                weight_decay=weight_decay,
                                optimizer=optimizer, 
                                lr=lr,
                                pos_weight=class_weight,
                                threshold=threshold)

        # define trainer
        trainer = pl.Trainer(max_epochs=epochs,
                                logger=False,
                                enable_progress_bar=False,
                                enable_checkpointing=True,
                                # callbacks=[early_stop_callback],
                                accelerator='gpu',
                                devices='2',
                                strategy=DDPStrategy(find_unused_parameters=True)
                                )
        # fit model
        trainer.fit(model, train_loader, val_loader)

        # return validation loss and F1-score
        val_loss = trainer.callback_metrics["val_loss"].item()
        val_f1score = trainer.callback_metrics["val_f1score"].item()

        return val_loss, val_f1score

# setup study
storage = optuna.storages.RDBStorage("sqlite:///src/optimization/binary_optimization.db")
pruner = MedianPruner(n_startup_trials=5,
                      n_warmup_steps=10,
                      interval_steps=1)
study = optuna.create_study(directions=['minimize', 'maximize'],
                            sampler=TPESampler(seed=42),
                            pruner=pruner,
                            storage=storage,
                            study_name='unet_binary_multiobj_study',
                            load_if_exists=True)

# start study
study.optimize(objective, n_trials=50, gc_after_trial=True)

# print best parameters
best_params = study.best_params
print("Best parameters:", best_params)
