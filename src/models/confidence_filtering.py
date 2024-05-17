import torch
import matplotlib.pyplot as plt
import torchmetrics
import gc

from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader

from data.planet_dataset_normalization import linear_norm_global_percentile as planet_norm_percentile
from data.planet_dataset_normalization import linear_norm_global_minmax as planet_norm_minmax
from data.planet_dataset_normalization import global_standardization as planet_standardization

from data.s1_dataset_normalization import global_standardization as s1_standardization
from data.s1_dataset_normalization import linear_norm_global_minmax as s1_norm_minmax
from data.s1_dataset_normalization import linear_norm_global_percentile as s1_norm_percentile

from data.fusion_dataset import FusionDataset
from models.lit_model_fusion import LitModelLateFusion


def confidence_filtering(model, test_loader, device, model_threshold, confidence_thresholds):
    # set the model to evaluation mode and freeze weights
    model.eval()
    model.freeze()
    model.to(device)

    # initialize a dictionary for storing F1 scores for each threshold
    f1_scores_dict = {thr.item(): [] for thr in confidence_thresholds}
    valid_preds_count_dict = {thr.item(): 0 for thr in confidence_thresholds}

    # initialize the F1 score metric for binary classification
    f1_score = torchmetrics.F1Score(task='binary', average='macro').to(device)

    total_predictions = 0

    with torch.no_grad():
        for data in test_loader:
            planet_data, s1_data, labels = data
            planet_data, s1_data, labels = planet_data.to(device), s1_data.to(device), labels.to(device)
            logits = model(planet_data, s1_data)  # forward pass to get logits
            probabilities = torch.sigmoid(logits)  # convert logits to probabilities

            # shape labels correctly
            labels = labels.unsqueeze(1)
            total_predictions += labels.numel()

            # flatten prob and labels tensors to handle them more easily
            probabilities_flat = probabilities.view(-1)
            labels_flat = labels.view(-1)

            for threshold in confidence_thresholds:
                # apply classification threshold to get binary predictions
                predictions_flat = (probabilities_flat > model_threshold).float()

                # create a mask where predictions meet the current confidence threshold
                valid_mask = (probabilities_flat >= threshold).float()

                # use the mask to filter out predictions below confidence threshold
                valid_predictions = predictions_flat[valid_mask > 0]
                valid_labels = labels_flat[valid_mask > 0]

                # check if there are any valid predictions
                if valid_predictions.numel() > 0:
                    f1_score.update(valid_predictions, valid_labels.int())
                    score = f1_score.compute()
                    f1_scores_dict[threshold.item()].append(score.item())
                    f1_score.reset()  # reset metric for the next iteration

                    valid_preds_count_dict[threshold.item()] += valid_predictions.numel()

    # average F1 scores for each threshold across all the batches of the data loader
    f1_scores_avg = [(thr, sum(scores) / len(scores)) for thr, scores in f1_scores_dict.items() if scores]

    # calculate percentage of remaining predictions for each threshold
    valid_preds_percent = [(thr, (count / total_predictions) * 100) for thr, count in valid_preds_count_dict.items()]

    return f1_scores_avg, valid_preds_percent

# set seed for reproducibility
seed_everything(42, workers=True)

# clear CUDA cache
torch.cuda.empty_cache()
gc.collect()

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the model checkpoint
checkpoint_path = "models/checkpoints/fusion_pretrained_trial11-epoch=43-val_f1score=0.84.ckpt"
model = LitModelLateFusion.load_from_checkpoint(checkpoint_path=checkpoint_path)

# initialize dataset and dataloader
dataset_dir = "/mnt/guanabana/raid/home/pasan001/asm-mapping-deeplearning/data/split_0/fusion/testing_data"
dataset = FusionDataset(dataset_dir,
                        planet_normalization=planet_norm_percentile,
                        s1_normalization=s1_norm_percentile)
test_loader = DataLoader(dataset, batch_size=32, shuffle=False)

# define confidence thresholds
confidence_thresholds = torch.linspace(0.0, 1.0, steps=18).to(device)  # from 0% to 100% with 5% steps

# run computation
f1_scores_avg, valid_preds_percent = confidence_filtering(model, test_loader, device, 0.4,
                                                          confidence_thresholds)

# Ensure lengths match
thresholds = [x[0] for x in f1_scores_avg]
f1_scores = [x[1] for x in f1_scores_avg]
percent_included = [x[1] for x in valid_preds_percent if x[0] in thresholds]

# plot F1 score vs confidence level
fig, ax1 = plt.subplots()
ax1.plot(thresholds, f1_scores, 'b')
ax1.set_xlabel('Classification confidence', fontsize=14)
ax1.set_ylabel('Macro F1 score', fontsize=14)
ax1.set_ylim(0.7, 1)
ax1.set_xlim(0.0, 1)
ax1.set_xticks([i/10 for i in range(11)])  
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
plt.grid(True)
plt.savefig(f'reports/figures/confidence_filtering.png')

# plot F1 score vs percentage of remaining predictions
fig, ax2 = plt.subplots()
ax2.plot(percent_included, f1_scores, 'r')
ax2.set_xlabel('Predictions included (%)', fontsize=14)
ax2.set_ylabel('Macro F1 score', fontsize=14)
ax1.set_ylim(0.7, 1.0)
ax1.set_xlim(0.0, 100.0)
ax2.set_xticks([i for i in range(0, 101, 10)])
ax2.set_yticks([i/20 for i in range(14, 21)])
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
plt.grid(True)
plt.savefig(f'reports/figures/remaining_preds.png')
