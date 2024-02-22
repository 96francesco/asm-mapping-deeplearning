import torch
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import colors
from matplotlib.colors import ListedColormap

def plot_segmentation_outputs(predictions_file: str, output_name: str, num_examples=3):
      # load saved predictions
      predictions = torch.load(predictions_file)

      minerals_dict = {0: 'Background',
                 1: 'Gold',
                 2: 'Cassiterite',
                 3: 'Coltan',
                 4: 'Copper',
                 5: 'Cobalt',
                 6: 'Diamond',
                 7: 'Sapphire',
                 8: 'Tourmaline',
                 9: 'Wolframite',
                 10: 'Unknown'}
      
      # create custom colormap
      colors = ['#000000', '#FFD700', '#7FfF00', '#0000FF',
            '#F0E68C', '#D2B48C', '#FF6347', '#9ACD32',
            '#FF00FF', '#800080', '#F0FFFF']
      custom_cmap = ListedColormap(colors)

      fig, axs = plt.subplots(num_examples, 3, figsize=(12, 4 * num_examples))

      # plot either num_examples or predictions if it is smaller
      for i in range(min(num_examples, len(predictions))):
            # unpack the tuple of the i-th prediction
            inputs, outputs, targets = predictions[i]

            # process input image for visualization
            input_img = inputs[0].cpu().numpy().transpose(1, 2, 0) # from PyTorch to NumPy
            input_img_rgb = input_img[:, :, [2, 1, 0]] # get the RGB bands
            input_img_rgb = (input_img_rgb - input_img_rgb.min()) / (input_img_rgb.max() - input_img_rgb.min())
            input_img_rgb = (input_img_rgb * 255).astype(np.uint8) # normalize for visualisation purposes

            # process ground truth and model output
            target_img = targets[0].cpu().numpy().squeeze()

            # apply softmax to output, convert to NumPy and detach from gradients
            output_img = torch.softmax(outputs[0].cpu(), dim=0).detach().numpy()
            
            # find class with highest probabily for each pixel
            output_img = np.argmax(output_img, axis=0)

            # calculate bin counts for target and output images
            target_counts = np.bincount(target_img.flatten())
            output_counts = np.bincount(output_img.flatten())

            # get sorted indices based on counts (descending order)
            target_indices_sorted = np.argsort(target_counts)[::-1]
            output_indices_sorted = np.argsort(output_counts)[::-1]

            # filter out classes with zero count to get existing classes
            existing_target_classes = [class_idx for class_idx in target_indices_sorted if target_counts[class_idx] > 0]
            existing_output_classes = [class_idx for class_idx in output_indices_sorted if output_counts[class_idx] > 0]

            # Get class names for existing classes
            target_class_names = ', '.join(minerals_dict[c] for c in existing_target_classes)
            output_class_names = ', '.join(minerals_dict[c] for c in existing_output_classes)

            # make plots
            axs[i, 0].imshow(input_img_rgb)
            axs[i, 0].set_title("Input image")
            axs[i, 1].imshow(output_img, cmap=custom_cmap, vmin=0, vmax=len(minerals_dict)-1)
            axs[i, 1].set_title("Model output")
            axs[i, 1].text(5, 5, f'Predicted classes: {output_class_names}', color='white', backgroundcolor='blue', fontsize=10, verticalalignment='top')
            axs[i, 2].imshow(target_img, cmap=custom_cmap, vmin=0, vmax=len(minerals_dict)-1)
            axs[i, 2].set_title("Ground truth")
            axs[i, 2].text(5, 15, f'Actual classes: {target_class_names}', color='white', backgroundcolor='blue', fontsize=10,
                        verticalalignment='center')

            for ax in axs[i]:
                  ax.axis("off")

      plt.tight_layout()
      plt.show()
      plt.savefig(f'reports/{output_name}.png')