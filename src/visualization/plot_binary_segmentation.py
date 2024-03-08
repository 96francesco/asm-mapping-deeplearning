import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_segmentation_outputs(predictions_file: str, output_name: str, is_optical=True, 
                              num_examples=3, original_dimensions=(180, 180)):
      # load saved predictions
      predictions = torch.load(predictions_file)

      fig, axs = plt.subplots(num_examples, 3, figsize=(15, 5 * num_examples))

      for i in range(min(num_examples, len(predictions))):
            inputs, outputs, targets = predictions[i]

            if is_optical:
                  # reorder the bands from B, G, R, NIR to R, G, B for RGB natural color visualization
                  input_img = inputs[0].cpu().numpy().transpose(1, 2, 0)
                  input_img = input_img[:, :, [2, 1, 0]]

                  # normalize for visualization purposes
                  input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min())
                  input_img = (input_img * 255).astype(np.uint8)
            else:
                  input_img = inputs[0][0].cpu().numpy()

            # convert target to numpy array
            target_img = targets[0].cpu().numpy()

            # apply sigmoid to the output logits to get probabilities
            probs = torch.sigmoid(outputs[0].cpu()).detach().numpy()

            # apply threshold to get binary mask
            output_img = (probs > 0.5).squeeze().astype(np.uint8)

            # crop the images to the original dimensions
            original_height, original_width = original_dimensions
            input_img = input_img[:original_height, :original_width]
            target_img = target_img[:original_height, :original_width]
            output_img = output_img[:original_height, :original_width]
            
            # make plot
            if is_optical:
                  axs[i, 0].imshow(input_img)
                  axs[i, 0].set_title("Input Image", fontsize=20)
            else:
                  axs[i, 0].imshow(input_img, cmap='gray')
                  axs[i, 0].set_title("Input Image", fontsize=20)

            axs[i, 1].imshow(output_img, cmap='gray')
            axs[i, 1].set_title("Model Segmentation", fontsize=20)
            axs[i, 2].imshow(target_img, cmap='gray')
            axs[i, 2].set_title("Ground Truth", fontsize=20)

            for ax in axs[i]:
                  ax.axis("off")

      plt.tight_layout()
      plt.show()
      plt.savefig(f'reports/{output_name}.png')