import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_segmentation_outputs(predictions_file: str, output_name: str, is_optical=True,
                               is_fusion=False, num_examples=3, original_dimensions=(375, 375),
                               threshold=0.5):
    
      predictions = torch.load(predictions_file)
      original_height, original_width = original_dimensions
      
      subplot_cols = 4 if is_fusion else 3
      fig, axs = plt.subplots(num_examples, subplot_cols, figsize=(20, 5 * num_examples))
      
      for i in range(min(num_examples, len(predictions))):
            if is_fusion:
                  planet_input, s1_input, outputs, targets = predictions[i]
                  
                  # process Planet input
                  planet_img = planet_input[0].cpu().numpy().transpose(1, 2, 0)
                  planet_img = planet_img[:, :, [2, 1, 0]]  # Convert BGR to RGB
                  planet_img = (planet_img - planet_img.min()) / (planet_img.max() - planet_img.min())
                  planet_img = (planet_img * 255).astype(np.uint8)

                  planet_img = planet_img[:original_height, :original_width]
                  
                  # process S1 input
                  s1_img = s1_input[0][0].cpu().numpy()
                  s1_img = s1_img[:original_height, :original_width]

                  axs[i, 0].imshow(planet_img)
                  axs[i, 0].set_title("Planet-NICFI Input", fontsize=20)
                  axs[i, 1].imshow(s1_img, cmap='gray')
                  axs[i, 1].set_title("Sentinel-1 Input", fontsize=20)
            else:
                  inputs, outputs, targets = predictions[i]
                  
                  input_img = inputs[0].cpu().numpy().transpose(1, 2, 0)
                  if is_optical:
                        input_img = input_img[:, :, [2, 1, 0]]  # convert BGR to RGB
                        
                        input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min())
                        input_img = (input_img * 255).astype(np.uint8)
                        
                        axs[i, 0].imshow(input_img, cmap=None if is_optical else 'gray')
                        axs[i, 0].set_title("Input Image", fontsize=20)
                  else:
                        input_img = inputs[0][0].cpu().numpy()
                        
                  input_img = input_img[:original_height, :original_width]
            
            # process outputs and targets
            target_img = targets[0].cpu().numpy()
            probs = torch.sigmoid(outputs[0].cpu()).detach().numpy()
            output_img = (probs > threshold).squeeze().astype(np.uint8)
            
            # crop to original dimensions
            output_img = output_img[:original_height, :original_width]
            target_img = target_img[:original_height, :original_width]

            # plot model segmentation and ground truth
            axs[i, subplot_cols-2].imshow(output_img, cmap='gray')
            axs[i, subplot_cols-2].set_title("Model Segmentation", fontsize=20)
            axs[i, subplot_cols-1].imshow(target_img, cmap='gray')
            axs[i, subplot_cols-1].set_title("Ground Truth", fontsize=20)

            for ax in axs[i]:
                  ax.axis("off")

      plt.tight_layout()
      plt.savefig(f'reports/{output_name}.png')
      plt.show()