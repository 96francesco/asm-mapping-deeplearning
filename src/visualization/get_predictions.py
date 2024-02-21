import torch

def get_predictions(model, test_loader, indices, output_name: str):
      model.eval()  # set model to evaluation mode
      model.freeze()
      predictions = []

      with torch.no_grad():
            for index in indices:
                  inputs, targets = test_loader.dataset[index]
                  inputs = inputs.to(model.device)
                  inputs = inputs.unsqueeze(0)  # Add batch dimension
                  targets = targets.unsqueeze(0)
                  outputs = model(inputs)
                  predictions.append((inputs, outputs, targets))
      
      # save predictions
      torch.save(predictions, f'models/predictions/{output_name}_{str(indices)}.pth')

      return predictions