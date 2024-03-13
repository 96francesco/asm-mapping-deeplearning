import torch

def get_predictions(model, test_loader, indices, output_name: str, is_fusion=False):
      model.eval()  # set model to evaluation mode
      model.freeze()
      predictions = []

      with torch.no_grad():
            for index in indices:
                  if is_fusion:
                        planet_input, s1_input, targets = test_loader.dataset[index]
                        planet_input = planet_input.to(model.device).unsqueeze(0)  # add batch dimension.
                        s1_input = s1_input.to(model.device).unsqueeze(0)  # add batch dimension.
                        outputs = model(planet_input, s1_input)
                  else:
                        inputs, targets = test_loader.dataset[index]
                        inputs = inputs.to(model.device).unsqueeze(0) # add batch dimension
                        outputs = model(inputs)

                  targets = targets.unsqueeze(0)
                  if is_fusion:
                        predictions.append((planet_input, s1_input, outputs, targets))
                  else:
                        predictions.append((inputs, outputs, targets))
      
      # save predictions
      torch.save(predictions, f'models/predictions/{output_name}-{str(indices)}.pth')

      return predictions