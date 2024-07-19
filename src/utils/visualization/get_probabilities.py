import torch

def get_probabilities(model, test_loader, indices, mode: str, output_name: str):
      probs_list = []

      with torch.no_grad():
            for index in indices:
                  if mode == 'late_fusion' or mode == 'early_fusion':
                        planet_input, s1_input, targets = test_loader.dataset[index]
                        planet_input = planet_input.to(model.device).unsqueeze(0)  # add batch dimension.
                        s1_input = s1_input.to(model.device).unsqueeze(0)  # add batch dimension.
                        logits = model(planet_input, s1_input)
                        probs = torch.sigmoid(logits) # convert logits to probabilities
                  else:
                        inputs, targets = test_loader.dataset[index]
                        inputs = inputs.to(model.device).unsqueeze(0) # add batch dimension
                        logits = model(inputs)
                        probs = torch.sigmoid(logits) # convert logits to probabilities

                  targets = targets.unsqueeze(0)
                  if mode == 'late_fusion' or mode == 'early_fusion':
                        probs_list.append((planet_input, s1_input, probs, targets))
                  else:
                        probs_list.append((inputs, probs, targets))
      
      # save predictions
      torch.save(probs_list, f'models/predictions/{output_name}-{str(indices)}.pth')

      return probs_list