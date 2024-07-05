#!/bin/bash
#SBATCH --job-name=inf
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=5G 
#SBATCH --gpus=1
#SBATCH --time=14-00:00:00  
#SBATCH --nodelist=grenadilla

export CUDA_VISIBLE_DEVICES=1

## scripts for hparams optimization
# python src/optimization/hparams_optimization_standalone.py
# python src/optimization/hparams_optimization_ef.py
# python src/optimization/hparams_optimization_fusion.py
# python src/optimization/hparams_optimization_fusion_pretrained.py

## scripts for training and testing
# python src/models/train_model.py
# python src/models/test_model.py

## scripts for inference and prediction examples
# python src/models/inference_gee.py
python src/models/batch_inference.py
# python src/models/predict_model.py

# python src/models/confidence_filtering.py

