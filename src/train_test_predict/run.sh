#!/bin/bash
#SBATCH --job-name=s1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=5G 
#SBATCH --gpus=1
#SBATCH --time=14-00:00:00  
#SBATCH --nodelist=guanabana

export CUDA_VISIBLE_DEVICES=1

## scripts for hparams optimization
# python src/optimization/hparams_optimization_standalone.py
# python src/optimization/hparams_optimization_ef.py
# python src/optimization/hparams_optimization_fusion.py
# python src/optimization/hparams_optimization_fusion_pretrained.py

## scripts for training and testing
python src/train_test_predict/train_model.py
# python src/train_test_predict/test_model.py

## scripts for inference and prediction examples
# python src/train_test_predict/batch_inference.py
# python src/train_test_predict/predict_model.py

