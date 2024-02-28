import optuna

# database file path
db_file_path = 'sqlite:////mnt/guanabana/raid/home/pasan001/asm-mapping-deeplearning/src/optimization/binary_optimization.db'

# load the study
study = optuna.load_study(study_name='unet_binary_multiobj_study', storage=db_file_path)

# retrieve the best trials
pareto_front_trials = study.best_trials

# print the details of the Pareto front trials
for trial in pareto_front_trials:
    print(f'Trial {trial.number}:')
    print(f'  Values: {trial.values}')
    print(f'  Params: {trial.params}')