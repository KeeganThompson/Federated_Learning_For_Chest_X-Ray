# Federated_Learning_For_Chest_X-Ray
Automate.py -- Runs full training loop going between training local hospitals to aggregation back to training local hospitals for a set amount of rounds
Eval_global.py -- Evaluates the global model with less detail
Eval_global_new.py -- Evaluates the global model on all metrics including BCE and AUROC
Eval_local.py -- Evaluates the local model on all metrics including BCE and AUROC
fedavg_aggregator.py -- used to aggregate the weights using fedavg
fedprox_aggregator.py -- used to aggregate the weights using fedprox
hospital_split.py -- used to split the dataset into 20 hospitals uniformly
hospital_split_skew.py -- used to split the dataset into 20 hospitals, created skewed datasets for hospitals 1-4, you can change the specific labels for which each hospital is skewed in the code
sample_evaluation.py -- creates script for evaluation/visuals of the progression (per round) of samples predicted incorrectly to correctly and v.v.
train_hospital.py -- used to train the local hospitals using global weights or no global weights.


Dataset Used:
https://www.kaggle.com/datasets/ashery/chexpert/data?select=train
-- This dataset contains train folder and validation folder. Each folder contains a number of patients, each patient contains a folder or folders of studies, each study contains jpg files of frontal or lateral Chest X-ray images. The dataset also contains a train.csv file and a valid.csv file, each containing all of the relevant information for each jpg attatched to a patient containing sex, age, frontal/lateral, AP/PA, and the 14 labels to be classified, including, for example, No Finding or Cardiomegaly.
