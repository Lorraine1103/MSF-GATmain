# MSF-GATmain

# MSF-GAT

# **Environment**

The most important python packages are:
- python == 3.11.0
- pytorch == 2.1.1+cu121
- torch-cluster == 1.6.3+pt21cu121
- torch_geometric == 2.4.0
- torch-scatter == 2.1.2+pt21cu121
- torch-sparse == 0.6.18+pt21cu121
- tensorboard == 2.15.2
- rdkit == 2023.9.2
- scikit-learn == 1.3.0
- hyperopt == 0.2.7
- numpy == 1.18.2


---
# **Command**

### **1. Train**
Use train.py

Args:
  - data_path : The path of input CSV file. *E.g. input.csv*
  - dataset_type : The type of dataset. *E.g. classification  or  regression*
  - save_path : The path to save output model. *E.g. model_save*
  - log_path : The path to record and save the result of training. *E.g. log*

E.g.

`python train.py --data_path Data/MoleculeNet/bace.csv --dataset_type classification --save_path model_save/bace --log_path log/bace`

### **2. Predict**
Use predict.py

Args:
  - predict_path : The path of input CSV file to predict. *E.g. input.csv*
  - result_path : The path of output CSV file. *E.g. output.csv*
  - model_path : The path of trained model. *E.g. model_save/model.pt*

E.g.

`python predict.py --predict_path Data/MoleculeNet/bace.csv --model_path model_save/bace/Seed_0/model.pt --result_path result_save/bace.csv`

### **3. Hyperparameters Optimization**
Use hyper_opti.py

Args:
  - data_path : The path of input CSV file. *E.g. input.csv*
  - dataset_type : The type of dataset. *E.g. classification  or  regression*
  - save_path : The path to save output model. *E.g. model_save*
  - log_path : The path to record and save the result of hyperparameters optimization. *E.g. log*

E.g.

`python hyper_opti.py --data_path Data/MoleculeNet/bace.csv --dataset_type classification --save_path model_save/bace_hyper --log_path log/bace_hyper`


