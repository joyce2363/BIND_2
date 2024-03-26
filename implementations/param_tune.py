from __future__ import division
from __future__ import print_function
from torch_geometric.utils import convert
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import load_bail, load_income, load_pokec_renewed
from GNNs import GCN, GAT, SAGE
from scipy.stats import wasserstein_distance
from tqdm import tqdm
from influence_computation_and_save_2 import part2
from removing_and_testing_3 import part3
from debiasing_gnns import part4
import optuna
import csv
from training_og1b import part1b
from training_og1 import part1
from influence_computation_and_save_2 import part2
from removing_and_testing_3 import part3
from debiasing_gnns import part4
from influence_nba2 import part2b
from debiasing_nba import part4b
# /home/joyce/BIND_2/implementations/2_influence_computation_and_save.py
import warnings
warnings.filterwarnings('ignore')

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--model', type=str, default='gcn')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of epochs to train.')
parser.add_argument('--dataset', type=str, default="income", help='One dataset from income, bail, pokec1, and pokec2.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
dataset_name = args.dataset
# args.seed = 1
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

def feature_norm(features):
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]
    return 2*(features - min_values).div(max_values-min_values) - 1

def objective(trial):
    # Define the hyperparameter search space
    # parser.add_argument('--dataset', type=str, default="income", help='One dataset from income, bail, pokec1, and pokec2.')
    num_hidden = trial.suggest_categorical("num_hidden", [4, 16, 64, 128])
    dropout = trial.suggest_categorical("dropout", [0.3, 0.4, 0.5, 0.6, 0.7])
    lr = trial.suggest_categorical("lr", [0.01, 0.001, 0.0001, 0.00001])
    weight_decay = trial.suggest_categorical("weight_decay", [0.01, 0.001, 0.0001, 0.00001])
    ap = trial.suggest_categorical("ap", [10, 25, 50, 60, 80, 100, 200, 1000])
    args.num_hidden = num_hidden
    args.lr = lr
    args.dropout = dropout
    args.weight_decay = weight_decay
    dataset_name = args.dataset
    # print("hi here")
    # print("model: ", model)
    if dataset_name == 'nba':
        part1b(dataset = args.dataset,
            pnum_hidden = num_hidden,
            pdropout = dropout, 
            plr = lr, 
            pweight_decay = weight_decay, 
            epochs = args.epochs,
            model = args.model,
            pseed = args.seed)
        part2b(
            dataset = args.dataset,
            model = args.model,
            aprox = ap,
            pseed = args.seed
        )
        part3(
            dataset = args.dataset,
            model = args.model, 
            plr = lr,
            pweight_decay = weight_decay,
            pnum_hidden = num_hidden,
            pdropout = dropout,
            pseed = args.seed
        )
        acc = part4b(dataset_name = args.dataset,
                    seed = args.seed)
        return acc
    else:   
        # part1(dataset = args.dataset,
        #     pnum_hidden = num_hidden,
        #     pdropout = dropout, 
        #     plr = lr, 
        #     pweight_decay = weight_decay, 
        #     epochs = args.epochs,
        #     model = args.model,
        #     pseed = args.seed)
        
        part2(
            dataset = args.dataset,
            model = args.model,
            aprox = ap,
            pseed = args.seed
        )

        part3(
            dataset = args.dataset,
            model = args.model, 
            plr = lr,
            pweight_decay = weight_decay,
            pnum_hidden = num_hidden,
            pdropout = dropout,
            pseed = args.seed
        )
        acc = part4(dataset_name = args.dataset,
                    seed = args.seed)
        return acc
# Create an Optuna study object and specify the optimization direction
study = optuna.create_study(direction='maximize')

# Run the optimization
study.optimize(objective, n_trials=100)  # Run 100 trials
best_params = study.best_params
best_value = study.best_value


# Get the best trial
best_trial = study.best_trial

# Print the best trial number and value
print(f"Best trial number: {best_trial.number}")
print(f"Best value: {best_trial.value}")

# Get the best parameters
best_params = best_trial.params
best_params["dataset: "] = args.dataset
best_params["seed: "] = args.seed
best_params["acc: "] = best_trial.value
best_params["model: "] = "BIND_" + str(args.model)
# Print the best parameters
print("Best parameters:")
for key, value in best_params.items():
    print(f"{key}: {value}")

if args.dataset == "pokec1": 
    filename = 'hyperparameter_NEW' + str(args.dataset) + '.csv'
elif args.dataset == "pokec2":
    filename = 'hyperparameter_NEW.csv'
elif args.dataset == "nba": 
    filename = 'hyperparameter_NEW' + str(args.dataset) + '.csv'
elif args.dataset == "income": 
    filename = 'hyperparameter_NEW' + str(args.dataset) + '.csv'
elif args.dataset == "bail": 
    filename = 'hyperparameter_NEW' + str(args.dataset) + '.csv'

# Writing data to CSV
with open(filename, 'a', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=best_params.keys())
    writer.writerow(best_params)

print(f"Data has been written to {filename}")