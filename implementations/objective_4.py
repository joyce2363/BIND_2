from __future__ import division
from __future__ import print_function
from torch_geometric.utils import convert
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import optuna
from utils import load_bail, load_income, load_pokec_renewed, load_nba
from GNNs import GCN, SAGE, GAT
from scipy.stats import wasserstein_distance
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import argparse

import csv

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="nba", help='One dataset from income, bail, pokec_z, and pokec_n.')
parser.add_argument('--seed', type=str, default="1")
parser.add_argument('--model', type=str, default='gcn')
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--lr', type=int, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--hidden', type=int, default=16)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--ap', type=float, default=25)
args = parser.parse_args()

def train(epoch, model, optimizer, features, edge_index, labels, idx_train, idx_val):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features, edge_index)
        loss_train = F.binary_cross_entropy_with_logits(output[idx_train], labels[idx_train].unsqueeze(1).float())
        preds = (output.squeeze() > 0).type_as(labels)
        acc_train = accuracy_new(preds[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()
        fastmode = False
        if not fastmode:
            model.eval()
            output = model(features, edge_index)

        loss_val = F.binary_cross_entropy_with_logits(output[idx_val], labels[idx_val].unsqueeze(1).float())
        acc_val = accuracy_new(preds[idx_val], labels[idx_val])
        return loss_val.item()

def accuracy_new(output, labels):
        correct = output.eq(labels).double()
        correct = correct.sum()
        return correct / len(labels)

def objective(trial):
    lr = trial.suggest_categorical("lr", [0.01, 0.001, 0.0001, 0.00001])
    wd = trial.suggest_categorical("wd", [0.00001, 0.0001, 0.001])
    hidden = trial.suggest_categorical("hidden", [4, 8, 16, 32, 64, 128])
    dropout = trial.suggest_categorical("dropout", [0.4, 0.5, 0.6, 0.7])

    if args.dataset == "bail":
        adj, features, labels, idx_train, idx_val, idx_test, sens = load_bail('bail', args.seed)
    elif args.dataset == "nba": 
        adj, features, labels, idx_train, idx_val, idx_test, sens = load_nba('nba', args.seed)
    elif args.dataset == "income": 
        adj, features, labels, idx_train, idx_val, idx_test, sens = load_income('income', args.seed)
    elif args.dataset == "pokec1": 
        adj, features, labels, idx_train, idx_val, idx_test, sens = load_pokec_renewed(1, args.seed)
    elif args.dataset == "pokec2": 
        adj, features, labels, idx_train, idx_val, idx_test, sens = load_pokec_renewed(2, args.seed)
    
    edge_index = convert.from_scipy_sparse_matrix(adj)[0]
    if args.model == "gcn":
        model = GCN(nfeat=features.shape[1], nhid=hidden, nclass=labels.unique().shape[0]-1, dropout=dropout)
    elif args.model == "sage":
        model = SAGE(nfeat=features.shape[1], nhid=hidden, nclass=labels.unique().shape[0]-1, dropout=dropout)
    elif args.model == "gat":
        model = GAT(nfeat=features.shape[1], nhid=hidden, nclass=labels.unique().shape[0]-1, dropout=dropout)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay = wd)

    #fit aka jump 
    t_total = time.time()
    final_epochs = 0
    loss_val_global = 1e10
    dataset_name = args.dataset
    starting = time.time()
    tracker = 0
    epoch = args.epoch
    while tracker != 5: 
        for epoch in tqdm(range(epoch)):
            loss_mid = train(epoch, model, optimizer, features, edge_index, labels, idx_train, idx_val)
        if loss_mid < loss_val_global:
            loss_val_global = loss_mid
            torch.save(model, 'gcn_' + str(dataset_name) + '.pth')
            final_epochs = epoch
        tracker += 1

        torch.save(model, 'gcn_' + str(dataset_name) + '.pth')

        ending = time.time()
        print("Time:", ending - starting, "s")
        model = torch.load('gcn_' + str(dataset_name) + '.pth')
        # acc_test = tst()
        model.eval()
        output = model(features, edge_index)
        preds = (output.squeeze() > 0).type_as(labels)
        loss_test = F.binary_cross_entropy_with_logits(output[idx_test], labels[idx_test].unsqueeze(1).float())
        acc_test = accuracy_new(preds[idx_test], labels[idx_test])

        print("*****************  Cost  ********************")
        print("SP cost:")
            # sens = sens.cuda()
        idx_sens_test = sens[idx_test]
        idx_output_test = output[idx_test]
        # print(wasserstein_distance(idx_output_test[idx_sens_test==0].squeeze().cpu().detach().numpy(), idx_output_test[idx_sens_test==1].squeeze().cpu().detach().numpy()))
        print("EO cost:")
        idx_sens_test = sens[idx_test][labels[idx_test]==1]
        idx_output_test = output[idx_test][labels[idx_test]==1]
        print("acc_test: ", acc_test, "for tracker: ", tracker)
        return acc_test

study = optuna.create_study(direction='maximize')

# Run the optimization
study.optimize(objective, n_trials=100)  # Run 100 trials

# Get the best hyperparameters and their value
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
best_params["model: "] = "BIND_" + str(args.model) + '_' + str(args.model)

# Print the best parameters
print("Best parameters:")
for key, value in best_params.items():
    print(f"{key}: {value}")

if args.dataset == "pokec1": 
    filename = 'hyperparameter_' + str(args.dataset) + '.csv'
elif args.dataset == "pokec2":
    filename = 'hyperparameter_' + str(args.dataset) + '.csv'
elif args.dataset == "nba": 
    filename = 'hyperparameter_' + str(args.dataset) + '.csv'
elif args.dataset == "income": 
    filename = 'hyperparameter_' + str(args.dataset) + '.csv'
elif args.dataset == "bail": 
    filename = 'hyperparameter_' + str(args.dataset) + '.csv'



# Writing data to CSV
with open(filename, 'a', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=best_params.keys())
    writer.writerow(best_params)

print(f"Data has been written to {filename}")