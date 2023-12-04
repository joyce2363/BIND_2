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
from GNNs import GCN
from scipy.stats import wasserstein_distance
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

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
    epoch = trial.suggest_int("epoch", 1, 3000)
    lr = trial.suggest_float("lr", 0.00001, 0.01)
    weight_decay = trial.suggest_float("weight_decay", 0.00001, 0.001)
    hidden = trial.suggest_int("hidden", 1, 200)
    dropout = trial.suggest_float("dropout", 0.2, 0.8)

    adj, features, labels, idx_train, idx_val, idx_test, sens = load_bail('bail')
        # adj, features, labels, idx_train, idx_val, idx_test, sens = load_income('income')

    edge_index = convert.from_scipy_sparse_matrix(adj)[0]
    model = GCN(nfeat=features.shape[1], nhid=hidden, nclass=1, dropout=dropout)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    #fit aka jump 
    t_total = time.time()
    final_epochs = 0
    loss_val_global = 1e10
    dataset_name = 'bail'
    starting = time.time()
    tracker = 0
    while tracker != 5: 
        for epoch in tqdm(range(epoch)):
        loss_mid = train(epoch, model, optimizer, features, edge_index, labels, idx_train, idx_val)
        if loss_mid < loss_val_global:
            loss_val_global = loss_mid
            torch.save(model, 'gcn_' + dataset_name + '.pth')
            final_epochs = epoch
        tracker += 1

        torch.save(model, 'gcn_' + dataset_name + '.pth')

        ending = time.time()
        print("Time:", ending - starting, "s")
        model = torch.load('gcn_' + dataset_name + '.pth')
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
    # for epoch in tqdm(range(epoch)):
    #     loss_mid = train(epoch, model, optimizer, features, edge_index, labels, idx_train, idx_val)
    #     if loss_mid < loss_val_global:
    #         loss_val_global = loss_mid
    #         torch.save(model, 'gcn_' + dataset_name + '.pth')
    #         final_epochs = epoch

    # torch.save(model, 'gcn_' + dataset_name + '.pth')

    # ending = time.time()
    # print("Time:", ending - starting, "s")
    # model = torch.load('gcn_' + dataset_name + '.pth')
    # # acc_test = tst()
    # model.eval()
    # output = model(features, edge_index)
    # preds = (output.squeeze() > 0).type_as(labels)
    # loss_test = F.binary_cross_entropy_with_logits(output[idx_test], labels[idx_test].unsqueeze(1).float())
    # acc_test = accuracy_new(preds[idx_test], labels[idx_test])

    # print("*****************  Cost  ********************")
    # print("SP cost:")
    #     # sens = sens.cuda()
    # idx_sens_test = sens[idx_test]
    # idx_output_test = output[idx_test]
    # # print(wasserstein_distance(idx_output_test[idx_sens_test==0].squeeze().cpu().detach().numpy(), idx_output_test[idx_sens_test==1].squeeze().cpu().detach().numpy()))
    # print("EO cost:")
    # idx_sens_test = sens[idx_test][labels[idx_test]==1]
    # idx_output_test = output[idx_test][labels[idx_test]==1]
    # return acc_test
    #

    # Create an Optuna study object and specify the optimization direction
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

# Print the best parameters
print("Best parameters:")
for key, value in best_params.items():
    print(f"{key}: {value}")