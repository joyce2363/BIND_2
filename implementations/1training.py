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
from GNNs import GCN, GAT, SAGE
from scipy.stats import wasserstein_distance
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
# import ctypes
# ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--dataset', type=str, default="bail", help='One dataset from income, bail, pokec1, and pokec2.')
parser.add_argument('--model', type=str, default="gcn")
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
dataset_name = args.dataset

def feature_norm(features):
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]
    return 2*(features - min_values).div(max_values-min_values) - 1

def accuracy_new(output, labels):
    correct = output.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def fair_metric(pred, labels, sens):
    idx_s0 = sens==0
    idx_s1 = sens==1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels==1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels==1)
    parity = abs(sum(pred[idx_s0])/sum(idx_s0)-sum(pred[idx_s1])/sum(idx_s1))
    equality = abs(sum(pred[idx_s0_y1])/sum(idx_s0_y1)-sum(pred[idx_s1_y1])/sum(idx_s1_y1))
    return parity.item(), equality.item()

def train(epoch, model, optimizer, features, edge_index, idx_train, idx_val, idx_test, labels, sens):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, edge_index)
    loss_train = F.binary_cross_entropy_with_logits(output[idx_train], labels[idx_train].unsqueeze(1).float())
    preds = (output.squeeze() > 0).type_as(labels)
    acc_train = accuracy_new(preds[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        model.eval()
        output = model(features, edge_index)

    loss_val = F.binary_cross_entropy_with_logits(output[idx_val], labels[idx_val].unsqueeze(1).float())
    acc_val = accuracy_new(preds[idx_val], labels[idx_val])
    return loss_val.item()

def tst(epoch, model, optimizer, features, edge_index, idx_train, idx_val, idx_test, labels, sens):
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
    print(wasserstein_distance(idx_output_test[idx_sens_test==0].squeeze().cpu().detach().numpy(), idx_output_test[idx_sens_test==1].squeeze().cpu().detach().numpy()))

    print("EO cost:")
    idx_sens_test = sens[idx_test][labels[idx_test]==1]
    idx_output_test = output[idx_test][labels[idx_test]==1]
    print(wasserstein_distance(idx_output_test[idx_sens_test==0].squeeze().cpu().detach().numpy(), idx_output_test[idx_sens_test==1].squeeze().cpu().detach().numpy()))
    print("**********************************************")

    parity, equality = fair_metric(preds[idx_test].cpu().numpy(), labels[idx_test].cpu().numpy(),
                                   sens[idx_test].cpu().numpy())

    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    print("Statistical Parity:  " + str(parity))
    print("Equality:  " + str(equality))


t_total = time.time()
final_epochs = 0
loss_val_global = 1e10

starting = time.time()

def objective(trial):
    # Define the hyperparameter search space
    hidden = trial.suggest_categorical("hidden", [4, 16, 64, 128])
    lr = trial.suggest_categorical("lr", [1e-2, 1e-3, 1e-4, 1e-5])
    dropout = trial.suggest_categorical("dropout", [0.3, 0.4, 0.5, 0.6, 0.7])
    weight_decay = trial.suggest_categorical("weight_decay", [1e-2, 1e-3, 1e-4, 1e-5])
    args.dropout = dropout
    args.lr = lr
    args.weight_decay = weight_decay
    args.hidden = hidden

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if dataset_name == 'bail':
        adj, features, labels, idx_train, idx_val, idx_test, sens = load_bail('bail', args.seed)
        norm_features = feature_norm(features)
        norm_features[:, 0] = features[:, 0]
        features = feature_norm(features)
    elif dataset_name == 'income':
        adj, features, labels, idx_train, idx_val, idx_test, sens = load_income('income', args.seed)
        norm_features = feature_norm(features)
        norm_features[:, 8] = features[:, 8]
        features = feature_norm(features)
    elif dataset_name == "nba":
        adj, features, labels, idx_train, idx_val, idx_test, sens = load_nba('nba', args.seed)
        norm_features = feature_norm(features)
        norm_features[:, 0] = features[:, 0]
        features = feature_norm(features)
    elif dataset_name == 'pokec1':
        adj, features, labels, idx_train, idx_val, idx_test, sens = load_pokec_renewed(1, args.seed)
    elif dataset_name == 'pokec2':
        adj, features, labels, idx_train, idx_val, idx_test, sens = load_pokec_renewed(2, args.seed)

    edge_index = convert.from_scipy_sparse_matrix(adj)[0]
    if args.model == "sage": 
        model = SAGE(nfeat=features.shape[1], nhid = 1)
    elif args.model == "gat":
        model = GAT(nfeat=features.shape[1], nhid = 1)
    elif args.model == "gcn": 
        # print("entered")
        model = GCN(nfeat=features.shape[1], nhid = args.hidden, nclass=labels.unique().shape[0]-1, dropout=args.dropout)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.cuda:
        model.cuda()
        features = features.cuda()
        edge_index = edge_index.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
    for epoch in tqdm(range(args.epochs)):
        loss_mid = train(epoch, model, optimizer, features, edge_index, idx_train, idx_val, idx_test, labels, sens)
        if loss_mid < loss_val_global:
            print("entered part 2")
            loss_val_global = loss_mid
            torch.save(model, str(args.model) + '_all_' + dataset_name + str(args.seed) + '.pth')
            final_epochs = epoch

    torch.save(model, str(args.model) + '_all_' + dataset_name + str(args.seed) + '.pth')

    ending = time.time()
    print("Time:", ending - starting, "s")
    model = torch.load(args.model, str(args.model) + '_all_' + dataset_name + str(args.seed) + '.pth')
    tst(epoch, model, optimizer, features, edge_index, idx_train, idx_val, idx_test, labels, sens)