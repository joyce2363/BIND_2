from __future__ import division
from __future__ import print_function
# from torch_geometric.utils import convert
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
# from utils import load_bail, load_income, load_pokec_renewed
# from GNNs import GCN, GAT, SAGE
# from scipy.stats import wasserstein_distance
from tqdm import tqdm
import csv
from training_og1b import part1b
from training_og1 import part1
from influence_computation_and_save_2 import part2
from removing_and_testing_3 import part3
from debiasing_gnns import part4
from influence_nba2 import part2b
from debiasing_nba_TEST import part4c
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
# parser.add_argument('--seed', nargs='+', type=int, default=[1])
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of epochs to train.')
parser.add_argument('--dataset', type=str, default="income", help='One dataset from income, bail, pokec1, and pokec2.')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--num_hidden', type=int, default=16)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--ap', type=float, default=25)
# parser.add_argument('--lr', nargs='+', type=float, default=[0.001])
# parser.add_argument('--weight_decay', nargs='+', type=float, default=[0.0001])
# parser.add_argument('--num_hidden', nargs='+', type=int, default=[16])
# parser.add_argument('--dropout', nargs='+', type=float, default=[0.5])
# parser.add_argument('--ap', nargs='+', type=float, default=[25])

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
dataset_name = args.dataset
# args.seed = 1
ap = args.ap
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

def feature_norm(features):
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]
    return 2*(features - min_values).div(max_values-min_values) - 1

if dataset_name == 'nba':
    part1b(dataset = args.dataset,
                pnum_hidden = args.num_hidden,
                pdropout = args.dropout, 
                plr = args.lr, 
                pweight_decay = args.weight_decay, 
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
        plr = args.lr,
        pweight_decay = args.weight_decay,
        pnum_hidden = args.num_hidden,
        pdropout = args.dropout,
        pseed = args.seed
    )    
    ACC_10, SP_10, EO_10, ACC_100, SP_100, EO_100 = part4c(
    dataset_name = args.dataset,
            seed = args.seed)
else:
    # part1(dataset = args.dataset,
    #         pnum_hidden = args.num_hidden,
    #         pdropout = args.dropout, 
    #         plr = args.lr, 
    #         pweight_decay = args.weight_decay, 
    #         epochs = args.epochs,
    #         model = args.model,
    #         pseed = args.seed)

    part2(dataset = args.dataset,
            model = args.model,
            aprox = ap,
            pseed = args.seed
    )

    part3(
        dataset = args.dataset,
        model = args.model, 
        plr = args.lr,
        pweight_decay = args.weight_decay,
        pnum_hidden = args.num_hidden,
        pdropout = args.dropout,
        pseed = args.seed
    )
    ACC_10, SP_10, EO_10, ACC_100, SP_100, EO_100 = part4(
        dataset_name = args.dataset,
                seed = args.seed)

filename = 'TEST_output.csv'
fieldnames = ['dataset', 'seed', 'model', 'ACC_10', 'SP_10', 'EO_10', 'ACC_100', 'SP_100', 'EO_100']

# Create a dictionary mapping fieldnames to the corresponding variables
row_data = {
    'dataset' : str(args.dataset),
    'seed': args.seed,
    'model': str(args.model),
    'ACC_10': ACC_10,
    'SP_10': SP_10,
    'EO_10': EO_10,
    'ACC_100': ACC_100,
    'SP_100': SP_100,
    'EO_100': EO_100
}

# Write the data to the CSV file
with open(filename, 'a', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()  # Include this if you want a header row in your CSV
    writer.writerow(row_data)

print(f"Data has been written to {filename}")
# return ACC_10, SP_10, EO_10, ACC_100, SP_100, EO_100