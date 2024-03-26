
import csv 
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--model', type=str, default='gcn')
parser.add_argument('--seed', nargs='+', type=int, default=[1])
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of epochs to train.')
parser.add_argument('--dataset', type=str, default="income", help='One dataset from income, bail, pokec1, and pokec2.')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--hidden', type=int, default=16)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--ap', type=float, default=25)
args = parser.parse_args()
dataset_name = args.dataset

# parser = argparse.ArgumentParser()
# parser = argparse.ArgumentParser()
# parser.add_argument('--no-cuda', action='store_true', default=False,
#                     help='Disables CUDA training.')
# parser.add_argument('--fastmode', action='store_true', default=False,
#                     help='Validate during training pass.')
# parser.add_argument('--model', type=str, default='gcn')
# parser.add_argument('--seed', type=int, default=1, help='Random seed.')
# parser.add_argument('--epochs', type=int, default=1000,
#                     help='Number of epochs to train.')
# parser.add_argument('--dataset', type=str, default="income", help='One dataset from income, bail, pokec1, and pokec2.')

# args = parser.parse_args()
# dataset_name = args.dataset

def part4c(dataset_name, seed):
    final_sets = None
    if dataset_name == 'nba':
        final_sets = np.load('1final_sets_' + str(args.model) + dataset_name + str(seed) + '.npy', allow_pickle=True).item()
    print("length:", len(final_sets['acc_records']))
    if (len(final_sets['acc_records']) >= 1): 
        print("BIND 1%:")
        budget = 0
        print("Acc:", final_sets['acc_records'][budget])
        ACC_10 = final_sets['acc_records'][budget]
        print("Statistical Parity:", final_sets['sp_records'][budget])
        SP_10 = final_sets['sp_records'][budget]
        print("Equal Opportunity:", final_sets['eo_records'][budget])
        EO_10 = final_sets['eo_records'][budget]

    if (len(final_sets['acc_records']) >= 10): 
        print("BIND 10%:")
        budget = 9
        print("Acc:", final_sets['acc_records'][budget])
        ACC_100 = final_sets['acc_records'][budget]
        print("Statistical Parity:", final_sets['sp_records'][budget])
        SP_100 = final_sets['sp_records'][budget]
        print("Equal Opportunity:", final_sets['eo_records'][budget])
        EO_100 = final_sets['eo_records'][budget]
        return ACC_10, SP_10, EO_10, ACC_100, SP_100, EO_100
    else: 
        return ACC_10, SP_10, EO_10, 0, 0, 0
