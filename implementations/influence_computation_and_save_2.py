from utils import load_bail, load_income, load_pokec_renewed, load_nba
from numpy import *
from approximator import grad_z_graph, cal_influence_graph, s_test_graph_cost, cal_influence_graph_nodal
import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm
import pandas as pd
from scipy.spatial import distance_matrix
import os
import networkx as nx
import time
import argparse
from torch_geometric.utils import convert
import warnings
# from removing_and_testing_3 import part3
warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True

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
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--hidden', type=int, default=16)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--ap', type=float, default=25)
args = parser.parse_args()


def part2(dataset,
        #   pnum_hidden, 
        #   pdropoput,
        #   plr, 
        #   pweight_decay,
        #   epochs,
          model,
          aprox,
          pseed,
          ):
    dataset_name = dataset 
    seed = pseed
    args.seed = pseed
    args.model = model
    def find123Nei(G, node):
        nodes = list(nx.nodes(G))
        nei1_li = []
        nei2_li = []
        nei3_li = []
        for FNs in list(nx.neighbors(G, node)):
            nei1_li .append(FNs)

        for n1 in nei1_li:
            for SNs in list(nx.neighbors(G, n1)):
                nei2_li.append(SNs)
        nei2_li = list(set(nei2_li) - set(nei1_li))
        if node in nei2_li:
            nei2_li.remove(node)

        for n2 in nei2_li:
            for TNs in nx.neighbors(G, n2):
                nei3_li.append(TNs)
        nei3_li = list(set(nei3_li) - set(nei2_li) - set(nei1_li))
        if node in nei3_li:
            nei3_li.remove(node)

        return nei1_li, nei2_li, nei3_li

    def feature_norm(features):
        min_values = features.min(axis=0)[0]
        max_values = features.max(axis=0)[0]
        return 2*(features - min_values).div(max_values-min_values) - 1

    def build_relationship(x, thresh=0.25):
        df_euclid = pd.DataFrame(1 / (1 + distance_matrix(x.T.T, x.T.T)), columns=x.T.columns, index=x.T.columns)
        df_euclid = df_euclid.to_numpy()
        idx_map = []
        for ind in range(df_euclid.shape[0]):
            max_sim = np.sort(df_euclid[ind, :])[-2]
            neig_id = np.where(df_euclid[ind, :] > thresh * max_sim)[0]
            import random
            random.seed(seed)
            random.shuffle(neig_id)
            for neig in neig_id:
                if neig != ind:
                    idx_map.append([ind, neig])
        idx_map = np.array(idx_map)

        return idx_map

    def get_adj(dataset_name):
        predict_attr = "RECID"
        if dataset_name == 'bail':
            predict_attr="RECID"
        elif dataset_name == 'income':
            predict_attr = "income"
        elif dataset_name == 'nba':
            predict_attr = "SALARY"
        if dataset_name == 'pokec1' or dataset_name == 'pokec2':
            if dataset_name == 'pokec1':
                edges = np.load('../data/pokec_dataset/region_job_1_edges.npy')
                labels = np.load('../data/pokec_dataset/region_job_1_labels.npy')
            else:
                edges = np.load('../data/pokec_dataset/region_job_2_2_edges.npy')
                labels = np.load('../data/pokec_dataset/region_job_2_2_labels.npy')

            adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                                shape=(labels.shape[0], labels.shape[0]),
                                dtype=np.float32)
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
            return adj

        path="../data/" + str(dataset_name) + "/"
        dataset = dataset_name
        print('Reconstructing the adj of {} dataset...'.format(dataset))

        idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
        header = list(idx_features_labels.columns)
        header.remove(predict_attr)
        if args.dataset == "nba": 
            header.remove("user_id")

        if args.dataset == 'nba':
            edges_unordered = np.genfromtxt(os.path.join(path,"{}_relationship.txt".format(dataset)), dtype=int)
        else: 
            edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')

        features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
        labels = idx_features_labels[predict_attr].values

        if args.dataset == 'nba': 
            print("inside if statement")
            idx = np.array(idx_features_labels["user_id"], dtype=int)
        else: 
            idx = np.arange(features.shape[0])
        idx_map = {j: i for i, j in enumerate(idx)}
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                        dtype=int).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        return adj

    def del_adj(harmful):
        adj = adj_vanilla
        mask = np.ones(adj.shape[0], dtype=bool)
        mask[harmful] = False
        adj = sp.coo_matrix(adj.tocsr()[mask,:][:,mask])

        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = adj + sp.eye(adj.shape[0])

        return adj

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if dataset_name == 'bail':
        model = torch.load(str(args.model) + '_' +  dataset_name + '_' + str(args.seed) + '.pth')
        adj_vanilla, features_vanilla, labels_vanilla, idx_train_vanilla, idx_val_vanilla, idx_test_vanilla, sens_vanilla = load_bail('bail', seed)
        norm_features = feature_norm(features_vanilla)
        norm_features[:, 0] = features_vanilla[:, 0]
        features_vanilla = norm_features
    elif dataset_name == 'income':
        model = torch.load(str(args.model) + '_' +  dataset_name + '_' + str(args.seed) + '.pth')
        adj_vanilla, features_vanilla, labels_vanilla, idx_train_vanilla, idx_val_vanilla, idx_test_vanilla, sens_vanilla = load_income('income', seed)
        norm_features = feature_norm(features_vanilla)
        norm_features[:, 8] = features_vanilla[:, 8]
        features_vanilla = norm_features
    elif dataset_name == 'nba': 
        model = torch.load(str(args.model) + '_' +  dataset_name + '_' + str(args.seed) + '.pth')
        adj_vanilla, features_vanilla, labels_vanilla, idx_train_vanilla, idx_val_vanilla, idx_test_vanilla, sens_vanilla = load_nba('nba', seed)
        norm_features = feature_norm(features_vanilla)
        norm_features[:, 0] = features_vanilla[:, 0]
        features_vanilla = norm_features
    elif dataset_name == 'pokec1':
        model = torch.load(str(args.model) + '_' +  dataset_name + '_' + str(args.seed) + '.pth')
        adj_vanilla, features_vanilla, labels_vanilla, idx_train_vanilla, idx_val_vanilla, idx_test_vanilla, sens_vanilla = load_pokec_renewed(1, seed)
    elif dataset_name == 'pokec2':
        model = torch.load(str(args.model) + '_' +  dataset_name + '_' + str(args.seed) + '.pth')
        adj_vanilla, features_vanilla, labels_vanilla, idx_train_vanilla, idx_val_vanilla, idx_test_vanilla, sens_vanilla = load_pokec_renewed(2, seed)


    edge_index = convert.from_scipy_sparse_matrix(adj_vanilla)[0]
    print("Pre-processing data...")
    computation_graph_involving = []
    the_adj = get_adj(dataset_name)
    hop = 1
    G = nx.Graph(the_adj)

    for i in tqdm(range(idx_train_vanilla.shape[0])):
        neighbors = find123Nei(G, idx_train_vanilla[i].item())
        mid = []
        for j in range(hop):
            mid += neighbors[j]
        mid = list(set(mid).intersection(set(idx_train_vanilla.numpy().tolist())))
        computation_graph_involving.append(mid)
    print("Pre-processing completed.")

    time1 = time.time()

    h_estimate_cost = s_test_graph_cost(edge_index, 
                                        features_vanilla, 
                                        idx_train_vanilla, 
                                        idx_test_vanilla, 
                                        labels_vanilla, 
                                        sens_vanilla, 
                                        model, 
                                        scale = aprox,
                                        gpu=1)

    gradients_list = grad_z_graph(edge_index, features_vanilla, idx_train_vanilla, labels_vanilla, model, gpu=0)
    influence, harmful, helpful, harmful_idx, helpful_idx = cal_influence_graph(idx_train_vanilla, h_estimate_cost, gradients_list, gpu=0)
    non_iid_influence = []

    for i in tqdm(range(idx_train_vanilla.shape[0])):

        if len(computation_graph_involving[i]) == 0:
            non_iid_influence.append(0)
            continue

        reference = list(range(adj_vanilla.shape[0]))
        for j in range(len(reference) - idx_train_vanilla[i]):
            reference[j + idx_train_vanilla[i]] -= 1

        mask = np.ones(idx_train_vanilla.shape[0], dtype=bool)
        mask[i] = False
        idx_train = idx_train_vanilla[mask]
        idx_val = idx_val_vanilla.clone()
        idx_test = idx_test_vanilla.clone()

        idx_train = torch.LongTensor(np.array(reference)[idx_train.numpy()])
        idx_val = torch.LongTensor(np.array(reference)[idx_val.numpy()])
        idx_test = torch.LongTensor(np.array(reference)[idx_test.numpy()])

        computation_graph_involving_copy = computation_graph_involving.copy()
        for j in range(len(computation_graph_involving_copy)):
            computation_graph_involving_copy[j] = np.array(reference)[computation_graph_involving_copy[j]]

        mask = np.ones(labels_vanilla.shape[0], dtype=bool)
        mask[idx_train_vanilla[i]] = False

        features = features_vanilla[mask, :]
        labels = labels_vanilla[mask]
        sens = sens_vanilla[mask]

        adj = del_adj(idx_train_vanilla[i])
        edge_index = convert.from_scipy_sparse_matrix(adj)[0]
        h_estimate_cost_nodal = h_estimate_cost.copy()
        gradients_list_nodal = grad_z_graph(edge_index, features, torch.LongTensor(computation_graph_involving_copy[i]), labels, model, gpu=0)
        influence_nodal, _, _, _, _ = cal_influence_graph_nodal(idx_train, torch.LongTensor(computation_graph_involving_copy[i]), h_estimate_cost_nodal, gradients_list_nodal,
                                                                                    gpu=0)
        non_iid_influence.append(sum(influence_nodal))

    final_influence = []
    for i in range(len(non_iid_influence)):
        ref = [idx_train_vanilla.numpy().tolist().index(item) for item in (computation_graph_involving[i] + [idx_train_vanilla[i]])]
        final_influence.append(non_iid_influence[i] - np.array(influence)[ref].sum())
    # print("FINAL_INFLUENCE: ", final_influence)
    time4 = time.time()
    print("LENGTH OF FINAL INFLUENCE: ", final_influence)
    print("Average time per training node:", (time4 - time1)/1000, "s")
    np.save('final_influence_' + str(args.model) + '_' + dataset_name + str(seed) + '.npy', np.array(final_influence))
    # NOTICE: here the helpfulness means the helpfulness to UNFAIRNESS
    # return dataset_name, lr, weight_decay, num_hidden, dropout