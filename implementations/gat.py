import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, nheads=16): 
        super(GAT, self).__init__()
        self.conv1 = GATConv(nfeat, nhid, heads=nheads, dropout=0.5)
        self.conv1.att = None 
        self.transition = nn.Sequential(
            nn.ReLU(), 
            nn.BatchNorm1d(nhid * nheads), 
            nn.Dropout(p=dropout)
        )
        self.fc = nn.Linear(nhid, nclass)
        for m in self.modules(): 
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
                
    def forward(self, edge_index, x):
        x = self.conv1(x, edge_index)
        x = x.flatten(start_dim=1)
        x = self.transition(x)
        x = self.fc(x)
        return x 