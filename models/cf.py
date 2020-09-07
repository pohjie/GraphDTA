import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

# GINConv model
class CF(torch.nn.Module):
    def __init__(self, n_output=1,num_features_xd=78, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):

        super(CF, self).__init__()

        # davis
        self.smiles_emb = nn.Embedding(68, 256)
        self.tgt_emb = nn.Embedding(379, 256)

        # kiba
        # self.smiles_emb = nn.Embedding(2068, 128)
        # self.tgt_emb = nn.Embedding(229, 128)

        self.drop1 = nn.Dropout(0.2)
        self.cf_fc1 = nn.Linear(512, 256)
        self.cf_fc2 = nn.Linear(256, 128)

        self.out = nn.Linear(128, n_output)        # n_output = 1 for regression task

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target = data.target

        # CF
        embedded_smiles = self.smiles_emb(data.smiles_idx)
        embedded_tgt = self.tgt_emb(data.tgt_idx)

        # pass through 2 layers then concat with GNN output
        cf_x = torch.cat([embedded_smiles, embedded_tgt], 1)
        cf_x = F.relu(self.cf_fc1(cf_x))
        cf_x = self.drop1(cf_x)
        cf_x = F.relu(self.cf_fc2(cf_x))
        cf_x = self.drop1(cf_x)

        out = self.out(cf_x)
        return out
