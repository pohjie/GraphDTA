import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp

import torchnlp.nn.attention as A

from numba import jit
import numpy as np

import pdb
import time

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def fast_reshape(batch, x, x_reshaped):
    count = np.zeros(x_reshaped.shape[0], dtype=np.int8)
    for i in range(batch.shape[0]):
        idx = batch[i]
        sec_dim = count[idx]
        x_reshaped[idx, sec_dim, :] = x[i]
        count[idx] += 1

    return x_reshaped

# GCN based model
class AttnGCNNet(torch.nn.Module):
    def __init__(self, n_output=1, n_filters=32, embed_dim=128,num_features_xd=78, num_features_xt=25, output_dim=128, dropout=0.2):

        super(AttnGCNNet, self).__init__()

        # SMILES graph branch
        self.n_output = n_output
        self.conv1 = GCNConv(num_features_xd, num_features_xd)
        self.conv2 = GCNConv(num_features_xd, num_features_xd+10)
        self.conv3 = GCNConv(num_features_xd+10, num_features_xd+10)
        self.conv4 = GCNConv(num_features_xd+10, num_features_xd+30)
        self.conv5 = GCNConv(num_features_xd+30, num_features_xd+30)
        self.conv6 = GCNConv(num_features_xd+30, num_features_xd+50)
        self.fc_g1 = torch.nn.Linear(num_features_xd+50, 1024)
        self.fc_g2 = torch.nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.attention = A.Attention(embed_dim)

        # protein sequence branch (1d conv)
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        self.fc1_xt = nn.Linear(32*121, output_dim)

        # combined layers
        self.fc1 = nn.Linear(2*output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data):
        # get graph input
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # get protein input
        target = data.target

        x = self.conv1(x, edge_index)
        x = self.relu(x)
        identity_x = x
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x += identity_x
        x = self.conv3(x, edge_index)
        x = self.relu(x)
        identity_x = x
        x = self.conv4(x, edge_index)
        x = self.relu(x)
        x += identity_x
        x = self.conv5(x, edge_index)
        x = self.relu(x)
        x = self.conv6(x, edge_index)
        x = self.relu(x)

        # 1d conv layers
        embedded_xt = self.embedding_xt(target)

        batch_size = target.shape[0]
        v, i = torch.mode(batch)
        v_freq = batch.eq(v.item()).sum().item()
        x_reshaped = torch.zeros([batch_size, v_freq, embedded_xt.shape[2]],
                                  dtype=torch.float64)

        # create a count for the 2nd dim
        device = x.get_device()
        x_reshaped = torch.from_numpy(fast_reshape(batch.cpu().numpy(),
                                      x.cpu().detach().numpy(), x_reshaped.numpy())).to(device)


        x = gmp(x, batch)       # global max pooling

        # flatten
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.dropout(x)

        output, weights = self.attention(embedded_xt, x_reshaped.float()) # query, context
        conv_xt = self.conv_xt_1(output)
        # conv_xt = self.conv_xt_1(embedded_xt)

        # flatten
        xt = conv_xt.view(-1, 32 * 121)
        xt = self.fc1_xt(xt)

        # concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out

        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.relu(self.out(x))
        return F.log_softmax(x, dim=-1)


