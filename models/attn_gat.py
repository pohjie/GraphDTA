import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_max_pool as gmp

import torchnlp.nn.attention as A

from numba import jit
import numpy as np

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def fast_reshape(batch, x, x_reshaped):
    count = np.zeros(x_reshaped.shape[0], dtype=np.int8)
    for i in range(batch.shape[0]):
        idx = batch[i]
        sec_dim = count[idx]
        x_reshaped[idx, sec_dim, :] = x[i]
        count[idx] += 1

    return x_reshaped

# GAT  model
class AttnGATNet(torch.nn.Module):
    def __init__(self, num_features_xd=78, n_output=1, num_features_xt=25,
                     n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):
        super(AttnGATNet, self).__init__()

        # graph layers
        self.gcn1 = GATConv(num_features_xd, num_features_xd, heads=10, dropout=dropout)
        self.gcn2 = GATConv(num_features_xd * 10, output_dim, dropout=dropout)
        self.fc_g1 = nn.Linear(output_dim, output_dim)

        self.attention = A.Attention(128)

        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        self.fc_xt1 = nn.Linear(32*121, output_dim)

        # combined layers
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, n_output)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        # graph input feed-forward
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # protein input feed-forward:
        target = data.target
        embedded_xt = self.embedding_xt(target)


        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.gcn1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gcn2(x, edge_index)
        x = self.relu(x)

        # do attention here
        # reshape x (drug) into appropriate dim
        batch_size = target.shape[0]
        v, i = torch.mode(batch)
        v_freq = batch.eq(v.item()).sum().item()
        x_reshaped = torch.zeros([batch_size, v_freq, embedded_xt.shape[2]],
                                    dtype=torch.float64)

        # create a count for the 2nd dim
        device = x.get_device()
        x_reshaped = torch.from_numpy(fast_reshape(batch.cpu().numpy(),
                     x.cpu().detach().numpy(), x_reshaped.numpy())).to(device)

        output, weights = self.attention(embedded_xt, x_reshaped.float()) # query, context

        x = gmp(x, batch)          # global max pooling
        x = self.fc_g1(x)
        x = self.relu(x)

        conv_xt = self.conv_xt1(output)
        conv_xt = self.relu(conv_xt)

        # flatten
        xt = conv_xt.view(-1, 32 * 121)
        xt = self.fc_xt1(xt)

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
