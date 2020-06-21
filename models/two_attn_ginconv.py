import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
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

# GINConv model
class TwoAttnGINConvNet(torch.nn.Module):
    def __init__(self, n_output=1,num_features_xd=78, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):

        super(TwoAttnGINConvNet, self).__init__()

        dim = 121
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output
        # convolution layers
        nn1 = Sequential(Linear(num_features_xd, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        self.attention_1 = A.Attention(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.conv_xt_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters, kernel_size=8)
        self.attention_2 = A.Attention(dim)

        self.fc1_xd = Linear(dim, output_dim)

        self.fc1_xt = nn.Linear(32*121, output_dim)

        # combined layers
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, self.n_output)        # n_output = 1 for regression task

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target = data.target

        device = x.get_device()
        batch_size = target.shape[0]
        v, i = torch.mode(batch)
        v_freq = batch.eq(v.item()).sum().item()

        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)

        embedded_xt = self.embedding_xt(target)
        conv_xt_1 = self.conv_xt_1(embedded_xt)

        x_reshaped = torch.zeros([batch_size, v_freq, conv_xt_1.shape[2]],
                                    dtype=torch.float64)
        x_reshaped = torch.from_numpy(fast_reshape(batch.cpu().numpy(),
                     x.cpu().detach().numpy(), x_reshaped.numpy())).to(device)
        output_1, weights_1 = self.attention(conv_xt_1, x_reshaped.float()) # query, context
        pdb.set_trace()

        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)

        conv_xt_2 = self.conv_xt_2(output_1)

        batch_size = target.shape[0]
        v, i = torch.mode(batch)
        v_freq = batch.eq(v.item()).sum().item()
        x_reshaped = torch.zeros([batch_size, v_freq, conv_xt_2.shape[2]],
                                    dtype=torch.float64)
        x_reshaped = torch.from_numpy(fast_reshape(batch.cpu().numpy(),
                     x.cpu().detach().numpy(), x_reshaped.numpy())).to(device)
        output_2, weights_2 = self.attention(conv_xt_2, x_reshaped.float()) # query, context

        # carry on with x (drug)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1_xd(x))
        x = F.dropout(x, p=0.2, training=self.training)

        # flatten
        xt = output_2.view(-1, 32 * 121)
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
