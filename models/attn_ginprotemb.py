import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

from numba import jit
import pdb

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def fast_reshape(batch, x, x_reshaped):
    count = np.zeros(x_reshaped.shape[0], dtype=np.int8)
    for i in range(batch.shape[0]):
        idx = batch[i]
        sec_dim = count[idx]
        x_reshaped[idx, sec_dim, :] = x[i]
        count[idx] += 1

    return x_reshaped

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def shapeback(output, batch, x): 
    batch_size = batch.shape[0]
    count = np.zeros(batch_size, dtype=np.int8)
    for i in range(1, batch_size):
        if batch[i] == batch[i-1]:
            count[i] = count[i-1] + 1

    output_reshaped = np.zeros_like(x)
    for i in range(batch_size):
        first_dim = batch[i]
        sec_dim = count[i]
        output_reshaped[i, :] = output[first_dim, sec_dim, :]

    return output_reshaped

# GINConv model
class AttnGINProtEmb(torch.nn.Module):
    def __init__(self, n_output=1,num_features_xd=78, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):

        super(AttnGINProtEmb, self).__init__()

        dim = 128
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

        self.fc1_xd = Linear(dim, output_dim)

        # protein sequence embedding
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.fc1_xt = nn.Linear(128 * 1000, output_dim)

        # combined layers
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, self.n_output)        # n_output = 1 for regression task

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target = data.target

        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)

        embedded_xt = self.embedding_xt(target)

        # Attention mechanism here
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

        output, weights = self.attention(x_reshaped.float(), embedded_xt) # query, context
        output_reshaped = torch.from_numpy(shapeback(output.cpu().detach().numpy(), 
                                batch.cpu().numpy(), x.cpu().detach().numpy())).to(device)

        x = global_add_pool(output_reshaped, batch)
        x = F.relu(self.fc1_xd(x))
        x = F.dropout(x, p=0.2, training=self.training)

        # flatten
        xt = embedded_xt.view(-1, 1000 * 128)
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
