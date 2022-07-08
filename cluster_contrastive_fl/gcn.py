import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, h_feats2):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats, allow_zero_in_degree=True)
        self.conv2 = GraphConv(h_feats, int(h_feats/2), allow_zero_in_degree=True)
        self.conv3 = GraphConv(int(h_feats/2), h_feats2, allow_zero_in_degree=True)

    def forward(self, g):
        in_feat = g.ndata['embedding'].float()
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        h = F.relu(h)
        h = self.conv3(g, h)
        g.ndata['h'] = h
        return dgl.mean_nodes(g, 'h')
