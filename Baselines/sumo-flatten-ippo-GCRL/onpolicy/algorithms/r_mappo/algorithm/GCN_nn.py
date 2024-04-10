import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GraphConvolutionLayer(nn.Module):
    """
    Simple GCN layer
    """
    def __init__(self, in_features, out_features, device='cpu'):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        
        # 定义可训练参数，即GCN中的权重矩阵W
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features), device=self.device))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # 使用xavier初始化
        self.to(device)
    
    def forward(self, feature, adj):
        """
        feature: 输入特征矩阵 [N, in_features]
        adj: 图的邻接矩阵 [N, N]
        """
        h = torch.mm(feature, self.W)   # [N, out_features]
        output = torch.mm(adj, h)       # [N, out_features]
        return output
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
    

class GCN(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, dropout, node_num, n_thr, device='cpu'):
        """
        Dense version of GCN.
        """
        super(GCN, self).__init__()
        self.dropout = dropout
        self.device = device
        self.node_num = node_num
        self.node_num_batch = node_num * n_thr
        
        # 定义图卷积层
        self.gc1 = GraphConvolutionLayer(n_feat, n_hid, device=self.device)
        self.gc2 = GraphConvolutionLayer(n_hid, n_class, device=self.device)
    
    
    def to_adj(self, edge_index):
        # adj = torch.zeros(self.node_num_batch, self.node_num_batch, device=self.device) ## 其实是 node_num* n_thr
        adj = torch.zeros(edge_index.shape[0], edge_index.shape[0], device=self.device) ## 其实是 node_num* n_thr
        for i in range(edge_index.shape[0]):
            for j in edge_index[i]:
                if j != -1:
                    div_n = i // self.node_num
                    if div_n >= 1:
                        j_ind = j + self.node_num * div_n
                    else:
                        j_ind = j
                    adj[i][j_ind.long()] = 1
                    
        return adj.float()
    
    def forward(self, x, edge_index):
        adj = self.to_adj(edge_index).to(self.device)
        
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


