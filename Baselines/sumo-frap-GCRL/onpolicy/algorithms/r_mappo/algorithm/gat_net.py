import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        # 定义可学习的参数
        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.a = nn.Parameter(torch.FloatTensor(2 * out_features, 1))
        nn.init.xavier_uniform_(self.W.data)
        nn.init.xavier_uniform_(self.a.data)


        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class MultiHeadGATLayer(nn.Module):
    def __init__(self, nhead, in_features, out_features, dropout=0.3, alpha=0.2, concat=True):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList([GATLayer(in_features, out_features, dropout, alpha, concat) for _ in range(nhead)])
        self.concat = concat

    def forward(self, input, adj):
        if self.concat:
            return torch.cat([head(input, adj) for head in self.heads], dim=1)
        else:
            return sum([head(input, adj) for head in self.heads]) / len(self.heads)



class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout
        
        # 多头第一层
        self.attentions = nn.ModuleList([GATLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)])
        
        # 多头第二层
        self.out_attentions = nn.ModuleList([GATLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False) for _ in range(nheads)])
        
    def forward(self, x):
        adj = torch.ones((x.shape[0], x.shape[0]), device=x.device)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = sum([att(x, adj) for att in self.out_attentions]) / len(self.out_attentions)
        # return F.log_softmax(x, dim=1)
        return x


class Categorical_Topk_GAT(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01, args=None):
        super(Categorical_Topk_GAT, self).__init__()      
        if args.sumocfg_files.split('/')[-2] == 'nanshan':
            self.gat = GAT(nfeat=num_inputs, nhid=4, nclass=num_outputs, dropout=0.6, alpha=0.2, nheads=1)
        else:
            self.gat = GAT(nfeat=num_inputs, nhid=16, nclass=num_outputs, dropout=0.6, alpha=0.2, nheads=8)
        if args is not None:
            self.K = args.use_K
        else:
            self.K = 3
            
    def forward(self, x, available_actions=None, adj=None):
        x = self.gat(x)

        if available_actions is not None:
            x[available_actions == 0] = -1e10
        # return FixedCategorical_Topk(logits=x, K=self.K)
        self.probs = torch.softmax(x, dim=-1)
        self.x = x
        return x
    
    def sample(self):
        return torch.multinomial(self.probs, num_samples=self.K, replacement=False)    
    
    def log_probs(self, actions):
        return torch.gather(torch.log(self.probs), 0, actions.long())

    def mode(self):
        # return torch.topk(self.probs, self.K, dim=-1, keepdim=True)
        return torch.topk(self.probs, self.K, dim=-1)[1]
        
    
    def entropy(self):
        return FixedCategorical_Topk_GAT(logits=self.x).entropy()
    
    
class FixedCategorical_Topk_GAT(torch.distributions.Categorical):
        
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)




class GAT_ACTLayer(nn.Module):
    """
    MLP Module to compute actions.
    :param action_space: (gym.Space) action space.
    :param inputs_dim: (int) dimension of network input.
    :param use_orthogonal: (bool) whether to use orthogonal initialization.
    :param gain: (float) gain of the output layer of the network.
    """
    def __init__(self, num_inputs, num_outputs, args):
        super(GAT_ACTLayer, self).__init__()
        self.args = args
        self.action_out = Categorical_Topk_GAT(num_inputs, num_outputs, args=args)


    def forward(self, x, available_actions=None, deterministic=False):
        action_logits = self.action_out(x, available_actions)
        actions = self.action_out.mode() if deterministic else self.action_out.sample() 
        action_log_probs = self.action_out.log_probs(actions)
        
        return actions, action_log_probs

    def evaluate_actions(self, x, action, available_actions=None, active_masks=None):
        action_logits = self.action_out(x, available_actions)
        action_log_probs = self.action_out.log_probs(action)
        if active_masks is not None:
            dist_entropy = (action_logits.entropy()*active_masks.squeeze(-1)).sum()/active_masks.sum()
        else:
            dist_entropy = self.action_out.entropy().mean()
    
        return action_log_probs, dist_entropy, action_logits
     


# # 创建模型实例
# # 示例
# adj = torch.ones((10, 10))  # 

# features = torch.rand(10, 64)  # 10个节点，每个节点5个特征

# model = Categorical_Topk(64, 10)

# # 训练
# optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
# model.train()

# for epoch in range(1):  # 假设我们训练200轮
#     # optimizer.zero_grad()
#     output = model(features)
#     # loss = F.nll_loss(output, labels)  # 假设 labels 是真实的类别标签
#     # loss.backward()
#     # optimizer.step()
    
#     print(output.shape)



