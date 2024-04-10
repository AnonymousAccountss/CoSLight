import torch
import torch.nn as nn
import math

from onpolicy.algorithms.utils.util import init, check
from onpolicy.algorithms.r_mappo.algorithm.frap_net import *


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, device='cpu'):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0).transpose(0, 1)
        self.tpdv = dict(dtype=torch.float32, device=device)

    def forward(self, x):
        self.pe = check(self.pe).to(**self.tpdv)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)



class PositionalEncoding_Emb(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, device='cpu'):
        super(PositionalEncoding_Emb, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Embedding(max_len, d_model)
        self.tpdv = dict(dtype=torch.float32, device=device)

    def forward(self, x): 
        self.pe = check(self.pe).to(**self.tpdv)     # # 1, N, x
        
        range_ = torch.arange(0, x.shape[1])   # # 
        range_ = check(range_).to(**self.tpdv).long()
        
        x = x + self.pe(range_)
        
        return self.dropout(x)



class TransformerEncoderModel(nn.Module):
    def __init__(self, args, obs_dim, hidden_size, output_size, num_layers, num_heads, dropout, device=torch.device("cpu"), class_token=None):
        super(TransformerEncoderModel, self).__init__()
        
        self.args=args
        self._phase_embedding = ModelBody(1, args.hidden_layer_size, self.args.state_key, device=device, args=args).to(device)
        # hidden_size = args.hidden_layer_size * 7 * 8 # 7个特征， 8个phase
        hidden_size = 64
        
        # 定义观测embeddings
        # self._obs_embedding = nn.Linear(
        #     in_features=obs_dim,
        #     out_features=hidden_size,
        #     bias=False
        # ).to(device)
        
        # 定义位置嵌入层
        self.ps_encoding = PositionalEncoding_Emb(
            d_model=hidden_size,
            dropout=0,
            max_len=1000,
            device=device
        )

        # 定义 Transformer Encoder 层
        self.encoder_layer = nn.TransformerEncoderLayer(hidden_size, num_heads, hidden_size, dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers, norm=nn.LayerNorm(hidden_size))

        # 定义输出层
        # self.output_layer = nn.Linear(hidden_size, output_size)
        self.output_layer = nn.Linear(hidden_size, args.trans_hidden)
        
        if class_token is not None:
            self.class_token = class_token
        else:
            self.class_token = None
        
        self.to(device)
        

    def forward(self, src, src_mask=None): # src.shape : T, B, x  这里指的是 T=1个时刻步, B=agent number;; src_mask:16,8
        T_batch, num_agents = src.shape[0], src.shape[1]
        phase_emb = self._phase_embedding(src.reshape(T_batch*num_agents, -1))
        x = phase_emb.reshape(T_batch, num_agents, -1)
        
        # src = src.permute(1, 0, 2)    # T, B, x  --> B, T, x 
        # x = self._obs_embedding(src)  # B, T, d_model
        # x = x.permute(1, 0, 2)        # T, B, x
        
        if self.class_token is not None:
            x = torch.cat((x, self.class_token.expand(T_batch, -1, -1)), dim=1)
        # 进行位置嵌入
        x_pos = self.ps_encoding(x)   # T, B or B+1, x

        # 添加mask
        src_mask = src_mask.repeat(self.args.trans_heads, 1, 1)
        # if src_mask is None:
        #     max_len = src.shape[0]
        #     src_mask = self.transformer_encoder.generate_square_subsequent_mask(max_len).to(x.device) 
        x_pos = x_pos.permute(1, 0, 2)        # B or B+1, T, x    ##### `(S, N, E)
        x = self.transformer_encoder(x_pos, src_mask)  ### transfomer的输入 S(序列长度，这里指的是agent个数), N(batchsize，这里是1个时刻), F(特征维度) 
        
        # 输出预测结果
        if self.args.use_trans_hidden:
            x = self.output_layer(x)      # # B or B+1, T, x 
            
            
        x = x.permute(1, 0, 2)        # T, B or B+1, 64
        
        return x
    
    
    def compute_score(self, src, src_mask=None): # src.shape : T, B, x  这里指的是 T=1个时刻步, B=agent number
        T_batch, num_agents = src.shape[0], src.shape[1]
        phase_emb = self._phase_embedding(src.reshape(T_batch*num_agents, -1))
        x = phase_emb.reshape(T_batch, num_agents, -1)
        
        # src = src.permute(1, 0, 2)    # T, B, x  --> B, T, x 
        # x = self._obs_embedding(src)  # B, T, d_model
        # x = x.permute(1, 0, 2)        # T, B, x
        
        # 进行位置嵌入
        x_pos = self.ps_encoding(x)   # T, B, x
        x_pos_ = x_pos.permute(1, 0, 2)  # B, T, d_model
        
        src_mask = src_mask.repeat(self.args.trans_heads, 1, 1)
        h, attn = self.transformer_encoder(x_pos_, src_mask, output_att=True)  ### transfomer的输入 S(序列长度，这里指的是agent个数), N(batchsize，这里是1个时刻), F(特征维度) 

        # h, attn = self.transformer_encoder.layers[-1].self_attn(x_pos_, x_pos_, x_pos_, src_mask)    

        return attn


if __name__ == '__main__':
    # device = torch.device("cuda:0")
    obs = torch.rand(1, 32, 56)
    # 使用自定义位置编码
    model = TransformerEncoderModel(obs_dim=56, hidden_size=32, output_size=64, num_layers=2, num_heads=4, dropout=0.1)
    x = model(obs)
    print(x.shape) # 32.1.64
