import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch.nn as nn
import torch
from torch.nn.init import xavier_normal_

import numpy as np

# from onpolicy.algorithms.r_mappo.algorithm.GAT_nn import *

def sequential_pack(layers):
    assert isinstance(layers, list)
    seq = nn.Sequential(*layers)
    for item in layers:
        if isinstance(item, nn.Conv2d) or isinstance(item, nn.ConvTranspose2d):
            seq.out_channels = item.out_channels
            break
        elif isinstance(item, nn.Conv1d):
            seq.out_channels = item.out_channels
            break
    return seq


def conv2d_block(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    pad_type='zero',
    activation=None,
):
    block = []
    assert pad_type in ['zero', 'reflect', 'replication'], "invalid padding type: {}".format(pad_type)
    if pad_type == 'zero':
        pass
    elif pad_type == 'reflect':
        block.append(nn.ReflectionPad2d(padding))
        padding = 0
    elif pad_type == 'replication':
        block.append(nn.ReplicationPad2d(padding))
        padding = 0
    block.append(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation, groups=groups)
    )
    xavier_normal_(block[-1].weight)
    if activation is not None:
        block.append(activation)
    return sequential_pack(block)


def fc_block(
        in_channels,
        out_channels,
        activation=None,
        use_dropout=False,
        norm_type=None,
        dropout_probability=0.5
):
    block = [nn.Linear(in_channels, out_channels)]
    xavier_normal_(block[-1].weight)
    if norm_type is not None and norm_type != 'none':
        if norm_type == 'LN':
            block.append(nn.LayerNorm(out_channels))
        else:
            raise NotImplementedError
    if isinstance(activation, torch.nn.Module):
        block.append(activation)
    elif activation is None:
        pass
    else:
        raise NotImplementedError
    if use_dropout:
        block.append(nn.Dropout(dropout_probability))
    return sequential_pack(block)


class ModelBody(nn.Module):
    def __init__(self, input_size, fc_layer_size, state_keys, device='cpu', args=None):  # (1, 4, ['current_phase', 'car_num', 'queue_length', 'occupancy', 'flow', 'stop_car_num'])
        self.state_keys = state_keys
        super(ModelBody, self).__init__()
        self.name = 'model_body'
        self.device = device
        if args is not None:
            self.args = args
        
        self.fc_layer_size = fc_layer_size
        # mlp
        self.fc_car_num = fc_block(input_size, fc_layer_size, activation=nn.Sigmoid())
        self.fc_queue_length = fc_block(input_size, fc_layer_size, activation=nn.Sigmoid())
        self.fc_occupancy = fc_block(input_size, fc_layer_size, activation=nn.Sigmoid())
        self.fc_flow = fc_block(input_size, fc_layer_size, activation=nn.Sigmoid())
        self.fc_stop_car_num = fc_block(input_size, fc_layer_size, activation=nn.Sigmoid())
        
        if self.args.use_pressure:
            self.fc_pressure = fc_block(input_size, fc_layer_size, activation=nn.Sigmoid())
        
        # current phase
        self.current_phase_act = nn.Sigmoid()
        self.current_phase_embedding = nn.Embedding(2, self.fc_layer_size)
        # mask
        self.mask_act = nn.Sigmoid()
        self.mask_embedding = nn.Embedding(2, self.fc_layer_size)
        # dirct liner
        self.dirct_fc = fc_block(24, 32, activation=nn.Sigmoid())
        # relation_embedding
        self.relation_embedding = nn.Embedding(2, 16)
        PHASE_LIST =  [
            'WT_ET',
            'EL_ET',
            'WL_WT',
            'WL_EL',
            'NT_ST',
            'SL_ST',
            'NT_NL',
            'NL_SL'
        ]

        self.constant = self.relation(PHASE_LIST, device)
        # self.constant = array([[[1, 1, 0, 0, 0, 0, 0],
        #                         [1, 0, 1, 0, 0, 0, 0],
        #                         [1, 0, 1, 0, 0, 0, 0],
        #                         [0, 1, 1, 0, 0, 0, 0],
        #                         [0, 0, 0, 0, 1, 1, 0],
        #                         [0, 0, 0, 0, 1, 0, 1],
        #                         [0, 0, 0, 0, 1, 0, 1],
        #                         [0, 0, 0, 0, 0, 1, 1]]])
        
        # self.constant =  torch.LongTensor([[[0, 0, 0, 1, 1, 0, 0],
        #                                     [0, 0, 0, 0, 0, 1, 1],
        #                                     [0, 0, 0, 1, 1, 0, 0],
        #                                     [0, 0, 0, 0, 0, 1, 1],
        #                                     [1, 0, 1, 0, 0, 0, 0],
        #                                     [1, 0, 1, 0, 0, 0, 0],
        #                                     [0, 1, 0, 1, 0, 0, 0],
        #                                     [0, 1, 0, 1, 0, 0, 0]]], device=device)
  
        self.drict_cnn = conv2d_block(56, 32, 1, activation=nn.ReLU())
        
        self.relation_cnn = conv2d_block(16, 32, 1, activation=nn.ReLU())
        self.cnn = conv2d_block(32, 16, 1, activation=nn.ReLU())
        # self.output = conv2d_block(16, 1, 1)
        self.output = conv2d_block(16, 8, 1)
        

    def relation(self, phase_list, device):
        relations = []
        num_phase = len(phase_list)
        if num_phase == 8:
            for p1 in phase_list:
                zeros = [0, 0, 0, 0, 0, 0, 0]
                count = 0
                for p2 in phase_list:
                    if p1 == p2:
                        continue
                    m1 = p1.split("_")
                    m2 = p2.split("_")
                    if len(list(set(m1 + m2))) == 3:
                        zeros[count] = 1
                    count += 1
                relations.append(zeros)
            relations = np.array(relations).reshape((1, 8, 7))
        else:
            relations = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]).reshape((1, 4, 3))
        # constant = torch.LongTensor(relations, device=device)
        constant = torch.tensor(relations, device=device).long()
 
        return constant

    def forward(self, input):
        names = ['current_phase', 'car_num', 'queue_length', 'occupancy', 'flow', 'stop_car_num', 'mask', 'neighbor_index', 'neighbor_dis']
        lengths = [8,8,8,8,8,8,8, 4,4]  # ['current_phase', 'car_num', 'queue_length', 'occupancy', 'flow', 'stop_car_num', neighbor]
        
        bs = input.shape[0]
        all_key_state = []
        all_key_state.append(self.fc_car_num(input[:, 8:16].reshape(-1, 1)))
        all_key_state.append(self.fc_queue_length(input[:, 16:24].reshape(-1, 1)))
        all_key_state.append(self.fc_occupancy(input[:, 24:32].reshape(-1, 1)))
        all_key_state.append(self.fc_flow(input[:, 32:40].reshape(-1, 1)))
        all_key_state.append(self.fc_stop_car_num(input[:, 40:48].reshape(-1, 1)))
        
        all_key_state.append(self.current_phase_act(self.current_phase_embedding(input[:, 0:8].long())).reshape(-1, self.fc_layer_size))
        all_key_state.append(self.mask_act(self.mask_embedding(input[:, 48:56].long())).reshape(-1, self.fc_layer_size))
        
        
        ##### 
        direct_all = torch.cat(all_key_state, dim=1).reshape(bs, 8, -1) #### 200.28  --> 25.8.28
        
        # return direct_all.reshape(bs, -1) # 8*14
        
        mix_direct = torch.cat(
            [
            torch.add(direct_all[:, 3, :], direct_all[:, 7, :]).unsqueeze(1),
            torch.add(direct_all[:, 6, :], direct_all[:, 7, :]).unsqueeze(1),
            torch.add(direct_all[:, 2, :], direct_all[:, 3, :]).unsqueeze(1),
            torch.add(direct_all[:, 2, :], direct_all[:, 6, :]).unsqueeze(1),
            torch.add(direct_all[:, 1, :], direct_all[:, 5, :]).unsqueeze(1),
            torch.add(direct_all[:, 4, :], direct_all[:, 5, :]).unsqueeze(1),
            torch.add(direct_all[:, 0, :], direct_all[:, 1, :]).unsqueeze(1),
            torch.add(direct_all[:, 0, :], direct_all[:, 4, :]).unsqueeze(1)
            ]
            , dim=1)  #  A: wt-et B: el-et C: wl-wt D: el-wl E: nt-st F: sl-st G: nt-nl H: nl-sl

        list_phase_pressure_recomb = []
        for i in range(8):
            for j in range(8):
                if i != j:
                    list_phase_pressure_recomb.append(
                        torch.cat([mix_direct[:, i, :], mix_direct[:, j, :]], dim=-1).unsqueeze(1))
        list_phase_pressure_recomb = torch.cat(
            list_phase_pressure_recomb, dim=1).reshape(bs, 8, 7, -1).permute(0, 3, 1, 2)  # torch.Size([25, 8, 7, 56]) --> torch.Size([25, 56, 8, 7])

        direct_cnn = self.drict_cnn(list_phase_pressure_recomb)   # torch.Size([25, 32, 8, 7])
        
        relation_embedding = self.relation_embedding(self.constant).permute(0, 3, 1, 2)  # torch.Size([1, 8, 7, 16]) --> torch.Size([1, 16, 8, 7])
        relation_conv = self.relation_cnn(relation_embedding)     # torch.Size([1, 32, 8, 7])
        combine_feature = direct_cnn * relation_conv              # torch.Size([25, 32, 8, 7])
        # hidden_layer = self.cnn(combine_feature)                  # torch.Size([25, 16, 8, 7])
        hidden_layer = self.cnn(combine_feature)                  # torch.Size([25, 32, 8, 7])
        
        # output = hidden_layer.sum(-1).reshape(hidden_layer.shape[0], -1)  # # torch.Size([25, 64])
        # return output
        
        output = self.output(hidden_layer).sum(-1).reshape(hidden_layer.shape[0], -1)
        return output   # torch.Size([32, 64])   
        
        #
        # if unava_phase_index:
        #     for i in range(bs):
        #         output[i, unava_phase_index[i]] = 0

        # out = self.output(hidden_layer).sum(-1).squeeze(1)
        # return hidden_layer, out   # torch.Size([25, 16, 8, 7])   torch.Size([32, 8])
        
        # return hidden_layer   # torch.Size([25, 16, 8, 7])   
    
    
