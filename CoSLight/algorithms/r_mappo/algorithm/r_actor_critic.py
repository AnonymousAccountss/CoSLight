import torch
import torch.nn as nn
from onpolicy.algorithms.utils.util import init, check
from onpolicy.algorithms.utils.cnn import CNNBase
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.act import ACTLayer, ACTLayer_TopK
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.utils.util import get_shape_from_obs_space

from onpolicy.algorithms.r_mappo.algorithm.trans_net import *
from gym import spaces
import numpy as np

class R_Actor_Trans(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(R_Actor_Trans, self).__init__()
        
        self.args = args
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        # base = CNNBase if len(obs_shape) == 3 else MLPBase
        # self.base = base(args, obs_shape)

        # if self._use_naive_recurrent_policy or self._use_recurrent_policy:
        #     self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        # self.hidden_size = 64
        self.model = TransformerEncoderModel(args=args, obs_dim=obs_shape[0], hidden_size=self.hidden_size, output_size=args.trans_hidden, 
                                             num_layers=args.trans_layers, num_heads=args.trans_heads, dropout=0.1, device=device)
        
        if args.use_trans_hidden:   
            self.hidden_size = args.trans_hidden
        else:
            self.hidden_size = 2 * 7 * 8 # 7 features 8 phases
            
        self.act = ACTLayer(action_space, self.hidden_size + self.hidden_size, self._use_orthogonal, self._gain).to(device)
        
        if args.use_kl:
            group_action_space = spaces.Discrete(self.args.num_agents)
            self.score = ACTLayer_TopK(group_action_space, self.hidden_size, self._use_orthogonal, self._gain, args=args).to(device)
            # init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][args.use_orthogonal]
            # def init_(m): 
            #     return init(m, init_method, lambda x: nn.init.constant_(x, 0), args.gain)
            # self.score = init_(nn.Linear(self.hidden_size, self.args.num_agents))
            
        self.to(device)

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False, trans_masks=None):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        roll_outs = self.args.n_rollout_threads
        obs = np.array(np.split(obs, roll_outs))
        trans_masks = np.array(np.split(trans_masks, roll_outs))
        
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        trans_masks = check(trans_masks).to(**self.tpdv)
        
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        
            
        actor_features_ = self.model(obs, src_mask=trans_masks)  # 2.8.32
        actor_features = actor_features_.reshape(-1, actor_features_.shape[-1]) # 16.32
        scores, score_log_probs = self.score(actor_features, deterministic=deterministic) # 16.5 16.5
        scores_ = scores.reshape(-1, self.args.num_agents, self.args.use_K) # 2.8.5
        
        group_con = torch.gather(actor_features_.unsqueeze(1).expand(-1, self.args.num_agents, -1, -1), dim=2, \
                                 index=scores_.unsqueeze(-1).expand(-1, -1, -1, actor_features_.size(-1))) # 2.8.5.32
        group_con = group_con.mean(2) # 2.8.32
        
        actor_features_new = torch.cat((actor_features_, group_con), dim=-1)
        actor_features_new = actor_features_new.reshape(-1, actor_features_new.shape[-1]) # 16.32
             
        actions, action_log_probs = self.act(actor_features_new, available_actions, deterministic)
        

        return actions, action_log_probs, rnn_states, actor_features_new, scores, score_log_probs

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None, score_batch=None, trans_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param rnn_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv).long()

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)
            trans_masks = check(trans_masks).to(**self.tpdv)

        # actor_features = self.base(obs)
        
        actor_features_ = self.model(obs, src_mask=trans_masks)  # 20.8.64
        
        if len(actor_features_.shape) == 3: # reshape  
            score_batch_ = check(score_batch).to(**self.tpdv)
            
            T_, B_ = actor_features_.shape[0], actor_features_.shape[1]
            actor_features = actor_features_.reshape(T_*B_, -1)
            action = action.reshape(T_*B_, -1)
            available_actions = available_actions.reshape(T_*B_, -1)
            active_masks = active_masks.reshape(T_*B_, -1)
            
            score_batch = score_batch_.reshape(T_*B_, -1)
            score_action_log_probs, score_dist_entropy, action_logits = self.score.evaluate_actions(actor_features, score_batch)
            score_action_log_probs = score_action_log_probs.reshape(T_, B_, -1)
            action_logits = action_logits.reshape(T_, B_, -1)
            
            
            group_con = torch.gather(actor_features_.unsqueeze(1).expand(-1, self.args.num_agents, -1, -1), dim=2, \
                                 index=score_batch_.unsqueeze(-1).expand(-1, -1, -1, actor_features_.size(-1)).long()) # 2.8.5.32
            group_con = group_con.mean(2) # 120.8.32
            
            actor_features_new = torch.cat((actor_features_, group_con), dim=-1)
            actor_features_new = actor_features_new.reshape(T_*B_, -1) # 16.32
        
            action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features_new,
                                                                    action, available_actions,
                                                                    active_masks=
                                                                    active_masks if self._use_policy_active_masks
                                                                    else None)
            action_log_probs = action_log_probs.reshape(T_, B_, 1)

        else:
            action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features,
                                                                    action, available_actions,
                                                                    active_masks=
                                                                    active_masks if self._use_policy_active_masks
                                                                    else None)
            
                
        return action_log_probs, dist_entropy, score_action_log_probs, score_dist_entropy, action_logits



class R_Critic_all(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO) or
                            local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(R_Critic_all, self).__init__()
        self.args = args
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self.device = device
        
        self.share_obs_space = cent_obs_space
        
        self.critics = [R_Critic(args, self.share_obs_space, self.device) for _ in range(args.num_agents)]
        
        # list(model1.parameters())+list(model2.parameters())
        
        # self.critic_optimizers = [
        #     torch.optim.Adam(self.critics[i].parameters(), lr=self.critic_lr, eps=self.opti_eps, weight_decay=self.weight_decay)
        #     for i in range(args.num_agents)
        # ]

        self.to(device)
        
        

    def forward(self, cent_obs, rnn_states, masks, topk_index=None, backward=False):
        
        if not backward:
            roll_outs = self.args.n_rollout_threads
            if cent_obs.type() == 'torch.cuda.FloatTensor':
                cent_obs = cent_obs.cpu().detach().numpy()  ## 16. 32
                
            cent_obs = np.array(np.split(cent_obs, roll_outs))
            masks = np.array(np.split(masks, roll_outs))
            rnn_states = np.array(np.split(rnn_states, roll_outs))
            
        values, rnn_states_ = [], []
        for i in range(self.args.num_agents):
            value, rnn_state = self.critics[i](cent_obs[:, i], rnn_states[:, i], masks[:, i])
            values.append(value)
            rnn_states_.append(rnn_state)

        return torch.vstack(values), torch.vstack(rnn_states_)



class R_Actor(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(R_Actor, self).__init__()
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain)

        self.to(device)

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
        
        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)

        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param rnn_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv).long()

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features,
                                                                   action, available_actions,
                                                                   active_masks=
                                                                   active_masks if self._use_policy_active_masks
                                                                   else None)

        return action_log_probs, dist_entropy


class R_Critic(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO) or
                            local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(R_Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        base = CNNBase if len(cent_obs_shape) == 3 else MLPBase
        
        
        if args.use_trans_hidden:   
            self.base = base(args, (args.trans_hidden+args.trans_hidden,))
        else:
            trans_hidden = 2 * 7 * 8 # 7 features 8 phases
            self.base = base(args, (trans_hidden+trans_hidden,))
        
        # self.base = base(args, cent_obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks, backward=False):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        
        if len(cent_obs.shape) == 3:
            T_, B_ = cent_obs.shape[0], cent_obs.shape[1]
            cent_obs = cent_obs.reshape(T_*B_, -1)
            rnn_states = rnn_states.reshape(T_*B_, *rnn_states.shape[2:])
            masks = masks.reshape(T_*B_, -1)

            critic_features = self.base(cent_obs)
            if self._use_naive_recurrent_policy or self._use_recurrent_policy:
                critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
            values = self.v_out(critic_features)
            
            values = values.reshape(T_, B_, 1)
        else:
            critic_features = self.base(cent_obs)
            if self._use_naive_recurrent_policy or self._use_recurrent_policy:
                critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
            values = self.v_out(critic_features)

        return values, rnn_states




class R_Critic_Trans_all(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO) or
                            local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(R_Critic_Trans_all, self).__init__()
        self.args = args
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self.device = device
        
        self.share_obs_space = cent_obs_space
        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        
        # self.share_backbone = TransformerEncoderModel(obs_dim=cent_obs_shape[0], hidden_size=32, output_size=self.args.hidden_size, num_layers=1, num_heads=2, dropout=0.1, device=device)
        self.share_backbone = TransformerEncoderModel(obs_dim=cent_obs_shape[0], hidden_size=32, output_size=self.args.hidden_size, num_layers=2, num_heads=4, dropout=0.1, device=device)
    
        self.critics = [R_Critic_Trans(args, self.share_obs_space, self.device, share_backbone=self.share_backbone) for _ in range(args.num_agents)]
        
        # self.critic_optimizers = [
        #     torch.optim.Adam(self.critics[i].parameters(), lr=self.critic_lr, eps=self.opti_eps, weight_decay=self.weight_decay)
        #     for i in range(args.num_agents)
        # ]

        self.to(device)
        
        

    def forward(self, obs, rnn_states, masks, topk_index, backward=False):
        # B_shape = obs.shape[0]
        rnn_states_bk = rnn_states
        if not backward:
            roll_outs = self.args.n_rollout_threads
            obs = np.array(np.split(obs, roll_outs))
            masks = np.array(np.split(masks, roll_outs))
            rnn_states = np.array(np.split(rnn_states, roll_outs))
        else:
            roll_outs = obs.shape[0]
            
        values, rnn_states_ = [], []
  
        for i_agent in range(self.args.num_agents):
            obs_i = obs[torch.arange(roll_outs).unsqueeze(1), topk_index[:, i_agent], :]  # 2.2.56
            masks_i = masks[torch.arange(roll_outs).unsqueeze(1), topk_index[:, i_agent], :]  # 2.2.56
            rnn_states_i = rnn_states[torch.arange(roll_outs).unsqueeze(1), topk_index[:, i_agent]]  # 2.2.56
            
            if not backward:
                obs_i = obs_i.reshape(-1, obs_i.shape[-1])
            value, rnn_state = self.critics[i_agent](obs_i, masks_i, rnn_states_i, backward)   
            values.append(value)
            rnn_states_.append(rnn_states_i)

        return torch.vstack(values), torch.from_numpy(rnn_states_bk)



class R_Critic_Trans(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO) or
                            local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, cent_obs_space, device=torch.device("cpu"), share_backbone=None):
        super(R_Critic_Trans, self).__init__()
        
        self.args = args
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
        
        self.share_obs_space = cent_obs_space
        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        
        # The patch position encoding is a trainable parameter
        embed_dim_inner = 32
        self.class_token_encoding = nn.Parameter(torch.zeros(1, 1, embed_dim_inner))
        # self.class_token_encoding = nn.Parameter(torch.zeros(1, args.num_agents, embed_dim_inner))
        nn.init.trunc_normal_(self.class_token_encoding, mean=0.0, std=0.02)
        
        # class_token
        if share_backbone is None: #### 
            self.base = TransformerEncoderModel(obs_dim=cent_obs_shape[0], hidden_size=embed_dim_inner, output_size=self.args.hidden_size, 
                                                num_layers=2, num_heads=4, dropout=0.1, device=device, class_token=self.class_token_encoding)
        else: 
            self.base = share_backbone

        # base = CNNBase if len(cent_obs_shape) == 3 else MLPBase
        # self.base = base(args, cent_obs_shape)

        # if self._use_naive_recurrent_policy or self._use_recurrent_policy:
        #     self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)
        
        # self.hidden_size = 64
        
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))
        self.to(device)

    def forward(self, cent_obs, rnn_states, masks, backward=False):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        # if not backward:
        #     roll_outs = self.args.n_rollout_threads
        #     cent_obs = np.array(np.split(cent_obs, roll_outs))
    
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        
        

        critic_features = self.base(cent_obs) # 2.9.64
        # actor_features = actor_features.reshape(-1, actor_features.shape[-1]) # 16.64
        # if self._use_naive_recurrent_policy or self._use_recurrent_policy:
        #     critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        values = self.v_out(critic_features[:,-1])    # 2.1
        if self.args.use_K == 0:
            if backward:
                values = values[:,None].expand(-1, self.args.num_agents, -1)
            else:
                values = values[:,None].expand(-1, self.args.num_agents, -1).reshape(-1, 1)
            
        return values, rnn_states



