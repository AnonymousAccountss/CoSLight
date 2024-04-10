import torch
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor, R_Critic, R_Actor_Trans, R_Critic_Trans, R_Critic_Trans_all, R_Critic_all
from onpolicy.utils.util import update_linear_schedule
from onpolicy.algorithms.utils.util import init, check


import numpy as np

class R_MAPPOPolicy_Trans:
    """
    MAPPO Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        
        self.args = args
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.actor = R_Actor_Trans(args, self.obs_space, self.act_space, self.device)
        self.critic = R_Critic(args, self.share_obs_space, self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                    lr=self.critic_lr,
                                                    eps=self.opti_eps,
                                                    weight_decay=self.weight_decay)
        
        # if not self.args.use_trans_critic: ##### pay attention here
        #     if self.args.use_K != 0:
        #         self.critic = R_Critic_all(args, self.share_obs_space, self.device)
        #     else:
        #         self.critic = R_Critic(args, self.share_obs_space, self.device)
        # else:
        #     if self.args.use_K != 0:
        #         self.critic = R_Critic_Trans_all(args, self.share_obs_space, self.device)
        #     else:
        #         self.critic = R_Critic_Trans(args, self.share_obs_space, self.device)

        
        # if self.args.use_K == 0:
        #     self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
        #                                             lr=self.critic_lr,
        #                                             eps=self.opti_eps,
        #                                             weight_decay=self.weight_decay)
        # else:
        #     lists_parameters = []
        #     [lists_parameters.extend(list(self.critic.critics[i].parameters())) for i in range(self.args.num_agents)]
        #     self.critic_optimizer = torch.optim.Adam(lists_parameters,
        #                                             lr=self.critic_lr,
        #                                             eps=self.opti_eps,
        #                                             weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                    deterministic=False, trans_masks=None):
        """
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
        #### 
        if np.random.uniform() > self.args.epsilon:
            deterministic = True
            
        actions, action_log_probs, rnn_states_actor, actor_features_new, score, score_log_probs = self.actor(obs,
                                                                 rnn_states_actor,
                                                                 masks,
                                                                 available_actions,
                                                                 deterministic, 
                                                                 trans_masks=trans_masks
                                                                 )

       

        values, rnn_states_critic = self.critic(actor_features_new, rnn_states_critic, masks)
            
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic, score, score_log_probs, actor_features_new

    def get_values(self, cent_obs, rnn_states_critic, masks):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """
        if self.args.use_K == 0:
            values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        else:
            roll_outs = self.args.n_rollout_threads
            obs_ = np.array(np.split(cent_obs, roll_outs))
            obs_ = check(obs_).to(**self.tpdv)
            atten_scores = self.actor.model.compute_score(obs_)  ## 2.8.8
            batch_size_obs = obs_.shape[0]
            topk_index = torch.topk(atten_scores, k=self.args.use_K, dim=-1)[1] #   20.16.5
            # # 
            batch_indices = torch.arange(self.args.num_agents)[None,:,None].repeat(batch_size_obs, 1, 1).to(self.device) # 20.16.1
            topk_index = torch.cat((topk_index, batch_indices), dim=-1) 
            
            values, _ = self.critic(cent_obs, rnn_states_critic, masks, topk_index.cpu())
            
        return values

    
    
    def get_values_three(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                    deterministic=False, trans_masks=None):
        actions, action_log_probs, rnn_states_actor, actor_features, scores, score_log_probs = self.actor(obs,
                                                                 rnn_states_actor,
                                                                 masks,
                                                                 available_actions,
                                                                 deterministic,
                                                                 trans_masks=trans_masks)
        values, _ = self.critic(actor_features, rnn_states_critic, masks)
            
        return values

    
    
    
    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None, score_batch=None, actor_features_batch=None, trans_masks=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param action: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        
        # if obs.shape == 3:
        #     obs = obs.reshape(-1, *obs.shape[2:])
        #     rnn_states_actor = rnn_states_actor.reshape(-1, *rnn_states_actor.shape[2:])
        #     rnn_states_critic = rnn_states_critic.reshape(-1, *rnn_states_critic.shape[2:])
        #     masks = masks.reshape(-1, *obs.shape[2:])
        #     available_actions = available_actions.reshape(-1, *available_actions.shape[2:])
        #     action = action.reshape(-1, *action.shape[2:])
        #     active_masks = active_masks.reshape(-1, *active_masks.shape[2:])
        # action_log_probs, dist_entropy, actor_features =  
        action_log_probs, dist_entropy, score_action_log_probs, score_dist_entropy, score_action_logits = self.actor.evaluate_actions(obs,
                                                                     rnn_states_actor,
                                                                     action,
                                                                     masks,
                                                                     available_actions,
                                                                     active_masks,
                                                                     score_batch,
                                                                     trans_masks=trans_masks
                                                                     )
       
        # if len(masks.shape) == 3:
        #     rnn_states_critic = rnn_states_critic.reshape(-1, *rnn_states_critic.shape[2:])
        #     masks = masks.reshape(-1, *masks.shape[2:])
        #     actor_features_batch = actor_features_batch.reshape(-1, *actor_features_batch.shape[2:])
            
        values, _ = self.critic(actor_features_batch, rnn_states_critic, masks, backward=True)
        
        return values, action_log_probs, dist_entropy, score_action_log_probs, score_dist_entropy, score_action_logits

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor



class R_MAPPOPolicy:
    """
    MAPPO Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        self.actor = R_Actor(args, self.obs_space, self.act_space, self.device)
        self.critic = R_Critic(args, self.share_obs_space, self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                    deterministic=False):
        """
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
        actions, action_log_probs, rnn_states_actor = self.actor(obs,
                                                                 rnn_states_actor,
                                                                 masks,
                                                                 available_actions,
                                                                 deterministic)

        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param action: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs,
                                                                     rnn_states_actor,
                                                                     action,
                                                                     masks,
                                                                     available_actions,
                                                                     active_masks)

        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor
