import time
import numpy as np
import torch
from onpolicy.runner.shared.base_runner import Runner
import wandb
import imageio

def _t2n(x):
    return x.detach().cpu().numpy()

class SUMORunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the MPEs. See parent class for details."""
    def __init__(self, config):
        super(SUMORunner, self).__init__(config)

    def run(self):
        
        self.epsilon = self.all_args.epsilon
        self.anneal_epsilon = (self.all_args.epsilon - self.all_args.min_epsilon) / self.all_args.anneal_steps
        
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        if self.all_args.not_update:
            episodes  = 20
        
        episode_score_1, episode_score_2 = [], []

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
                
            for step in range(self.episode_length):
                print('-----step', step)
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env, score, score_log_probs, actor_features = self.collect(step)
                    
                # Obser reward and next obs
                ###### tsc specified actions
                obs, rewards, dones, infos = self.envs.step(actions.astype(np.int64)[:,:,0])
                episode_score_1.append(score[0])
                episode_score_2.append(score[1])
                
                self.ava = self.envs.get_unava_phase_index()
                available_actions = self.get_ava_actions(self.ava)

                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, available_actions, score, score_log_probs, actor_features

                # insert data into buffer
                self.insert(data)
                
                ##### episide decay
                self.all_args.epsilon = self.all_args.epsilon - self.anneal_epsilon if self.all_args.epsilon > self.all_args.min_epsilon else self.all_args.epsilon

            # if self.all_args.not_update:
            #     save_dir = '/home/xingdp/jqruan/data/TSC/sumo-flatten-ours-v9-sumolib-final-ablation-K/onpolicy/scripts/results_sumo/SUMO/static_K1_team_id/' + self.all_args.sumocfg_files.split('/')[-2]
            #     import os
            #     if not os.path.exists(save_dir):
            #         os.makedirs(save_dir)
                
            #     np.save(f'{save_dir}/K{self.all_args.use_K}_episode_score_1.npy', np.array(episode_score_1), allow_pickle=True)
            #     np.save(f'{save_dir}/K{self.all_args.use_K}_episode_score_2.npy', np.array(episode_score_2), allow_pickle=True)
            
            if not self.all_args.not_update:
                # compute return and update network
                self.warmup()   
                
                self.compute()
                train_infos = self.train()
                
                # post process
                total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
                
                # save model
                if (episode % self.save_interval == 0 or episode == episodes - 1):
                    self.save(episode)

                # log information
                if episode % self.log_interval == 0:
                    end = time.time()
                    print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                            .format(self.all_args.scenario_name,
                                    self.algorithm_name,
                                    self.experiment_name,
                                    episode,
                                    episodes,
                                    total_num_steps,
                                    self.num_env_steps,
                                    int(total_num_steps / (end - start))))

                    if self.env_name == "SUMO":
                        env_infos = {}
                        # for agent_id in range(self.num_agents):
                        #     idv_rews = []
                        #     for info in infos:
                        #         if 'individual_reward' in info.keys():
                        #             idv_rews.append(info['individual_reward'][agent_id][0])
                                
                        #     agent_k = 'agent%i/individual_rewards' % agent_id
                        #     env_infos[agent_k] = idv_rews
                        for info in infos:
                            for agent_id in range(self.num_agents):
                                for k, v in info[list(info.keys())[agent_id]].items():
                                    if k not in env_infos:
                                        env_infos[k] = []
                                    env_infos[k].append(v)  

                    train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                    print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                    self.log_train(train_infos, total_num_steps)
                    self.log_env(env_infos, total_num_steps)

                # eval
                if episode % self.eval_interval == 0 and self.use_eval:
                    self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs = self.envs.reset()

        # replay buffer
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.ava = self.envs.get_unava_phase_index()
        
        available_actions = self.get_ava_actions(self.ava)
        self.buffer.available_actions[0] = available_actions.copy()
        
        #################### 填充actor features：： TODO！
        
    def get_ava_actions(self, ava):
        available_actions = np.ones((self.all_args.n_rollout_threads, self.all_args.num_agents, self.all_args.num_actions))
        if len(ava.shape) == 2: 
            for i in range(self.all_args.num_agents):
                for j in range(self.all_args.n_rollout_threads):
                    available_actions[j, i, ava[j][i]] = 0
        elif ava is not None and ava.shape[-1] != 0:
            for i in range(self.all_args.n_rollout_threads):
                for j in range(self.all_args.num_agents):
                    available_actions[i, j, ava[i][j][0]] = 0
            
        return available_actions
    
    
    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        
        if self.all_args.part_mask:
             trans_masks = np.concatenate(self.buffer.trans_masks[step-1])
        else:
             trans_masks = np.concatenate(self.buffer.trans_masks[step])
            
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            , score, score_log_probs, actor_features\
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                            np.concatenate(self.buffer.obs[step]),
                            np.concatenate(self.buffer.rnn_states[step]),
                            np.concatenate(self.buffer.rnn_states_critic[step]),
                            np.concatenate(self.buffer.masks[step]),
                            available_actions=np.concatenate(self.buffer.available_actions[step]),
                            trans_masks = trans_masks
                            )
        # available_actions
        # [self.envs, agents, dim]
        score = np.array(np.split(_t2n(score), self.n_rollout_threads))
        score_log_probs = np.array(np.split(_t2n(score_log_probs), self.n_rollout_threads))
        actor_features = np.array(np.split(_t2n(actor_features), self.n_rollout_threads))
        
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        # rearrange action
        if self.envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
            for i in range(self.envs.action_space[0].shape):
                uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
        elif self.envs.action_space[0].__class__.__name__ == 'Discrete':
            actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
        else:
            raise NotImplementedError

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env, score, score_log_probs, actor_features

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, available_actions, score, score_log_probs, actor_features = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks, available_actions=available_actions,
                           score=score, score_log_probs=score_log_probs, actor_features=actor_features)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(np.concatenate(eval_obs),
                                                np.concatenate(eval_rnn_states),
                                                np.concatenate(eval_masks),
                                                deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            
            if self.eval_envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.eval_envs.action_space[0].shape):
                    eval_uc_actions_env = np.eye(self.eval_envs.action_space[0].high[i]+1)[eval_actions[:, :, i]]
                    if i == 0:
                        eval_actions_env = eval_uc_actions_env
                    else:
                        eval_actions_env = np.concatenate((eval_actions_env, eval_uc_actions_env), axis=2)
            elif self.eval_envs.action_space[0].__class__.__name__ == 'Discrete':
                eval_actions_env = np.squeeze(np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)
            else:
                raise NotImplementedError

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos['eval_average_episode_rewards'] = np.sum(np.array(eval_episode_rewards), axis=0)
        eval_average_episode_rewards = np.mean(eval_env_infos['eval_average_episode_rewards'])
        print("eval average episode rewards of agent: " + str(eval_average_episode_rewards))
        self.log_env(eval_env_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        """Visualize the env."""
        envs = self.envs
        
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            obs = envs.reset()
            if self.all_args.save_gifs:
                image = envs.render('rgb_array')[0][0]
                all_frames.append(image)
            else:
                envs.render('human')

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            
            episode_rewards = []
            
            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(np.concatenate(obs),
                                                    np.concatenate(rnn_states),
                                                    np.concatenate(masks),
                                                    deterministic=True)
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

                if envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                    for i in range(envs.action_space[0].shape):
                        uc_actions_env = np.eye(envs.action_space[0].high[i]+1)[actions[:, :, i]]
                        if i == 0:
                            actions_env = uc_actions_env
                        else:
                            actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
                elif envs.action_space[0].__class__.__name__ == 'Discrete':
                    actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
                else:
                    raise NotImplementedError

                # Obser reward and next obs
                obs, rewards, dones, infos = envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = envs.render('rgb_array')[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)
                else:
                    envs.render('human')

            print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))

        if self.all_args.save_gifs:
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)
