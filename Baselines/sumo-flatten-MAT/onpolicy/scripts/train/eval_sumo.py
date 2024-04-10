#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
from onpolicy.config import get_config
# from onpolicy.envs.mpe.MPE_env import MPEEnv
# from onpolicy.envs.sumo_files.SUMO_env import SUMOEnv

# from onpolicy.envs.sumo_files_new.SUMO_env import SUMOEnv
# from onpolicy.envs.sumo_files_new.config import config as config_env

from onpolicy.envs.sumo_files_marl.config import config as config_env
from onpolicy.envs.sumo_files_marl.SUMO_env import SUMOEnv

from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv


"""Train script for MPEs."""

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "SUMO":
                env = SUMOEnv(all_args, rank)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "SUMO":
                env = SUMOEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument("--num_agents", type=int, default=0, help="the number of the agents.")
    parser.add_argument('--scenario_name', type=str, default='sumo_test', help="Which scenario to run on")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    
    #########################################
    all_args.episode_length = config_env['episode']['rollout_length']
    all_args.num_actions = config_env['environment']['num_actions']
    

    if all_args.algorithm_name == "rmappo":
        print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo" or all_args.algorithm_name == "mat" or all_args.algorithm_name == "mat_dec":
        print("u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False")
        all_args.use_recurrent_policy = False 
        all_args.use_naive_recurrent_policy = False
        all_args.use_centralized_V = False
    elif all_args.algorithm_name == "ippo":
        print("u are choosing to use ippo, we set use_centralized_V to be False")
        all_args.use_centralized_V = False
    else:
        raise NotImplementedError

    assert (all_args.share_policy == True and all_args.scenario_name == 'simple_speaker_listener') == False, (
        "The simple_speaker_listener scenario can not use shared policy. Please check the config.py.")

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
    #     str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))
    setproctitle.setproctitle("bq-test")
    
    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                0] + "/results_sumo") / all_args.env_name / all_args.experiment_name / all_args.algorithm_name 
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    ##### 保存args
    argsDict = all_args.__dict__
    with open(str(run_dir) + '/args.txt', 'w') as f:
        f.writelines('------------------ start save arguments------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')

    # wandb
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                        project=all_args.env_name,
                        entity=all_args.user_name,
                        notes=socket.gethostname(),
                        name=str(all_args.algorithm_name) + "_" +
                        str(all_args.experiment_name) +
                        "_seed" + str(all_args.seed),
                        group=all_args.scenario_name,
                        dir=str(run_dir),
                        job_type="training",
                        reinit=True)
    else:
        curr_run = 'seed_' + str(all_args.seed) + '_'
        if not run_dir.exists():
            curr_run += 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run += 'run1'
            else:
                curr_run += 'run%i' % (max(exst_run_nums) + 1)
        
        
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))
            
    
    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    all_args.num_agents = len(envs.action_space)
    

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": all_args.num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    if all_args.share_policy:
        from onpolicy.runner.shared.sumo_runner import SUMORunner as Runner

    runner = Runner(config)
    runner.run()
    # if not all_args.use_test:
    #     runner.run()
    # else:
    #     runner.test()
    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if not all_args.use_test:
        if all_args.use_wandb:
            run.finish()
        else:
            runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
            runner.writter.close()


if __name__ == "__main__":
    for i in range(0, 1):
        args = ['--env_name', 'SUMO', '--algorithm_name', 'mat', '--num_agents', '0', '--seed', '1',  '--experiment_name', 'grid4x4-frap',
                '--n_training_threads', '8', '--n_rollout_threads', '2', '--num_mini_batch', '1', '--num_env_steps', '3000000',
                '--ppo_epoch', '3', '--gain', '0.01', '--lr', '7e-4', '--critic_lr', '7e-4', '--use_wandb', 'False', '--use_ReLU']
        # 
        args[7] = str(i)
        
        
        # cong_lst = ['grid4x4-frap-0328-test', 'fenglin-frap-0323-pre', 'nanshan-frap-0323-pre', 'arterial4x4-frap-0323-pre', 'ingolstadt21-frap-0323-pre', 'cologne8-frap-0323-pre']
        # cong_lst = ['grid4x4-gat-0403-rand_ind', 'fenglin-gat-0403-rand_ind', 'nanshan-gat-0403-rand_ind', 'arterial4x4-gat-0403-rand_ind', 'ingolstadt21-gat-0403-rand_ind', 'cologne8-gat-0403-rand_ind']
        # cong_lst = ['grid4x4-frap-eval', 'fenglin-frap-eval', 'nanshan-frap-eval', 'arterial4x4-frap-eval', 'ingolstadt21-frap-eval', 'cologne8-frap-eval']
        # eval_model_path = ['grid4x4', 'fenglin', 'nanshan', 'arterial4x4', 'ingolstadt21', 'cologne8'] 
        # cong_lst = ['grid4x4-frap-test', 'fenglin-frap-test', 'nanshan-frap-test', 'arterial4x4-frap-test', 'ingolstadt21-frap-test', 'cologne8-frap-test']
        
        
        ################# 暂时不改
        threads = ['2', '32', '64', '128']
        num_mini_batch = ['8', '32', '16', '16']
        index_ = 0
        args[13] = threads[index_] # n_rollout_threads
        args[11] = threads[index_] # n_training_threads
        args[15] = num_mini_batch[index_] # num_mini_batch
        
        args[3] = 'mat'
        # args[3] = 'mappo'
        
        import sys
        args_ = sys.argv
        index = int(args_[1]) # 训练哪个地图？ # grid44:0   art44: 3    grid55: 6   colo: 5  nanshan: 2
        # index = 0
        
        # args[9] = cong_lst[index] + '_thr_' + args[13] 
        
        

        cong_flag = f'SUMO-best-MAT/20240406/'
        
        cong_lst = ['grid4x4-MAT-', 'fenglin-MAT-', 'nanshan-MAT-',
                    'arterial4x4-MAT-', 'ingolstadt21-MAT-', 'cologne8-MAT-', 'grid5x5-MAT-']

        
        sumocfg_files_lst = [
            'sumo_files_marl/scenarios/resco_envs/grid4x4/grid4x4.sumocfg',
            "sumo_files_marl/scenarios/sumo_fenglin_base_road/base.sumocfg",
            "sumo_files_marl/scenarios/nanshan/osm.sumocfg",
            'sumo_files_marl/scenarios/resco_envs/arterial4x4/arterial4x4.sumocfg',
            'sumo_files_marl/scenarios/resco_envs/ingolstadt21/ingolstadt21.sumocfg',
            'sumo_files_marl/scenarios/resco_envs/cologne8/cologne8.sumocfg',
            'sumo_files_marl/scenarios/large_grid2/exp_0.sumocfg'
        ]
        state_key = config_env['environment']['state_key']
        args.extend(['--state_key', state_key, '--port_start', '14444', '--sumocfg_files', sumocfg_files_lst[index]])
        ################################################################
        args[9] = cong_flag + cong_lst[index] + args[3] + '_thr_' + args[13]
        
        
        
        args.extend(['--use_gat', 'True'])  #### 如果使用GAT的话
        
        ########################################  如果评测！！！！！！！！！！！！！！
        args.extend(['--not_update', 'True'])  #### 如果使用评估的话就不更新模型了
        
        id_list = [2000, -1, 200, 2000, -1, 2000, 1800]
        model_dir_root = '/home/jqruan/data/TSC/add_baselines/sumo-flatten-MAT/onpolicy/scripts/results_sumo/SUMO/SUMO-best-MAT/20240406/'
        eval_model_path = [  
            f'/grid4x4-MAT-mat_thr_2/mat/seed_0_run1/models/', # 0
            '',
            '/nanshan-MAT-mat_thr_2/mat/seed_0_run1/models/', # 2
            f'/arterial4x4-MAT-mat_thr_2/mat/seed_0_run1/models/', # 3
            '',
            f'/cologne8-MAT-mat_thr_2/mat/seed_0_run1/models/', # 5
            f'/grid5x5-MAT-mat_thr_2/mat/seed_0_run1/models/' # 6
        ]
        args.extend(['--model_dir', model_dir_root + eval_model_path[index]])
        args.extend(['--model_id', str(id_list[index])])
        
        
        
        
        main(args)
        
        # cd /home/jqruan/data/TSC/add_baselines/sumo-flatten-MAT/onpolicy/scripts/train; conda activate tsc-hx;
        # index = int(args_[1]) # 训练哪个地图？ # grid44:0   art44: 3    grid55: 6   colo: 5  nanshan: 2
        # CUDA_VISIBLE_DEVICES=0 python eval_sumo.py 0
        # CUDA_VISIBLE_DEVICES=1 python eval_sumo.py 3
        # CUDA_VISIBLE_DEVICES=1 python eval_sumo.py 6
        # CUDA_VISIBLE_DEVICES=2 python eval_sumo.py 5
        # CUDA_VISIBLE_DEVICES=2 python eval_sumo.py 2
