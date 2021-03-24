from os import environ
environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
environ['MKL_THREADING_LAYER']='GNU'

import os
import ray
import copy
import json
import neorl
import torch
import pickle
import random
import hashlib
import argparse
import importlib
import numpy as np
import pandas as pd
import time as timer
from ray import tune

from tabulate import tabulate

import mjrl.utils.tensor_utils as tensor_utils
from mjrl.utils.gym_env import GymEnv, GymEnvCompact
from mjrl.utils.logger import DataLog
from mjrl.algos.mbrl.nn_dynamics import WorldModel
from mjrl.policies.gaussian_mlp import MLP
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.utils.logger import DataLog
from mjrl.utils.make_train_plots import make_train_plots
from mjrl.algos.mbrl.nn_dynamics import WorldModel
from mjrl.algos.mbrl.model_based_npg import ModelBasedNPG
from mjrl.algos.mbrl.sampling import evaluate_policy

class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

SEEDS = [16, 42, 1024]

def setup_seed(seed=1024):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True

def neorl2mjrl(dataset):
    paths = []
    start_index = list(dataset['index'].astype(int))
    end_index = start_index[1:] + [dataset['obs'].shape[0]]

    traj_equal = True
    traj_length = start_index[0] - end_index[0]
    for s, e in zip(start_index, end_index):
        if not (e - s) == traj_length:
            traj_equal = False
            break

    for s, e in zip(start_index, end_index):
        paths.append({
            'observations' : dataset['obs'][s:e],
            'actions' : dataset['action'][s:e],
            'rewards' : dataset['reward'][s:e].reshape((-1)),
            'terminals' : dataset['done'][s:e].reshape((-1)),
            'timeouts' : np.array([False] * (e - s - 1) + [traj_equal]),
        })

    return paths

def get_score(path):
    metrics = pd.read_csv(path)
    performance = metrics['eval_score'].to_numpy()[-101:-1].mean()
    return performance    

def launch(config, seeds=SEEDS):
    job_data = config.pop('job_data')
    job_data.update(config)
    scores = []
    for seed in seeds:
        scores.append(run_neorl(job_data['task'], job_data['level'], job_data['amount'], job_data, seed))
    return {
        "raw_score" : tuple(scores),
        "mean_score" : np.mean(scores),
        "std_score" : np.std(scores),
    }

def run_neorl(task, level, amount, job_data, seed=42):
    setup_seed(seed)
    
    '''Prepare dataset'''
    act_repeat = job_data['act_repeat']

    env = neorl.make(task)
    dataset, _ = env.get_dataset(data_type=level, train_num=amount)
    raw_paths = neorl2mjrl(dataset)
    env = GymEnvCompact(env)
    env.spec = AttributeDict(id=task, max_episode_steps=2516 if task=='finance' else 1000)
    env = GymEnv(env)
    env.set_seed(seed)

    obs_dim = raw_paths[0]['observations'].shape[-1]
    action_dim = raw_paths[0]['actions'].shape[-1]

    # print some statistics
    returns = np.array([np.sum(p['rewards']) for p in raw_paths])
    num_samples = np.sum([p['rewards'].shape[0] for p in raw_paths])
    print("Number of samples collected = %i" % num_samples)
    print("Collected trajectory return mean, std, min, max = %.2f , %.2f , %.2f, %.2f" % \
        (np.mean(returns), np.std(returns), np.min(returns), np.max(returns)) )

    # prepare trajectory dataset (scaling, transforms etc.)
    paths = []
    for p in raw_paths:
        path = dict()
        raw_obs = p['observations']
        raw_act = p['actions']
        raw_rew = p['rewards']
        traj_length = raw_obs.shape[0]
        obs = raw_obs[::act_repeat]
        act = np.array([np.mean(raw_act[i * act_repeat : (i+1) * act_repeat], axis=0) for i in range(traj_length // act_repeat)])
        rew = np.array([np.sum(raw_rew[i * act_repeat : (i+1) * act_repeat]) for i in range(traj_length // act_repeat)])
        path['observations'] = obs
        path['actions'] = act
        path['rewards'] = rew
        paths.append(path)

    '''Train Models'''
    models = [WorldModel(state_dim=obs_dim, act_dim=action_dim, 
              seed=seed+i, **job_data) for i in range(job_data['num_models'])]


    init_states_buffer = [p['observations'][0] for p in paths]
    best_perf = -1e8
    s = np.concatenate([p['observations'][:-1] for p in paths])
    a = np.concatenate([p['actions'][:-1] for p in paths])
    sp = np.concatenate([p['observations'][1:] for p in paths])
    r = np.concatenate([p['rewards'][:-1] for p in paths])
    rollout_score = np.mean([np.sum(p['rewards']) for p in paths])
    num_samples = np.sum([p['rewards'].shape[0] for p in paths])
    for i, model in enumerate(models):
        dynamics_loss = model.fit_dynamics(s, a, sp, **job_data)
        # loss_general = model.compute_loss(s, a, sp) # generalization error
        if job_data['learn_reward']:
            reward_loss = model.fit_reward(s, a, r.reshape(-1, 1), **job_data)

    # pickle.dump(models, open(args.output, 'wb'))


    '''Train Policy'''
    s = hashlib.sha256()
    s.update(str(job_data).encode())
    b = s.hexdigest()
    OUT_DIR = '/home/ubuntu/morel/projects/morel/' + f'{task}-{level}-{amount}/' + b + '/' + str(seed)
    print(OUT_DIR)
    if not os.path.exists(OUT_DIR): os.makedirs(OUT_DIR)
    if not os.path.exists(OUT_DIR+'/iterations'): os.mkdir(OUT_DIR+'/iterations')
    if not os.path.exists(OUT_DIR+'/logs'): os.mkdir(OUT_DIR+'/logs')

    # Unpack args and make files for easy access
    logger = DataLog()
    EXP_FILE = OUT_DIR + '/job_data.json'

    # base cases
    if 'eval_rollouts' not in job_data.keys():  job_data['eval_rollouts'] = 0
    if 'save_freq' not in job_data.keys():      job_data['save_freq'] = 10
    if 'device' not in job_data.keys():         job_data['device'] = 'cpu'
    if 'hvp_frac' not in job_data.keys():       job_data['hvp_frac'] = 1.0
    if 'start_state' not in job_data.keys():    job_data['start_state'] = 'init'
    if 'learn_reward' not in job_data.keys():   job_data['learn_reward'] = True
    if 'num_cpu' not in job_data.keys():        job_data['num_cpu'] = 1
    if 'npg_hp' not in job_data.keys():         job_data['npg_hp'] = dict()
    if 'act_repeat' not in job_data.keys():     job_data['act_repeat'] = 1

    assert job_data['start_state'] in ['init', 'buffer']
    with open(EXP_FILE, 'w') as f:  json.dump(job_data, f, indent=4)
    job_data['base_seed'] = seed

    # ===============================================================================
    # Helper functions
    # ===============================================================================
    def buffer_size(paths_list):
        return np.sum([p['observations'].shape[0]-1 for p in paths_list])

    # ===============================================================================
    # Setup functions and environment
    # ===============================================================================

    # check for reward and termination functions
    if 'reward_file' in job_data.keys():
        splits = job_data['reward_file'].split("/")
        dirpath = "" if splits[0] == "" else os.path.dirname(os.path.abspath(__file__))
        for x in splits[:-1]: dirpath = dirpath + "/" + x
        filename = splits[-1].split(".")[0]
        module = importlib.import_module(f'func.{filename}')
        reward_function = module.__getattribute__('reward_function')
        termination_function = module.__getattribute__('termination_function')

    # ===============================================================================
    # Setup policy, model, and agent
    # ===============================================================================

    # Construct policy and set exploration level correctly for NPG
    if 'init_policy' in job_data.keys():
        policy = pickle.load(open(job_data['init_policy'], 'rb'))
        policy.set_param_values(policy.get_param_values())
        init_log_std = job_data['init_log_std']
        min_log_std = job_data['min_log_std']
        if init_log_std:
            params = policy.get_param_values()
            params[:policy.action_dim] = tensor_utils.tensorize(init_log_std)
            policy.set_param_values(params)
        if min_log_std:
            policy.min_log_std[:] = tensor_utils.tensorize(min_log_std)
            policy.set_param_values(policy.get_param_values())
    else:
        policy = MLP(observation_dim=obs_dim, action_dim=action_dim, seed=seed, hidden_sizes=job_data['policy_size'], 
                     init_log_std=job_data['init_log_std'], min_log_std=job_data['min_log_std'])

    baseline = MLPBaseline(env.spec, inp_dim=obs_dim, reg_coef=1e-3, batch_size=256, epochs=1,  learn_rate=1e-3, device=job_data['device'])               
    agent = ModelBasedNPG(learned_model=models, env=env, policy=policy, baseline=baseline, seed=seed,
                          normalized_step_size=job_data['step_size'], save_logs=True, 
                          reward_function=reward_function, termination_function=termination_function,
                          **job_data['npg_hp'])

    # ===============================================================================
    # Model training loop
    # ===============================================================================

    init_states_buffer = [p['observations'][0] for p in paths]
    best_perf = -1e8
    ts = timer.time()
    s = np.concatenate([p['observations'][:-1] for p in paths])
    a = np.concatenate([p['actions'][:-1] for p in paths])
    sp = np.concatenate([p['observations'][1:] for p in paths])
    r = np.concatenate([p['rewards'][:-1] for p in paths])
    rollout_score = np.mean([np.sum(p['rewards']) for p in paths])
    num_samples = np.sum([p['rewards'].shape[0] for p in paths])
    logger.log_kv('fit_epochs', job_data['fit_epochs'])
    logger.log_kv('rollout_score', rollout_score)
    logger.log_kv('iter_samples', num_samples)
    logger.log_kv('num_samples', num_samples)
    try:
        rollout_metric = env.evaluate_success(paths)
        logger.log_kv('rollout_metric', rollout_metric)
    except:
        pass
    
    with torch.no_grad():
        for i, model in enumerate(models):
            loss_general = model.compute_loss(s, a, sp)
            logger.log_kv('dyn_loss_gen_' + str(i), loss_general)
            torch.cuda.empty_cache()

    tf = timer.time()
    logger.log_kv('model_learning_time', tf-ts)
    print("Model learning statistics")
    print_data = sorted(filter(lambda v: np.asarray(v[1]).size == 1,
                                logger.get_current_log().items()))
    print(tabulate(print_data))
    pickle.dump(models, open(OUT_DIR + '/models.pickle', 'wb'))
    logger.log_kv('act_repeat', job_data['act_repeat']) # log action repeat for completeness

    # ===============================================================================
    # Pessimistic MDP parameters
    # ===============================================================================

    with torch.no_grad():
        delta = np.zeros(s.shape[0])
        for idx_1, model_1 in enumerate(models):
            pred_1 = model_1.predict(s, a)
            for idx_2, model_2 in enumerate(models):
                if idx_2 > idx_1:
                    pred_2 = model_2.predict(s, a)
                    disagreement = np.linalg.norm((pred_1-pred_2), axis=-1)
                    delta = np.maximum(delta, disagreement)
    torch.cuda.empty_cache()

    if 'pessimism_coef' in job_data.keys():
        if job_data['pessimism_coef'] is None or job_data['pessimism_coef'] == 0.0:
            truncate_lim = None
            print("No pessimism used. Running naive MBRL.")
        else:
            truncate_lim = (1.0 / job_data['pessimism_coef']) * np.max(delta)
            print("Maximum error before truncation (i.e. unknown region threshold) = %f" % truncate_lim)
        job_data['truncate_lim'] = truncate_lim
        job_data['truncate_reward'] = job_data['truncate_reward'] if 'truncate_reward' in job_data.keys() else 0.0
    else:
        job_data['truncate_lim'] = None
        job_data['truncate_reward'] = 0.0

    with open(EXP_FILE, 'w') as f:
        job_data['seed'] = seed
        json.dump(job_data, f, indent=4)
        del(job_data['seed'])

    # ===============================================================================
    # Behavior Cloning Initialization
    # ===============================================================================
    if 'bc_init' in job_data.keys():
        if job_data['bc_init']:
            from mjrl.algos.behavior_cloning import BC
            policy.to(job_data['device'])
            bc_agent = BC(paths, policy, epochs=5, batch_size=256, loss_type='MSE')
            bc_agent.train()

    # ===============================================================================
    # Policy Optimization Loop
    # ===============================================================================

    for outer_iter in range(job_data['num_iter']):
        ts = timer.time()
        agent.to(job_data['device'])
        if job_data['start_state'] == 'init':
            print('sampling from initial state distribution')
            buffer_rand_idx = np.random.choice(len(init_states_buffer), size=job_data['update_paths'], replace=True).tolist()
            init_states = [init_states_buffer[idx] for idx in buffer_rand_idx]
        else:
            # Mix data between initial states and randomly sampled data from buffer
            print("sampling from mix of initial states and data buffer")
            if 'buffer_frac' in job_data.keys():
                num_states_1 = int(job_data['update_paths']*(1-job_data['buffer_frac'])) + 1
                num_states_2 = int(job_data['update_paths']* job_data['buffer_frac']) + 1
            else:
                num_states_1, num_states_2 = job_data['update_paths'] // 2, job_data['update_paths'] // 2
            buffer_rand_idx = np.random.choice(len(init_states_buffer), size=num_states_1, replace=True).tolist()
            init_states_1 = [init_states_buffer[idx] for idx in buffer_rand_idx]
            buffer_rand_idx = np.random.choice(s.shape[0], size=num_states_2, replace=True)
            init_states_2 = list(s[buffer_rand_idx])
            init_states = init_states_1 + init_states_2

        train_stats = agent.train_step(N=len(init_states), init_states=init_states, **job_data)
        logger.log_kv('train_score', train_stats[0])
        agent.policy.to('cpu')
        
        # evaluate true policy performance
        if job_data['eval_rollouts'] > 0:
            print("Performing validation rollouts ... ")
            # set the policy device back to CPU for env sampling
            eval_paths = evaluate_policy(agent.env, agent.policy, agent.learned_model[0], noise_level=0.0,
                                         real_step=True, num_episodes=job_data['eval_rollouts'], visualize=False)
            eval_score = np.mean([np.sum(p['rewards']) for p in eval_paths])
            logger.log_kv('eval_score', eval_score)
            try:
                eval_metric = env.evaluate_success(eval_paths)
                logger.log_kv('eval_metric', eval_metric)
            except:
                pass
        else:
            eval_score = -1e8

        # track best performing policy
        policy_score = eval_score if job_data['eval_rollouts'] > 0 else rollout_score
        if policy_score > best_perf:
            best_policy = copy.deepcopy(policy) # safe as policy network is clamped to CPU
            best_perf = policy_score

        tf = timer.time()
        logger.log_kv('iter_time', tf-ts)
        for key in agent.logger.log.keys():
            logger.log_kv(key, agent.logger.log[key][-1])
        print_data = sorted(filter(lambda v: np.asarray(v[1]).size == 1,
                                logger.get_current_log_print().items()))
        print(tabulate(print_data))
        logger.save_log(OUT_DIR+'/logs')

        if outer_iter > 0 and outer_iter % job_data['save_freq'] == 0:
            # convert to CPU before pickling
            agent.to('cpu')
            # make observation mask part of policy for easy deployment in environment
            old_in_scale = policy.in_scale
            for pi in [policy, best_policy]: pi.set_transformations(in_scale=1.0)
            # pickle.dump(agent, open(OUT_DIR + '/iterations/agent_' + str(outer_iter) + '.pickle', 'wb'))
            pickle.dump(policy, open(OUT_DIR + '/iterations/policy_' + str(outer_iter) + '.pickle', 'wb'))
            pickle.dump(best_policy, open(OUT_DIR + '/iterations/best_policy.pickle', 'wb'))
            agent.to(job_data['device'])
            for pi in [policy, best_policy]: pi.set_transformations(in_scale = old_in_scale)
            make_train_plots(log=logger.log, keys=['rollout_score', 'eval_score', 'rollout_metric', 'eval_metric'],
                             x_scale=float(job_data['act_repeat']), y_scale=1.0, save_loc=OUT_DIR+'/logs/')

    # final save
    # pickle.dump(agent, open(OUT_DIR + '/iterations/agent_final.pickle', 'wb'))
    policy.set_transformations(in_scale=1.0)
    pickle.dump(policy, open(OUT_DIR + '/iterations/policy_final.pickle', 'wb'))

    return get_score(os.path.join(OUT_DIR, 'logs', 'log.csv'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='HalfCheetah-v3')
    parser.add_argument('--level', type=str, default='medium')
    parser.add_argument('--amount', type=int, default=999)
    parser.add_argument('--mode', type=str, default='tune', choices=['default', 'tune'])
    parser.add_argument('--trial_per_gpu', type=int, default=4)
    args = parser.parse_args()

    task = args.task
    level = args.level
    amount = args.amount

    if not os.path.exists('neorl-results'): os.makedirs('neorl-results')

    env = neorl.make(task)
    env.get_dataset(data_type=level, train_num=amount)

    ray.init()

    with open(os.path.join('configs', f'{task}_{level}.txt'), 'r') as f:
        job_data = eval(f.read())

    job_data['task'] = task
    job_data['level'] = level
    job_data['amount'] = amount

    if args.mode == 'tune':    
        analysis = tune.run(
            launch,
            config={
                "truncate_reward" : tune.grid_search([0.0, -50.0, -100.0, -200.0]),
                "fit_epochs" : tune.grid_search([25, 50, 100, 200]),
                "pessimism_coef" : tune.grid_search([0.2, 1.0, 2.0, 5.0]),
                "job_data" : job_data,
            },
            resources_per_trial={"gpu": 1 / args.trial_per_gpu},
            queue_trials = True,
            metric='mean_score',
            mode='max',
        )

        print("Best config: ", analysis.get_best_config(metric="mean_score", mode="max"))
        print(analysis.best_result)

        with open(f'neorl-results/{task}-{level}-{amount}-tune.pkl', 'wb') as f:
            pickle.dump(analysis, f)

    else:
        run_func = ray.remote(max_calls=1)(run_neorl)
        run_func = run_func.options(num_gpus=1.0)
        scores = ray.get([run_func.remote(task, level, amount, job_data, seed) for seed in SEEDS])
        
        print(np.mean(scores), np.std(scores))

        with open(f'neorl-results/{task}-{level}-{amount}-default.txt', 'w') as f:
            for score in scores:
                f.write(f'{score}\n')