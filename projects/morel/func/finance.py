import numpy as np

def reward_function(paths):
    # path has two keys: observations and actions
    # path["observations"] : (num_traj, horizon, obs_dim)
    # return paths that contain rewards in path["rewards"]
    # path["rewards"] should have shape (num_traj, horizon)

    obs = paths["observations"][:, :-1]
    next_obs = paths["observations"][:, 1:]
    action = paths["actions"][:, :-1]

    reward_scaling = 1e-4
    stock_dim = 30

    begin_total_asset = obs[..., 0] + np.sum(obs[..., 1:(stock_dim + 1)] * obs[..., (stock_dim + 1):(stock_dim * 2 + 1)], axis=-1)
    end_total_asset = next_obs[..., 0] + np.sum(next_obs[..., 1:(stock_dim + 1)] * next_obs[..., (stock_dim + 1):(stock_dim * 2 + 1)], axis=-1)
    rewards = end_total_asset - begin_total_asset
    rewards = reward_scaling * rewards

    paths["observations"] = paths["observations"][:, :-1]
    paths["actions"] = paths["actions"][:, :-1]
    paths["rewards"] = rewards if rewards.shape[0] > 1 else rewards.ravel()
    return paths

def termination_function(paths):
    # paths is a list of path objects for this function
    for path in paths:
        obs = path["observations"]
        T = obs.shape[0]
        done = False
        path["observations"] = path["observations"][:T]
        path["actions"] = path["actions"][:T]
        path["rewards"] = path["rewards"][:T]
        path["terminated"] = done
    return paths