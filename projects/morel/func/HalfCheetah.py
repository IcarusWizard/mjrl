import numpy as np

def reward_function(paths):
    # path has two keys: observations and actions
    # path["observations"] : (num_traj, horizon, obs_dim)
    # return paths that contain rewards in path["rewards"]
    # path["rewards"] should have shape (num_traj, horizon)

    obs = paths["observations"][:, :-1]
    next_obs = paths["observations"][:, 1:]
    action = paths["actions"][:, :-1]

    forward_reward_weight = 1.0 
    ctrl_cost_weight = 0.1
    dt = 0.05
    
    ctrl_cost = ctrl_cost_weight * np.sum(np.square(action), axis=-1)
    
    x_position_before = obs[..., 0]
    x_position_after = next_obs[..., 0]
    x_velocity = ((x_position_after - x_position_before) / dt)

    forward_reward = forward_reward_weight * x_velocity
    
    rewards = forward_reward - ctrl_cost

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