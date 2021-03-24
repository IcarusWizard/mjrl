import numpy as np

def reward_function(paths):
    # path has two keys: observations and actions
    # path["observations"] : (num_traj, horizon, obs_dim)
    # return paths that contain rewards in path["rewards"]
    # path["rewards"] should have shape (num_traj, horizon)

    obs = paths["observations"][:, :-1]
    next_obs = paths["observations"][:, 1:]
    action = paths["actions"][:, :-1]

    index = [25, 32, 38, 45, 52, 59, 66, 73]
    electricity_demand = - next_obs[..., index]
    
    reward_ = - np.sum(electricity_demand, axis=1)
    reward_ = np.clip(reward_, 0, np.max(reward_))

    rewards = reward_ ** 3.0 * 0.00001

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