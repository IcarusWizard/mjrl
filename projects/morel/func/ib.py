import numpy as np

def reward_function(paths):
    # path has two keys: observations and actions
    # path["observations"] : (num_traj, horizon, obs_dim)
    # return paths that contain rewards in path["rewards"]
    # path["rewards"] should have shape (num_traj, horizon)

    obs = paths["observations"][:, :-1]
    next_obs = paths["observations"][:, 1:]
    action = paths["actions"][:, :-1]


    CRF = 3.0
    CRC = 1.0

    fatigue = next_obs[..., -2]
    consumption = next_obs[:, -1]

    cost = CRF * fatigue + CRC * consumption

    rewards = - cost

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