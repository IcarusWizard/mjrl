import numpy as np

def reward_function(paths):
    # path has two keys: observations and actions
    # path["observations"] : (num_traj, horizon, obs_dim)
    # return paths that contain rewards in path["rewards"]
    # path["rewards"] should have shape (num_traj, horizon)

    obs = paths["observations"][:, :-1]
    next_obs = paths["observations"][:, 1:]
    action = paths["actions"][:, :-1]

    timestep = 0.002
    frame_skip = 4
    dt = timestep * frame_skip
    x_velocity = (next_obs[..., 0] - obs[..., 0]) / dt
    forward_reward = x_velocity

    healthy_reward = 1

    rewards = forward_reward + healthy_reward
    costs = 1e-3 * (action ** 2).sum(axis=-1)
    rewards = rewards - costs

    paths["observations"] = paths["observations"][:, :-1]
    paths["actions"] = paths["actions"][:, :-1]
    paths["rewards"] = rewards if rewards.shape[0] > 1 else rewards.ravel()
    return paths

def termination_function(paths):
    # paths is a list of path objects for this function
    min_z, max_z = (0.8, 2.0)
    min_angle, max_angle = (-1.0, 1.0)

    for path in paths:
        obs = path["observations"]
        z = obs[..., 1]
        angle = obs[..., 2]
        T = obs.shape[0]
        t = 0
        done = False
        while t < T and done is False:
            healthy_angle = min_angle < angle[t] < max_angle
            healthy_z = min_z < z[t] < max_z
            is_healthy = healthy_z and healthy_angle
            done = not is_healthy
            t += 1
            T = t if done else T
        path["observations"] = path["observations"][:T]
        path["actions"] = path["actions"][:T]
        path["rewards"] = path["rewards"][:T]
        path["terminated"] = done
    return paths