import numpy as np

def reward_function(paths):
    # path has two keys: observations and actions
    # path["observations"] : (num_traj, horizon, obs_dim)
    # return paths that contain rewards in path["rewards"]
    # path["rewards"] should have shape (num_traj, horizon)

    obs = paths["observations"][:, :-1]
    next_obs = paths["observations"][:, 1:]
    action = paths["actions"][:, :-1]

    x_position_before = obs[..., 0]
    x_position_after = next_obs[..., 0]
    dt = 0.008
    _forward_reward_weight = 1.0
    x_velocity = (x_position_after - x_position_before) / dt

    forward_reward = _forward_reward_weight * x_velocity
    healthy_reward = 1.0

    rewards = forward_reward + healthy_reward
    costs = (action ** 2).sum(axis=-1)

    rewards = rewards - 1e-3 * costs

    paths["observations"] = paths["observations"][:, :-1]
    paths["actions"] = paths["actions"][:, :-1]
    paths["rewards"] = rewards if rewards.shape[0] > 1 else rewards.ravel()
    return paths

def termination_function(paths):
    # paths is a list of path objects for this function
    min_state, max_state = (-100.0, 100.0)
    min_z, max_z = (0.7, float('inf'))
    min_angle, max_angle = (-0.2, 0.2)

    for path in paths:
        obs = path["observations"]
        z = obs[..., 1]
        angle = obs[..., 2]
        state = obs[..., 2:]
        T = obs.shape[0]
        t = 0
        done = False
        while t < T and done is False:
            healthy_state = np.all(np.logical_and(min_state < state[t], state[t] < max_state))
            healthy_angle = min_angle < angle[t] < max_angle
            healthy_z = min_z < z[t] < max_z
            is_healthy = all((healthy_state, healthy_z, healthy_angle))
            done = not is_healthy
            t += 1
            T = t if done else T
        path["observations"] = path["observations"][:T]
        path["actions"] = path["actions"][:T]
        path["rewards"] = path["rewards"][:T]
        path["terminated"] = done
    return paths