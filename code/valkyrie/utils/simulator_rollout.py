import numpy as np
import time


def rollout(
        env,
        agent,
        max_path_length=np.inf,
        animated=False,
        speedup=1,
        always_return_paths=False):
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    obs = env.reset()
    # agent.reset()
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:
        act, agent_info = agent.get_action(obs)
        next_obs, reward, done, env_info = env.step(act)
        obs_flat = np.asarray(obs).flatten()
        observations.append(obs_flat)
        rewards.append(reward)
        act_onehot = np.zeros(env.action_space.n)
        act_onehot[act] = 1
        actions.append(act_onehot)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if done:
            break
        obs = next_obs
        if animated:
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)
    if animated and not always_return_paths:
        return

    return dict(
        observations=np.array(observations),
        actions=np.array(actions),
        rewards=np.array(rewards),
        agent_infos=stack_tensor_dict_list(agent_infos),
        env_infos=stack_tensor_dict_list(env_infos),
    )


def stack_tensor_dict_list(tensor_dict_list):
    """ Stack a list of dictionaries of {tensors or dictionary of tensors}.

    Parameters
    ----------
    tensor_dict_list : list
        list of dicts of tensors or list of dict of dict of tensors

    Returns
    -------
    dict containing numpy.ndarray of tensors or dicts of ndarray of tensor
    """
    keys = list(tensor_dict_list[0].keys())
    ret = dict()
    for k in keys:
        example = tensor_dict_list[0][k]
        if isinstance(example, dict):
            v = stack_tensor_dict_list([x[k] for x in tensor_dict_list])
        else:
            v = np.array([x[k] for x in tensor_dict_list])
        ret[k] = v
    return ret
