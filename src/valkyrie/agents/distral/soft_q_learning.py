import numpy as np


# distral hyperparams
alphas = [0.5, 1]
betas = [3e3, 1e3, 3e4]
learning_rate_epslions = [2e-4, 4e-4, 8e-4]
env_steps_per_task = 4e8
action_repeat = 4  # num times each action output by network is fed into env
train_steps_per_env = env_steps_per_task / action_repeat

# for two room grid env
beta = 5
learning_rate = 0.1
discount_gamma = 0.95  # for soft_q_learning


def log_sum_exp(z, factors=1):
    """Compute log sum_i factor_i exp(z_i)."""
    z_max = np.max(z)  # for numerical stability
    return np.log(np.sum(factors*np.exp(z - z_max)))


def distral_objective(
        policy_0,
        task_policies,
        envs,
        time_steps,
        alpha=0.5,
        beta=1e3,
        gamma=1):
    """

    See equation (1) in Distral paper.
    """
    reward_term = []
    shaping_term = []
    entropy_term = []
    for i, policy_i in enumerate(task_policies):
        state = envs[i].observation
        for t in range(time_steps):
            action = policy_i(state)
            state, reward, done, _ = envs[i].step(action)
            gamma *= gamma
            reward_term.append(gamma * reward)
            shaping_term.append(
                gamma * alpha/beta * np.log(policy_0(action, state)))
            entropy_term.append(gamma / beta * np.log(policy_i(action, state)))
            state, reward, done, _ = envs[i].step(action)
    objectiv = np.sum(reward_term) + np.sum(shaping_term) - np.sum(entropy_term)
    return objectiv


def soft_V(state, policy, env, beta=1e3):
    """

    See equation (2) in Distral paper.
    """
    q_values = np.array([soft_Q(action, state) for action in env.action_space])
    return log_sum_exp(beta * q_values, factors=policy(state)) / beta


def soft_Q(action, state, env, gamma=1):
    """

    See equation (3) in Distral paper.
    """
    next_states, p_next_states = env.transition(action, state)
    expected_next_state_value = gamma * np.sum(
        p_next_states * soft_V(next_states))
    return env.reward(action, state) + expected_next_state_value


def soft_advantage_function(action, state, policy, env, beta=1e3):
    """

    See equation (7) in Distral paper.
    """
    Q = soft_Q(action, state, env)
    V = soft_V(state, policy, env, beta=beta)
    return Q - V


def reward_regularised(action, state, env, policy_0, policy_i, alpha, beta):
    """Reward function regualised by shapng and entropy terms.

    See equation (9) in Distral paper.
    """
    reward = env.reward(action, state)
    reg_shaping = alpha / beta * np.log(policy_0(action, state))
    reg_entropy = 1.0 / beta * np.log(policy_i(action, state))
    return reward + reg_shaping - reg_entropy
