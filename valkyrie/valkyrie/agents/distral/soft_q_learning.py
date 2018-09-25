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
