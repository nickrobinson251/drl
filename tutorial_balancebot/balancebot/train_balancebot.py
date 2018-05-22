import gym
from baselines import deepq

import balancebot

def stop_condition(_locals, _globals):
    """Stop if reward in last 100 eposides exceeds 199."""
    return (_locals['t'] > 100 and
            sum(_locals['episode_rewards'][-101:-1]) / 100 >= 199)


def main():
    env = gym.make("balancebot-v0")
    mlp = deepq.models.mlp([16, 16])
    train = deepq.learn(env, q_func=mlp, lr=1e-3, max_timesteps=int(1e5),
                        buffer_size=15000, exploration_fraction=0.1,
                        exploration_final_eps=0.02, print_freq=10,
                        callback=stop_condition)
    train.save("balance.pkl")


if __name__ == '__main__':
    main()
