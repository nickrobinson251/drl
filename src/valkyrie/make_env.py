import gym
import time
from valkyrie.envs.valkyrie_env import ValkyrieEnv

def main():
    # env = gym.make('Valkyrie-v0')
    env = ValkyrieEnv()
    time.sleep(5)  # seconds


if __name__ == '__main__':
    main()
