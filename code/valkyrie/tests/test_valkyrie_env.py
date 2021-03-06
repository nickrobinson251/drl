import gym
import time
from numpy import zeros
from pprint import pprint
from valkyrie.envs.valkyrie_env import ValkyrieBalanceEnv


def test_imported_env():
    env = ValkyrieBalanceEnv(render=True)
    time.sleep(10)

    env._setup_camera(yaw=90, target_position=[0.5, 0, 0.9])
    env.step(zeros(env.action_space.n))
    time.sleep(10)

    env._setup_camera(yaw=0, target_position=[0, 0, 0.9])
    env.step(zeros(env.action_space.n))
    time.sleep(10)

    # pprint(env.get_reading())
    print("[PASSED] {} works!".format(env.__class__))


def test_gym_make_env():
    env_name = 'Valkyrie-v0'
    env = gym.make(env_name)
    time.sleep(3)

    env._setup_camera(yaw=90, target_position=[0.5, 0, 0.9])
    env.step(zeros(env.action_space.n))
    time.sleep(3)

    env._setup_camera(yaw=0, target_position=[0, 0, 0.9])
    env.step(zeros(env.action_space.n))
    time.sleep(3)
    env.close()

    # pprint(env.get_reading())
    print("[PASSED] gym.make('{}') works!".format(env_name))


def main():
    try:
        test_gym_make_env()
    except Exception as e:
        print("[FAILED] on gym.make : ", e)
    try:
        test_imported_env()
    except Exception as e:
        print("[FAILED] on imported env : ", e)


if __name__ == '__main__':
    main()
