import time
from numpy import zeros
from valkyrie.envs.pendulum_env import InvertedPendulumEnv


def test_pendulum_env():
    env = InvertedPendulumEnv()
    for _ in range(5):
        env.step(env.action_space.sample())
        time.sleep(1)
    print("[PASSED] {} works!".format(env.__class__))


def main():
    try:
        test_pendulum_env()
    except Exception as e:
        print("[FAILED] : ", e)


if __name__ == '__main__':
    main()
