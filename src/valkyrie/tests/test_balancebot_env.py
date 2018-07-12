import time
from valkyrie.envs.balancebot_env import BalancebotEnv


def test_balancebot_env():
    env = BalancebotEnv()
    for _ in range(5):
        env.step(env.action_space.sample())
        time.sleep(1)
    print("[PASSED] {} works!".format(env.__class__))


def main():
    try:
        test_balancebot_env()
    except Exception as e:
        print("[FAILED] : ", e)


if __name__ == '__main__':
    main()
