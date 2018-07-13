from pprint import pprint
from valkyrie.envs.valkyrie_env import ValkyrieEnvBasic
import numpy as np


def main():
    env = ValkyrieEnvBasic(render=False)

    # check we can get observations
    env.step_without_torque()
    obs1 = env.get_observation()
    obs1_filtered = env.get_filtered_observation()

    env.step_without_torque()
    obs2 = env.get_observation()
    obs2_filtered = env.get_filtered_observation()

    pprint(obs1)
    pprint(obs2)
    pprint(obs1_filtered)
    pprint(obs2_filtered)

    # check we haven't changed the observations we get
    obs2_true = np.load(file='obs2.npy')
    obs2_filtered_true = np.load(file='obs2_filtered.npy')
    try:
        assert(np.allclose(obs2, obs2_true))
    except:
        print(obs2 - obs2_true)
    assert(np.allclose(obs2_filtered, obs2_filtered_true))

    print("[PASSED]")


if __name__ == '__main__':
    main()
