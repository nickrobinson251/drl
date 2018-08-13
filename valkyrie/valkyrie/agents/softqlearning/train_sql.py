"""Example script to perform soft Q-learning in a gym environment."""
import argparse
import numpy as np

from softqlearning.algorithms import SQL
from softqlearning.misc.kernel import adaptive_isotropic_gaussian_kernel
from softqlearning.value_functions import NNQFunction
from softqlearning.misc.plotter import QFPolicyPlotter
from softqlearning.policies import StochasticNNPolicy
from softqlearning.misc.sampler import SkipSampler

from valkyrie.utils import ReplayBuffer
from valkyrie.envs import ValkyrieBalanceEnv, BalancebotEnv, TwoRoomGridEnv


parser = argparse.ArgumentParser(description="Choose env to train.")
parser.add_argument("--env", default="valkyrie",
                    help="valkyrie, balancebot or gridworld.")
parser.add_argument("--plot", action="store_true",
                    help="Enable plotting of Q function during training.")
args = parser.parse_args()


def test():
    if args.env == "valkyrie":
        # env_lateral = ValkyrieBalanceEnv(
        #     controlled_joints=[
        #         "leftAnkleRoll",
        #         "leftAnkleRoll",
        #         "leftHipRoll",
        #         "leftKneePitch",
        #         "rightAnkleRoll",
        #         "rightHipRoll",
        #         "rightKneePitch",
        #         "torsoRoll",
        #     ],
        #     render=False)
        # sagittal
        env = ValkyrieBalanceEnv(
            balance_task='3d',
            render=False)
    elif args.env == "balancebot":
        env = BalancebotEnv()
    elif args.env == "gridworld":
        env = TwoRoomGridEnv()

    replay_buffer = ReplayBuffer(size=1e6)

    sampler = SkipSampler(
        max_episode_length=50,
        min_replay_buffer_size=500,
        batch_size=64)

    M = 128
    policy = StochasticNNPolicy(
        env=env,
        hidden_layer_sizes=(M, M),
        squash=True)

    q_function = NNQFunction(
        env=env,
        hidden_layer_sizes=(M, M))

    if args.plot:
        plotter = QFPolicyPlotter(
            q_function=q_function,
            policy=policy,
            observations=np.ones((4, 51)),
            default_action=[np.nan]*env.action_space.n,
            n_samples=100)
    else:
        plotter = None

    algorithm = SQL(
        sampler=sampler,
        env=env,
        replay_buffer=replay_buffer,
        q_function=q_function,
        policy=policy,
        plotter=plotter,
        discount=1.0,
        epoch_length=1,
        eval_n_episodes=1,
        eval_render=True,
        kernel_fn=adaptive_isotropic_gaussian_kernel,
        kernel_n_particles=32,
        kernel_update_ratio=0.5,
        n_epochs=1,
        n_train_repeat=1,
        policy_lr=3e-4,
        q_function_lr=3e-4,
        reward_scale=0.1,
        save_full_state=True,
        td_target_update_interval=1000,
        value_n_particles=16)

    env.reset()
    env.start_logging_video()
    algorithm.train()


if __name__ == "__main__":
    test()
