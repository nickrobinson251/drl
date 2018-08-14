"""Example script for training from an existing Q-function and Policy"""
import argparse
import joblib
import numpy as np
import tensorflow as tf

from softqlearning.algorithms import SQL
from softqlearning.misc.kernel import adaptive_isotropic_gaussian_kernel
from softqlearning.value_functions import NNQFunction
from softqlearning.misc.plotter import QFPolicyPlotter
from softqlearning.policies import StochasticNNPolicy
from softqlearning.misc.sampler import SkipSampler
from softqlearning.misc.logger import set_snapshot_dir

from valkyrie.utils import ReplayBuffer
from valkyrie.envs import ValkyrieBalanceEnv, BalancebotEnv, TwoRoomGridEnv

set_snapshot_dir(".")

parser = argparse.ArgumentParser(description="Choose env to train.")
parser.add_argument('file', type=str, help='Path to the snapshot file.')
parser.add_argument("--env", default="valkyrie",
                    help="valkyrie, balancebot or gridworld.")
parser.add_argument("--epochs", default=10,
                    help="number of training epochs.")
parser.add_argument("--plot", action="store_true",
                    help="Enable plotting of Q function during training.")
parser.add_argument("--render", action="store_true",
                    help="Enable video during training.")
args = parser.parse_args()


def main():
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
            render=args.render,
            logfilename="."
            )
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

    with tf.Session().as_default():
        data = joblib.load(args.file)
        if 'algo' in data.keys():
            saved_qf = data['algo'].qf
            saved_policy = data['algo'].policy
        else:
            saved_qf = data['qf']
            saved_policy = data['policy']

        algorithm = SQL(
            discount=0.99,
            env=env,
            epoch_length=1000,
            eval_n_episodes=1,
            eval_render=False,
            kernel_fn=adaptive_isotropic_gaussian_kernel,
            kernel_n_particles=16,
            kernel_update_ratio=0.5,
            n_epochs=args.epochs,
            n_train_repeat=1,
            plotter=plotter,
            policy=saved_policy,
            policy_lr=3E-4,
            replay_buffer=replay_buffer,
            qf=saved_qf,
            qf_lr=3E-4,
            reward_scale=30,
            sampler=sampler,
            save_full_state=True,
            td_target_update_interval=1000,
            use_saved_policy=True,
            use_saved_qf=True,
            value_n_particles=16)

        env.reset()
        env.start_logging_video()
        algorithm.train()


if __name__ == "__main__":
    main()
