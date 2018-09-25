import argparse
from configobj import ConfigObj

from softqlearning.algorithms import SQL
from softqlearning.misc.kernel import adaptive_isotropic_gaussian_kernel
from softqlearning.replay_buffers import SimpleReplayBuffer
from softqlearning.value_functions import NNQFunction
from softqlearning.policies import StochasticNNPolicy
from softqlearning.misc.sampler import SimpleSampler

SHARED_PARAMS = {
    'seed': 1,
    'policy_lr': 3E-4,
    'qf_lr': 3E-4,
    'discount': 0.99,
    'layer_size': 128,
    'batch_size': 128,
    'max_pool_size': 1E6,
    'n_train_repeat': 1,
    'epoch_length': 1000,
    'kernel_particles': 16,
    'kernel_update_ratio': 0.5,
    'value_n_particles': 16,
    'td_target_update_interval': 1000,
    'snapshot_mode': 'last',
    'snapshot_gap': 100,
}


ENV_PARAMS = {
    'swimmer': {  # 2 DoF
        'prefix': 'swimmer',
        'env_name': 'swimmer-rllab',
        'max_episode_length': 1000,
        'n_epochs': 500,
        'reward_scale': 30}}
DEFAULT_ENV = 'swimmer'
AVAILABLE_ENVS = list(ENV_PARAMS.keys())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--env', type=str, choices=AVAILABLE_ENVS, default=DEFAULT_ENV)
    parser.add_argument('--config_file', type=str)
    args = parser.parse_args()
    return args


def run_experiment(env, config):
    pool = SimpleReplayBuffer(
        env_spec=env.spec,
        max_replay_buffer_size=config['max_pool_size'])

    sampler = SimpleSampler(
        max_episode_length=config['max_episode_length'],
        min_pool_size=config['max_episode_length'],
        batch_size=config['batch_size'])

    base_kwargs = dict(
        epoch_length=config['epoch_length'],
        n_epochs=config['n_epochs'],
        n_train_repeat=config['n_train_repeat'],
        eval_render=False,
        eval_n_episodes=1,
        sampler=sampler)

    layer_size = config['layer_size']
    q_function = NNQFunction(
        env_spec=env.spec,
        hidden_layer_sizes=(layer_size, layer_size))

    policy = StochasticNNPolicy(
        env_spec=env.spec,
        hidden_layer_sizes=(layer_size, layer_size))

    algorithm = SQL(
        base_kwargs=base_kwargs,
        env=env,
        pool=pool,
        q_function=q_function,
        policy=policy,
        discount=config['discount'],
        kernel_fn=adaptive_isotropic_gaussian_kernel,
        kernel_n_particles=config['kernel_particles'],
        kernel_update_ratio=config['kernel_update_ratio'],
        policy_lr=config['policy_lr'],
        qf_lr=config['qf_lr'],
        reward_scale=config['reward_scale'],
        save_full_state=False,
        td_target_update_interval=config['td_target_update_interval'],
        value_n_particles=config['value_n_particles'])
    algorithm.train()


def main():
    args = parse_args()
    config = ConfigObj(args.config)
    run_experiment(args.env, config)


if __name__ == '__main__':
    main()
