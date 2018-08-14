import argparse
import joblib
import tensorflow as tf

from valkyrie.utils.simulator_rollout import rollout


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='Path to the snapshot file.')
    parser.add_argument('--length', '-l', type=int, default=100,
                        help="Length of episode to simulate.")
    parser.add_argument('--speedup', '-s', type=float, default=1)
    parser.set_defaults(deterministic=True)
    args = parser.parse_args()
    return args


def simulate_policy(args):
    with tf.Session() as sess:
        data = joblib.load(args.file)
        if 'algo' in data.keys():
            policy = data['algo'].policy
            env = data['algo'].env
        else:
            policy = data['policy']
            env = data['env']

        # sess.run(tf.global_variables_initializer())
        print(sess.run(tf.report_uninitialized_variables()))
        while True:
            rollout(
                env=env,
                agent=policy,
                max_path_length=args.length,
                animated=True,
                speedup=args.speedup)


def main():
    args = cli()
    simulate_policy(args)


if __name__ == "__main__":
    main()
