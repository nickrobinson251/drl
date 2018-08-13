from datetime import datetime
import numpy as np
import os
from pprint import pprint

from configuration import config, save_configuration, record_configuration
from ddpg import DDPG
from interpolate import JointTrajectoryInterpolate
from logger import logger
from valkyrie.envs.valkyrie_env import ValkyrieEnv

START_TIME = datetime.now().strftime('%Y_%m_%d_%H.%M.%S')
DIR_PATH = './record/3D/' + START_TIME


def normalise_observation(agent, state):
    state_norm = agent.ob_normalize1.normalize(
        np.asarray(state))
    # reshape intp(?,)
    return state_norm.reshape(agent.state_dim)


def compute_disturbance_force(config):
    """Return force in x, y, z directions."""
    f = np.random.normal(0, 0.2) * 60*config['env']['HLC-frequency']
    theta = np.random.uniform(-np.pi, np.pi)
    fx = f * np.cos(theta)
    fy = f * np.sin(theta)
    return [fx, fy, 0]


def main():
    BEST_REWARD = 0
    network_frequency = config['env']['HLC-frequency']
    sampling_skip = int(config['env']['physics-frequency'] / network_frequency)
    reward_scale = config['train']['reward-scale']
    episode_steps_lim = config['train']['num-episode-steps']
    max_train_steps = int(config['train']['max-time'] * network_frequency)
    max_test_steps = int(config['test']['max-time'] * network_frequency)
    reward_decay = 1.0

    env = ValkyrieEnv(
        Kd=config['env']['Kd'],
        Kp=config['env']['Kp'],
        bullet_default_pd=config['env']['use-bullet-default-PD'],
        controlled_joints_list=config['env']['controlled-joints'],
        initial_gap_time=1,
        max_time=config['train']['max-time'],
        pd_freq=config['env']['LLC-frequency'],
        physics_freq=config['env']['physics-frequency'],
        renders=False)
    config['env']['state-dim'] = env.state_space
    agent = DDPG(env, config)

    # create new network
    os.makedirs(DIR_PATH, exist_ok=True)
    os.makedirs(DIR_PATH+'/saved_actor_networks', exist_ok=True)
    os.makedirs(DIR_PATH+'/saved_critic_networks', exist_ok=True)
    logging = logger(DIR_PATH)
    save_configuration(config, DIR_PATH)
    record_configuration(config, DIR_PATH)
    agent.load_weight(DIR_PATH)
    pprint(config)

    total_steps = 0
    # env.monitor.start('experiments/' + ENV_NAME,force=True)
    prev_action = np.zeros((agent.action_dim,))

    if config['env']['use-joint-interpolation']:
        joint_interpolate = {}
        for joint in config['actor']['action-joints']:
            interpolate = JointTrajectoryInterpolate()
            joint_interpolate[joint] = interpolate

    loss = 0
    for episode in range(config['train']['num-episodes']):
        state = env.reset(Kp=config['env']['Kp'], Kd=config['env']['Kd'])

        # Train
        # 4 dimension output of actor network: hip, knee, waist, ankle
        action = np.zeros(len(config['actor']['action-joints']))
        control_action = np.zeros(len(config['env']['controlled-joints']))
        next_state, reward, done, _ = env.step(control_action)
        agent.reset()

        episode_steps = 0
        while True:  # infinite loop
            rollout = 0
            for rollout in range(config['train']['num-rollout-steps']):
                episode_steps += 1
                total_steps += 1
                prev_action = np.array(action)

                # update action
                state = env.get_extended_observation()
                if agent.config['train']['normalize-observations']:
                    state = normalise_observation(agent, state)
                action = agent.action_noise(state)

                if config['env']['use-joint-interpolation']:
                    # setup joint interpolation
                    for i, joint in enumerate(config['actor']['action-joints']):
                        joint_interpolate[joint].cubic_interpolation_setup(
                            prev_action[i],
                            0,
                            action[i],
                            0,
                            1.0 / float(network_frequency))

                if config['env']['is-external-force-disturbance']:
                    if (episode_steps == network_frequency or
                        episode_steps == network_frequency * 6 or
                            episode_steps == network_frequency * 11):
                        # apply force for every 5 second
                        force = compute_disturbance_force(config)
                else:
                    force = [0, 0, 0]

                reward_accumulation = 0
                for i in range(sampling_skip):
                    if config['env']['use-joint-interpolation']:
                        for i, joint in enumerate(
                                config['actor']['action-joints']):
                            action[i] = joint_interpolate[joint].interpolate(
                                1.0 / config['env']['LLC-frequency'])

                    if len(control_action) == 7 and len(action) == 4:
                        control_action[0:4] = action
                        # duplicate leg control signals
                        control_action[4:7] = action[1:4]
                    elif len(control_action) == 11 and len(action) == 4:
                        control_action[0] = action[0]  # torso pitch
                        control_action[1] = 0.0  # hip roll
                        control_action[2] = action[1]  # hip pitch
                        control_action[3] = action[2]  # knee pitch
                        control_action[4] = action[3]  # ankle pitch
                        control_action[5] = 0.0  # ankle roll
                        control_action[6:11] = control_action[1:6]
                    elif len(control_action) == 13 and len(action) == 4:
                        control_action[0] = action[0]  # torso pitch
                        control_action[1] = 0.0  # hip yaw
                        control_action[2] = 0.0  # hip roll
                        control_action[3] = action[1]  # hip pitch
                        control_action[4] = action[2]  # knee pitch
                        control_action[5] = action[3]  # ankle pitch
                        control_action[6] = 0.0  # ankle roll
                        control_action[7:13] = control_action[1:7]
                    elif len(control_action) == 11 and len(action) == 11:
                        control_action[:] = action[:]
                    elif len(control_action) == 13 and len(action) == 11:
                        control_action[0] = action[0]  # torso pitch
                        control_action[1] = 0.0  # hip yaw
                        control_action[2] = action[1]  # hip roll
                        control_action[3] = action[2]  # hip pitch
                        control_action[4] = action[3]  # knee pitch
                        control_action[5] = action[4]  # ankle pitch
                        control_action[6] = action[5]  # ankle roll
                        control_action[7] = 0.0  # hip yaw
                        control_action[8] = action[6]  # hip roll
                        control_action[9] = action[7]  # hip pitch
                        control_action[10] = action[8]  # knee pitch
                        control_action[11] = action[9]  # ankle pitch
                        control_action[12] = action[10]  # ankle roll
                    elif len(control_action) == 13 and len(action) == 13:
                        control_action[:] = action[:]

                    next_state, reward, done, _ = env.step(
                        control_action, force)
                    reward_accumulation *= reward_decay
                    reward_accumulation += reward

                reward = reward_accumulation * reward_scale  # /sampling_skip
                reward -= (abs(prev_action[0] - action[0])
                           + abs(prev_action[1] - action[1])
                           + abs(prev_action[2] - action[2])
                           + abs(prev_action[3] - action[3])
                           + abs(prev_action[1] - action[1])
                           + abs(prev_action[2] - action[2])
                           + abs(prev_action[3] - action[3]))
                agent.store_transition(state, action, reward, next_state, done)
                if done or (episode_steps > max_train_steps):
                    break

            train_step = min(rollout+1, config['train']['num-episode-steps'])
            for train in range(train_step):
                loss = agent.perceive()
                logging.add_train('critic_loss', loss)
            if done or (episode_steps > max_train_steps):
                break

        if episode == 1 or (
            episode % 10 == 0 and
                total_steps > config['replay']['record-start-size']):
            total_reward = 0
            for i in range(config['test']['num-episodes']):
                env.reset(Kp=config['env']['Kp'], Kd=config['env']['Kd'])

                # 4 dimension output of actor network, hip, knee, waist, ankle
                action = np.zeros((len(config['actor']['action-joints']),))
                control_action = np.zeros(
                    (len(config['env']['controlled-joints']),))
                state, reward, done, _ = env.step(control_action)

                for j in range(max_test_steps):
                    prev_action = np.array(action)
                    state = env.getExtendedObservation()
                    if agent.config['train']['normalize-observations']:
                        state = normalise_observation(agent, state)
                    action = agent.action(state)  # direct action for test

                    reward_add = 0
                    if config['env']['joint-interpolation']:
                        # setup joint interpolation
                        for n in range(len(config['actor']['action-joints'])):
                            joint_name = config['actor']['action-joints'][n]
                            joint_interpolate[joint_name].cubic_interpolation_setup(
                                prev_action[n],
                                0,
                                action[n],
                                0,
                                1.0 / float(network_frequency))

                    if config['env']['is-external-force-disturbance']:
                        f = env.rejectableForce_xy(1.0/network_frequency)

                        if j == 5 * network_frequency:
                            if f[1] == 0:
                                force = np.array(
                                    [600.0*network_frequency/10.0, 0, 0])
                            else:
                                force = np.array([1.0 * f[1], 0, 0])
                            print(force)
                        elif j == 11 * network_frequency:
                            if f[0] == 0:
                                force = np.array(
                                    [-600.0*network_frequency/10.0, 0, 0])
                            else:
                                force = np.array([1.0 * f[0], 0, 0])
                            print(force)
                        elif j == 17 * network_frequency:
                            if f[2] == 0:
                                force = np.array(
                                    [0, -800.0*network_frequency/10.0, 0])
                            else:
                                force = np.array([0, 1.0 * f[2], 0])
                            print(force)
                        elif j == 23 * network_frequency:
                            if f[3] == 0:
                                force = np.array(
                                    [0, 800.0*network_frequency/10.0, 0])
                            else:
                                force = np.array([0, 1.0 * f[3], 0])
                            print(force)
                        else:
                            force = [0, 0, 0]
                    else:
                        force = [0, 0, 0]

                    # env.render()
                    for k in range(sampling_skip):
                        # if(sampling_skip%10==0):
                            # env.render()
                        if config['env']['joint-interpolation']:
                            for n in range(
                                    len(config['actor']['action-joints'])):
                                joint_name = config['actor']['action-joints'][n]
                                action[n] = joint_interpolate[joint_name].interpolate(
                                    1.0 / config['env']['LLC-frequency'])

                        if len(control_action) == 7 and len(action) == 4:
                            control_action[0:4] = action
                            # duplicate leg control signals
                            control_action[4:7] = action[1:4]
                        elif len(control_action) == 11 and len(action) == 4:
                            control_action[0] = action[0]  # torso pitch
                            control_action[1] = 0.0  # hip roll
                            control_action[2] = action[1]  # hip pitch
                            control_action[3] = action[2]  # knee pitch
                            control_action[4] = action[3]  # ankle pitch
                            control_action[5] = 0.0  # ankle roll
                            control_action[6:11] = control_action[1:6]
                        elif len(control_action) == 13 and len(action) == 4:
                            control_action[0] = action[0]  # torso pitch
                            control_action[1] = 0.0  # hip yaw
                            control_action[2] = 0.0  # hip roll
                            control_action[3] = action[1]  # hip pitch
                            control_action[4] = action[2]  # knee pitch
                            control_action[5] = action[3]  # ankle pitch
                            control_action[6] = 0.0  # ankle roll
                            control_action[7:13] = control_action[1:7]
                        elif len(control_action) == 11 and len(action) == 11:
                            control_action[:] = action[:]
                        elif len(control_action) == 13 and len(action) == 11:
                            control_action[0] = action[0]  # torso pitch
                            control_action[1] = 0.0  # hip yaw
                            control_action[2] = action[1]  # hip roll
                            control_action[3] = action[2]  # hip pitch
                            control_action[4] = action[3]  # knee pitch
                            control_action[5] = action[4]  # ankle pitch
                            control_action[6] = action[5]  # ankle roll
                            control_action[7] = 0.0  # hip yaw
                            control_action[8] = action[6]  # hip roll
                            control_action[9] = action[7]  # hip pitch
                            control_action[10] = action[8]  # knee pitch
                            control_action[11] = action[9]  # ankle pitch
                            control_action[12] = action[10]  # ankle roll
                        elif len(control_action) == 13 and len(action) == 13:
                            control_action[:] = action[:]

                        _, reward, done, _ = env.episode_steps(
                            control_action, force)
                        reward_add = reward+reward_decay*reward_add

                    reward = reward_add*reward_scale  # / sampling_skip
                    total_reward += reward
                    if done:
                        break

            avg_reward = total_reward/config['test']['num-episodes']
            # save training data
            if (BEST_REWARD < avg_reward and total_steps >
                    config['replay']['record-start-size']):
                BEST_REWARD = avg_reward
                agent.save_weight(total_steps, DIR_PATH)
            print(
                'episode:' +
                str(episode) +
                ' episode_steps:' +
                str(total_steps) +
                ' Evaluation Average Reward:' +
                str(avg_reward))
            logging.add_train('episode', episode)
            logging.add_train('episode_steps', total_steps)
            logging.add_train('avg_reward', avg_reward)

            logging.save_train()
            # TODO test observation normalization
            agent.ob_normalize1.save_normalization(DIR_PATH)
            # TODO test observation normalization
            agent.ob_normalize2.save_normalization(DIR_PATH)
        # if episode % 200 == 0 and episode > 1:
        #     agent.replay_buffer.save_menory(DIR_PATH)
        if total_steps > episode_steps_lim:
            break
    #  agent.save_weight(episode_steps, DIR_PATH)
    logging.save_train()
    agent.save_memory("replay_buffer.txt")


if __name__ == '__main__':
    main()
