from datetime import datetime
import gc
import math
import numpy as np
import os
from pprint import pprint

from configuration import (config, save_configuration, record_configuration)
from ddpg import DDPG
from interpolate import JointTrajectoryInterpolate
from logger import logger
from valkyrie.envs.valkyrie_env import ValkyrieEnv

gc.enable()


def main():
    external_force_disturbance = True
    if external_force_disturbance:
        path_str = 'with_external_force_disturbance/'
    else:
        path_str = 'without_external_force_disturbance/'
    time_now = datetime.now().strftime('%Y_%m_%d_%H.%M.%S')
    DIR_PATH = 'record/' + '3D/' + path_str + time_now
    EPISODES = config['training']['epoch-num']
    TEST = config['training']['test-num']
    step_lim = config['training']['total-step-num']
    PD_frequency = config['env']['LLC-frequency']
    Physics_frequency = config['env']['physics-frequency']
    network_frequency = config['env']['HLC-frequency']
    sampling_skip = int(PD_frequency/network_frequency)

    reward_decay = 1.0
    reward_scale = config['training']['reward-scale']
    # /10.0#normalizing reward to 1#1.0/float(sampling_skip)
    reward_scale = reward_scale/float(sampling_skip)
    max_train_time = config['training']['max-train-time']
    max_train_steps = int(max_train_time*network_frequency)
    max_test_time = config['training']['max-test-time']
    max_test_steps = int(max_test_time*network_frequency)
    BEST_REWARD = 0

    env = ValkyrieEnv(
        max_time=max_train_time,
        renders=False,
        initial_gap_time=1,
        pd_freq=PD_frequency,
        physics_freq=Physics_frequency,
        Kp=config['env']['Kp'],
        Kd=config['env']['Kd'],
        bullet_default_pd=config['env']['use-bullet-default-PD'],
        controlled_joints_list=config['env']['controlled-joints'])
    config['env']['state-dim'] = env.stateNumber
    agent = DDPG(env, config)

    # load weight from previous network
    # DIR_PATH = 'record/2017_12_04_15.20.44/no_force'  #
    # '2017_05_29_18.23.49/with_force'

    # create new network
    # def make_recording_subdirectories(directory):
    os.makedirs(DIR_PATH, exist_ok=True)
    os.makedirs(DIR_PATH+'/saved_actor_networks', exist_ok=True)
    os.makedirs(DIR_PATH+'/saved_critic_networks', exist_ok=True)
    logging = logger(DIR_PATH)
    save_configuration(config, DIR_PATH)
    record_configuration(config, DIR_PATH)
    agent.load_weight(DIR_PATH)
    pprint(config)

    step_count = 0
    # env.monitor.start('experiments/' + ENV_NAME,force=True)

    prev_action = np.zeros((agent.action_dim,))

    if config['env']['joint-interpolation']:
        joint_interpolate = {}
        for joint in config['actor']['action-joints']:
            interpolate = JointTrajectoryInterpolate()
            # joint_interpolate[joint] = interpolate
            joint_interpolate.update({joint: interpolate})

    loss = 0

    for episode in range(EPISODES):
        # state = env._reset()
        state = env._reset(Kp=config['env']['Kp'], Kd=config['env']['Kd'])

        # Train
        # 4 dimension output of actor network, hip, knee, waist, ankle
        action = np.zeros((len(config['actor']['action-joints']),))
        control_action = np.zeros((len(config['env']['controlled-joints']),))
        next_state, reward, done, _ = env._step(control_action)
        # next_state = Valkyrie.getExtendedObservation()

        agent.reset()

        step = 0
        while 1 > 0:  # infinite loop
            rollout = 0

            for rollout in range(config['training']['rollout-step-num']):
                step += 1  # step count within an episode
                step_count += 1  # counting total steps during training
                prev_action = np.array(action)
                # update action
                state = env.getExtendedObservation()
                if agent.config['training']['normalize-observations']:
                    state_norm = agent.ob_normalize1.normalize(
                        np.asarray(state))
                    state_norm = np.reshape(
                        state_norm, (agent.state_dim))  # reshape intp(?,)
                else:
                    state_norm = state
                action = agent.action_noise(state_norm)
                # action = np.clip(action,action_bounds[0],action_bounds[1])

                # print(action)
                # env.render()
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

                if external_force_disturbance:
                    if (step == network_frequency or step ==
                            6 * network_frequency or step ==
                            11 * network_frequency):
                        # apply force for every 5 second
                        f = np.random.normal(0, 0.2) * 600*network_frequency/10
                        theta = np.random.uniform(-math.pi, math.pi)
                        fx = f * math.cos(theta)
                        fy = f * math.sin(theta)
                        force = [fx, fy, 0]
                    else:
                        force = [0, 0, 0]
                else:
                    force = [0, 0, 0]

                for i in range(sampling_skip):
                    if config['env']['joint-interpolation']:
                        for n in range(len(config['actor']['action-joints'])):
                            joint_name = config['actor']['action-joints'][n]
                            action[n] = joint_interpolate[joint_name].interpolate(
                                1.0 / PD_frequency)

                    # env.render()
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

                    next_state, reward, done, _ = env._step(
                        control_action, force)
                    reward_add = reward+reward_decay*reward_add

                reward = reward_add * reward_scale  # /sampling_skip
                reward -= (abs(prev_action[0] - action[0])
                           + abs(prev_action[1] - action[1])
                           + abs(prev_action[2] - action[2])
                           + abs(prev_action[3] - action[3])
                           + abs(prev_action[1] - action[1])
                           + abs(prev_action[2] - action[2])
                           + abs(prev_action[3] - action[3]))
                agent.store_transition(state, action, reward, next_state, done)
                if done:
                    break
                if step > max_train_steps:
                    break

            train_step = min(rollout+1, config['training']['train-step-num'])
            for train in range(train_step):
                loss = agent.perceive()
                logging.add_train('critic_loss', loss)

            if done:
                break
            if step > max_train_steps:
                break

        if episode == 1 or (
                episode %
                10 == 0 and step_count > config['replay']['record-start-size']):
            total_reward = 0
            for i in range(TEST):
                _ = env._reset(Kp=config['env']['Kp'], Kd=config['env']['Kd'])

                # 4 dimension output of actor network, hip, knee, waist, ankle
                action = np.zeros((len(config['actor']['action-joints']),))
                control_action = np.zeros(
                    (len(config['env']['controlled-joints']),))
                state, reward, done, _ = env._step(control_action)

                for j in range(max_test_steps):
                    prev_action = np.array(action)

                    state = env.getExtendedObservation()
                    if agent.config['training']['normalize-observations']:
                        state_norm = agent.ob_normalize1.normalize(np.asarray(
                                                                       state))
                        state_norm = np.reshape(
                            state_norm, (agent.state_dim))  # reshape intp(?,)
                    else:
                        state_norm = state
                    action = agent.action(state_norm)  # direct action for test
                    # action = np.clip(action,action_bounds[0],
                    #                  action_bounds[1])

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

                    if external_force_disturbance:
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
                                    1.0 / PD_frequency)

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

                        _, reward, done, _ = env._step(control_action, force)
                        reward_add = reward+reward_decay*reward_add

                    reward = reward_add*reward_scale  # / sampling_skip
                    total_reward += reward
                    if done:
                        break

            ave_reward = total_reward/TEST
            # save training data
            if (BEST_REWARD < ave_reward and step_count >
                    config['replay']['record-start-size']):
                BEST_REWARD = ave_reward
                agent.save_weight(step_count, DIR_PATH)
            print(
                'episode:' +
                str(episode) +
                ' step:' +
                str(step_count) +
                ' Evaluation Average Reward:' +
                str(ave_reward))
            logging.add_train('episode', episode)
            logging.add_train('step', step_count)
            logging.add_train('ave_reward', ave_reward)

            logging.save_train()
            # TODO test observation normalization
            agent.ob_normalize1.save_normalization(DIR_PATH)
            # TODO test observation normalization
            agent.ob_normalize2.save_normalization(DIR_PATH)
        # if episode % 200 == 0 and episode > 1:
        #     agent.replay_buffer.save_menory(DIR_PATH)
        if step_count > step_lim:
            break
    #  agent.save_weight(step, DIR_PATH)
    logging.save_train()
    agent.save_memory("replay_buffer.txt")


if __name__ == '__main__':
    main()
