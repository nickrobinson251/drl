import copy
import gc
import os
import time
from datetime import datetime

import gym
import pybullet
import pybullet_data
import pybullet_envs
from configuration import *
from agent_TRPO import *
from logger import logger
from replay_buffer import ReplayBuffer

gc.enable()

class Train():
    def __init__(self, config):
        self.config = config

        self.network_freq = 125#self.config.conf['HLC-frequency']

        self.max_time_per_train_episode = 10#self.config.conf['max-train-time']
        self.max_step_per_train_episode = int(self.max_time_per_train_episode*self.network_freq)
        self.max_time_per_test_episode = 10#self.config.conf['max-test-time']#16
        self.max_step_per_test_episode = int(self.max_time_per_test_episode*self.network_freq)

        env_name = 'Walker2DBulletEnv-v0'#'AntBulletEnv-v0'#'Walker2DBulletEnv-v0'#'HumanoidBulletEnv-v0'
        self.env = gym.make(env_name)
        # self.env.render()

        print(self.env.observation_space)
        print(self.env.action_space)
        self.config.conf['state-dim'] = self.env.observation_space.shape[0]
        self.config.conf['action-dim'] = self.env.action_space.shape[0]

        self.config.conf['actor-logstd-initial'] = np.zeros((1, self.config.conf['action-dim']))
        self.config.conf['actor-logstd-bounds'] = np.ones((2,self.config.conf['action-dim']))
        self.config.conf['actor-output-bounds'] = np.ones((2,self.config.conf['action-dim']))
        self.config.conf['actor-output-bounds'][0][:] = -1 * np.ones(self.config.conf['action-dim'],)
        self.config.conf['actor-output-bounds'][1][:] = 1* np.ones(self.config.conf['action-dim'],)

        self.config.conf['actor-logstd-initial'] *= np.log(1.0)  # np.log(min(std*0.25, 1.0))#0.5
        self.config.conf['actor-logstd-bounds'][0] *= np.log(0.2)
        self.config.conf['actor-logstd-bounds'][1] *= np.log(1.0)  # 0.6

        self.agent = Agent(self.env, self.config)

        self.episode_count = 0
        self.step_count = 0
        self.train_iter_count = 0

        self.best_reward = 0
        self.best_episode = 0
        self.best_train_iter = 0

        # load weight from previous network
        # dir_path = 'record/2017_12_04_15.20.44/no_force'  # '2017_05_29_18.23.49/with_force'

        # create new network
        dir_path = 'TRPO/record/' + 'gym/' + env_name +'/' + datetime.now().strftime('%Y_%m_%d_%H.%M.%S')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        if not os.path.exists(dir_path + '/saved_actor_networks'):
            os.makedirs(dir_path + '/saved_actor_networks')
        if not os.path.exists(dir_path + '/saved_critic_networks'):
            os.makedirs(dir_path + '/saved_critic_networks')
        self.logging = logger(dir_path)
        config.save_configuration(dir_path)
        config.record_configuration(dir_path)
        config.print_configuration()
        self.agent.load_weight(dir_path)
        self.dir_path = dir_path

        self.on_policy_paths = []
        self.off_policy_paths = []
        # self.buffer = ReplayBuffer(self.config.conf['replay-buffer-size'])

        self.force = [0,0,0]
        self.force_chest = [0, 0, 0]  # max(0,force_chest[1]-300*1.0 / EXPLORE)]
        self.force_pelvis = [0, 0, 0]

    def get_single_path(self):
        observations = []
        next_observations = []
        actions = []
        rewards = []
        actor_infos = []
        means = []
        logstds = []
        dones = []

        state = self.env.reset()
        self.episode_count+=1
        for step in range(self.max_step_per_train_episode):
            self.step_count+=1

            action, actor_info = self.agent.agent.actor.get_action(state)
            mean = actor_info['mean']
            logstd = actor_info['logstd']
            action = np.array([action]) if len(np.shape(action)) == 0 else np.array(action)
            next_state, reward, terminal, _ = self.env.step(action)

            observations.append(state)
            actions.append(action)
            rewards.append(reward)
            actor_infos.append(actor_info)
            means.append(mean)
            logstds.append(logstd)
            dones.append(terminal)
            next_observations.append(next_state)

            state = np.array(next_state)

            if terminal:
                break

        path = dict(observations=np.array(observations), actions=np.array(actions), rewards=np.array(rewards),
                    actor_infos=actor_infos, means=means, logstds=logstds, dones=dones,
                    next_observations=next_observations)
        return path

    def get_paths(self, num_of_paths=None, prefix='', verbose=True):
        if num_of_paths is None:
            num_of_paths = self.config.conf['max-path-num']
        paths = []
        t = time.time()
        if verbose:
            print(prefix + 'Gathering Samples')
        step_count = 0

        path_count = 0
        while(1):
            path = self.get_single_path()
            paths.append(path)
            step_count += len(path['dones'])
            path_count +=1
            num_of_paths = path_count
            if step_count>=self.config.conf['max-path-step']:
                break

        if verbose:
            print('%i paths sampled. %i steps sampled. %i total paths sampled. Total time used: %f.' % (num_of_paths, step_count, self.episode_count, time.time() - t))
        return paths

    def train_paths(self):
        self.train_iter_count+=1
        self.on_policy_paths = []  # clear
        self.off_policy_paths = [] #clear
        self.on_policy_paths = self.get_paths()
        # self.buffer.add_paths(self.on_policy_paths)
        self.off_policy_paths = copy.deepcopy(self.on_policy_paths)
        self.train_actor(True)
        self.train_critic(True)

    def train_critic(self, on_policy=True, prefix='', verbose = True):
        t = time.time()
        if on_policy == True:
            paths = copy.deepcopy(self.on_policy_paths)

            for path in paths:
                path['V_target'] = []
                path['Q_target'] = []

                rewards = path['rewards']
                next_observations= path['next_observations']
                dones = path['dones']


                r = np.array(rewards)

                if dones[-1] == False:  # not terminal state
                    r[-1] = r[-1] + self.config.conf['gamma']*self.agent.agent.V(next_observations[-1])  # bootstrap

                #path['returns'] = discount(path['rewards'], self.config.conf['gamma'])
                path['returns'] = discount(r, self.config.conf['gamma'])
                # print(discount(path['rewards'], self.config.conf['gamma']))
                # print(discount(r, self.config.conf['gamma']))
                # print(discount(path['rewards'], self.config.conf['gamma'])-discount(r, self.config.conf['gamma']))

            observations = np.concatenate([path['observations'] for path in paths])
            returns = np.concatenate([path['returns'] for path in paths])

            observations = np.vstack(observations)
            returns = np.vstack(returns)

            qloss = 0
            vloss = 0
            vloss += self.agent.train_V(observations, returns)
            print('qloss', qloss, 'vloss', vloss)

        else:
            paths = copy.deepcopy(self.off_policy_paths)
            #paths = self.off_policy_paths
            for path in paths:
                length = len(path['rewards'])

                path['V_target'] = []
                path['Q_target'] = []
                rewards = path['rewards']
                states = path['states']
                actions = path['actions']
                next_states = path['next_states']
                dones = path['dones']
                means = path['means']
                logstds = path['logstds']

                r = np.array(rewards)

                if dones[-1][0] == False:  # not terminal state
                    r[-1][0] = r[-1][0] + self.config.conf['gamma']*self.agent.agent.V(next_states[-1])  # bootstrap

                V_trace = self.agent.V_trace(states, actions, next_states, rewards, dones, means, logstds)

                path['V_target'] = V_trace
            V_target = np.concatenate([path['V_target'] for path in paths])
            states = np.concatenate([path['states'] for path in paths])
            # print(states)
            actions = np.concatenate([path['actions'] for path in paths])
            # print(actions)

            qloss = 0
            vloss = 0
            vloss += self.agent.train_V(states, V_target)
            print('qloss', qloss, 'vloss', vloss)

        if verbose:
            print(prefix + 'Training critic network. Total time used: %f.' % (time.time() - t))

        return

    def train_actor(self, on_policy=True, prefix='', verbose = True): # whether or not on policy
        t = time.time()
        stats = dict()
        if on_policy == True:
            paths = copy.deepcopy(self.on_policy_paths)
            length = len(paths)

            for path in paths:
                rewards = path['rewards']
                observations = path['observations']
                next_observations = path['next_observations']
                dones = path['dones']

                path['baselines'] = self.agent.agent.V(path['observations'])
                path['returns'] = discount(path['rewards'], self.config.conf['gamma'])
                if not self.config.conf['GAE']:
                    path['advantages'] = path['returns'] - path['baselines']
                else:
                    b = np.append(path['baselines'], path['baselines'][-1])
                    deltas = path['rewards'] + self.config.conf['gamma'] * b[1:] - b[:-1]
                    deltas[-1] = path['rewards'][-1] + (1-dones[-1])*self.config.conf['gamma']*b[-1]-b[-1]
                    #path['advantages'] = discount(deltas, self.config.conf['gamma'] * self.config.conf['lambda'])
                    path['advantages'] = np.squeeze(self.agent.GAE(observations, next_observations, rewards, dones))
                    # print(discount(deltas, self.config.conf['gamma'] * self.config.conf['lambda']))
                    # print(self.agent.GAE(observations, next_observations, rewards, dones))
                    # print(discount(deltas, self.config.conf['gamma'] * self.config.conf['lambda'])-np.squeeze(self.agent.GAE(observations, next_observations, rewards, dones)))

                if not self.config.conf['use-critic']:
                    r = np.array(rewards)
                    path['advantages'] = discount(r, self.config.conf['gamma'])

            observations = np.concatenate([path['observations'] for path in paths])
            actions = np.concatenate([path['actions'] for path in paths])
            rewards = np.concatenate([path['rewards'] for path in paths])
            advantages = np.concatenate([path['advantages'] for path in paths])
            actor_infos = np.concatenate([path['actor_infos'] for path in paths])
            means = np.concatenate([path['means'] for path in paths])
            logstds = np.concatenate([path['logstds'] for path in paths])
            returns = np.concatenate([path['returns'] for path in paths])

            if self.config.conf['center-advantage']:
                advantages -= np.mean(advantages)
                advantages /= (np.std(advantages) + 1e-8)

            # advantages = np.vstack(advantages)
            #advantages = advantages.reshape(length,1)

            self.agent.train_actor_TRPO(observations, actions, advantages, means, logstds)

        else:
            paths = self.off_policy_paths
            off_policy_states = np.concatenate([path['states'] for path in paths])
            off_policy_actions = np.concatenate([path['actions'] for path in paths])
            # print(states)

            self.agent.train_actor_DPG(off_policy_states, off_policy_actions)

        if verbose:
            print(prefix + 'Training actor network. Total time used: %f.' % (time.time() - t))

        return stats

    def test(self):
        total_reward = 0
        for i in range(self.config.conf['test-num']):#
            state = self.env.reset()

            for step in range(self.max_step_per_test_episode):

                action, actor_info = self.agent.agent.actor.get_action(state)
                mean = actor_info['mean']
                logstd = actor_info['logstd']
                action = mean
                action = np.array([action]) if len(np.shape(action)) == 0 else np.array(action)

                next_state, reward, done, _ = self.env.step(action)
                # self.env.render()
                # time.sleep(1. / 60.)
                total_reward += reward

                state = np.array(next_state)
                if done:
                    break
            #self.env.stopRendering()
        ave_reward = total_reward/self.config.conf['test-num']
        self.agent.save_weight(self.step_count, self.dir_path+'/latest_network')
        if self.best_reward<ave_reward and self.episode_count>self.config.conf['record-start-size']:
            self.best_episode = self.episode_count
            self.best_train_iter = self.train_iter_count
            self.best_reward=ave_reward
            self.agent.save_weight(self.step_count, self.dir_path+'/best_network')

        episode_rewards = np.array([np.sum(path['rewards']) for path in self.on_policy_paths])

        print('iter:' + str(self.train_iter_count) + ' episode:' + str(self.episode_count) + ' step:' + str(self.step_count)
              + ' Deterministic policy return:' + str(ave_reward) + ' Average return:' + str(np.mean(episode_rewards)))
        print('best train_iter', self.best_train_iter, 'best reward', self.best_reward)
        self.logging.add_train('episode', self.episode_count)
        self.logging.add_train('step', self.step_count)
        self.logging.add_train('ave_reward', ave_reward)
        self.logging.add_train('average_return', np.mean(episode_rewards))
        self.logging.add_train('logstd',np.squeeze(self.agent.agent.logstd()))

        #self.logging.save_train()
        #self.agent.ob_normalize1.save_normalization(self.dir_path)  # TODO test observation normalization
        #self.agent.ob_normalize2.save_normalization(self.dir_path)  # TODO test observation normalization
        self.logging.save_train()


def main():
    config = Configuration()
    train = Train(config)
    while 1:
        train.train_paths()
        #print(train.episode_count)
        # if train.episode_count == 10 or (train.episode_count%10 == 0 and train.step_count>train.config.conf['record-start-size']):
        #     train.test()
        # if train.episode_count <= 11 or (train.episode_count%10 == 0 and train.episode_count>train.config.conf['record-start-size']):
        #     train.test()
        # if train.episode_count <= 11 or (train.episode_count%10 == 0):
        #     train.test()
        train.test()
        if train.episode_count>config.conf['max-episode-num']:
            break
        if train.step_count>config.conf['max-step-num']:
            break
    return

if __name__ == '__main__':
    main()
