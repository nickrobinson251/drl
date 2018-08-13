import gc
import inspect
import os
import time

import gym
import pybullet
import pybullet_data
import pybullet_envs

from configuration import *
from agent_TRPO import *
from logger import logger

gc.enable()

class Run():
    def __init__(self, config, dir_path):
        self.dir_path = dir_path
        self.config = config
        self.config.load_configuration(dir_path)
        self.config.print_configuration()

        self.network_freq = 125#self.config.conf['HLC-frequency']

        self.max_time_per_train_episode = 10#self.config.conf['max-train-time']
        self.max_step_per_train_episode = int(self.max_time_per_train_episode*self.network_freq)
        self.max_time_per_test_episode = 10#self.config.conf['max-test-time']#16
        self.max_step_per_test_episode = int(self.max_time_per_test_episode*self.network_freq)

        self.external_force_disturbance = True
        if self.external_force_disturbance == True:
            path_str = 'with_external_force_disturbance/'
        else:
            path_str = 'without_external_force_disturbance/'
        env_name = 'Walker2DBulletEnv-v0'#'HumanoidBulletEnv-v0'#'Walker2DBulletEnv-v0'
        self.env = gym.make(env_name)
        self.env.render()

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
        self.config.conf['actor-logstd-bounds'] *= np.log(0.01)
        self.config.conf['actor-logstd-bounds'] *= np.log(1.5)  # 0.6

        self.agent = Agent(self.env, self.config)
        self.agent.load_weight(dir_path+'/best_network')

        self.episode_count = 0
        self.step_count = 0
        self.train_iter_count = 0

        self.best_reward = 0
        self.best_episode = 0
        self.best_train_iter = 0

        self.logging = logger(dir_path)

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
                self.env.render()
                time.sleep(1. / 60.)
                total_reward += reward

                state = np.array(next_state)
                if done:
                    break
            #self.env.stopRendering()
        ave_reward = total_reward/self.config.conf['test-num']
        print(ave_reward)
        self.logging.save_run()


def main():
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(os.path.dirname(currentdir))
    os.sys.path.insert(0, parentdir)
    config = Configuration()
    dir_path = 'TRPO/record/gym/Walker2DBulletEnv-v0/2018_06_27_13.34.14'  # '2017_05_29_18.23.49/with_force'
    test = Run(config, dir_path)
    test.test()

    return

if __name__ == '__main__':
    main()
