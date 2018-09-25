from Interpolate import *
import numpy as np

class Control:
    def __init__(self, config, env):
        self.env = env
        self.config = config

        self.PD_freq = self.config.conf['LLC-frequency']
        self.Physics_freq = self.config.conf['Physics-frequency']
        self.network_freq = self.config.conf['HLC-frequency']
        self.sampling_skip = int(self.PD_freq/self.network_freq)

        self.joint_interpolate = {}
        for joint in self.config.conf['actor-action-joints']:
            interpolate = JointTrajectoryInterpolate()
            # joint_interpolate[joint] = interpolate
            self.joint_interpolate.update({joint: interpolate})

        self.reward_decay = 1.0
        self.reward_scale = config.conf['reward-scale']

        self.action = np.zeros((len(self.config.conf['actor-action-joints']),))
        self.prev_action = np.zeros((len(self.config.conf['actor-action-joints']),))
        self.action_interpolate = np.zeros((len(self.config.conf['actor-action-joints']),))
        self.action_control = np.zeros((len(self.config.conf['controlled-joints']),))

        # self.actor_bound = np.ones((2,len(self.config.conf['controlled-joints'])))
        # self.actor_bound[0][:] = -1 * np.ones((len(self.config.conf['controlled-joints']),))
        # self.actor_bound[1][:] = 1* np.ones((len(self.config.conf['controlled-joints']),))
        self.actor_bound = np.ones((2,len(self.config.conf['actor-action-joints'])))
        # self.actor_bound[0][:] = -1 * np.ones((len(self.config.conf['actor-action-joints']),))
        # self.actor_bound[1][:] = 1* np.ones((len(self.config.conf['actor-action-joints']),))
        self.actor_bound=self.config.conf['actor-output-bounds']
        self.control_bound = self.config.conf['action-bounds']

        self.info = []

    def control_step(self, action, force = np.array([0,0,0]), action_reference = None):
        self.force = force
        self.action_reference = action_reference
        self.action = action
        self.action = self.rescale(self.action, self.actor_bound, self.control_bound) #rescaled action
        # self.action = np.clip(self.action, self.config.conf['action-bounds'][0], self.config.conf['action-bounds'][1])
        #self.action = np.array(self.action)
        for n in range(len(self.config.conf['actor-action-joints'])):
            joint_name = self.config.conf['actor-action-joints'][n]
            self.joint_interpolate[joint_name].cubic_interpolation_setup(self.prev_action[n], 0, self.action[n], 0,
                                                                         1.0 / float(self.network_freq))
        reward_add = 0
        for i in range(self.sampling_skip):
            if self.config.conf['joint-interpolation']:
                for n in range(len(self.config.conf['actor-action-joints'])):
                    joint_name = self.config.conf['actor-action-joints'][n]
                    self.action_interpolate[n] = self.joint_interpolate[joint_name].interpolate(1.0 / self.PD_freq)
            else:
                self.action_interpolate = self.action
            # env.render()
            if len(self.action_control) == 7 and len(self.action_interpolate) == 4:
                self.action_control[0:4] = self.action_interpolate[0:4]
                self.action_control[4:7] = self.action_interpolate[1:4]  # duplicate leg control signals
            elif len(self.action_control) == 11 and len(self.action_interpolate) == 4:
                self.action_control[0] = self.action_interpolate[0]  # torso pitch
                self.action_control[1] = 0.0  # hip roll
                self.action_control[2] = self.action_interpolate[1]  # hip pitch
                self.action_control[3] = self.action_interpolate[2]  # knee pitch
                self.action_control[4] = self.action_interpolate[3]  # ankle pitch
                self.action_control[5] = 0.0  # ankle roll
                self.action_control[6:11] = self.action_control[1:6]
            elif len(self.action_control) == 13 and len(self.action_interpolate) == 4:
                self.action_control[0] = self.action_interpolate[0]  # torso pitch
                self.action_control[1] = 0.0  # hip yaw
                self.action_control[2] = 0.0  # hip roll
                self.action_control[3] = self.action_interpolate[1]  # hip pitch
                self.action_control[4] = self.action_interpolate[2]  # knee pitch
                self.action_control[5] = self.action_interpolate[3]  # ankle pitch
                self.action_control[6] = 0.0  # ankle roll
                self.action_control[7:13] = self.action_control[1:7]
            elif len(self.action_control) == 11 and len(self.action_interpolate) == 11:
                self.action_control[:] = self.action_interpolate[:]
            elif len(self.action_control) == 13 and len(self.action_interpolate) == 11:
                self.action_control[0] = self.action_interpolate[0]  # torso pitch
                self.action_control[1] = 0.0  # hip yaw
                self.action_control[2] = self.action_interpolate[1]  # hip roll
                self.action_control[3] = self.action_interpolate[2]  # hip pitch
                self.action_control[4] = self.action_interpolate[3]  # knee pitch
                self.action_control[5] = self.action_interpolate[4]  # ankle pitch
                self.action_control[6] = self.action_interpolate[5]  # ankle roll
                self.action_control[7] = 0.0  # hip yaw
                self.action_control[8] = self.action_interpolate[6]  # hip roll
                self.action_control[9] = self.action_interpolate[7]  # hip pitch
                self.action_control[10] = self.action_interpolate[8]  # knee pitch
                self.action_control[11] = self.action_interpolate[9]  # ankle pitch
                self.action_control[12] = self.action_interpolate[10]  # ankle roll
            elif len(self.action_control) == 13 and len(self.action_interpolate) == 13:
                self.action_control[:] = self.action_interpolate[:]
            # self.action_control = self.rescale(self.action_control, self.actor_bound, self.control_bound)
            self.action_control = np.clip(self.action_control, self.config.conf['action-bounds'][0], self.config.conf['action-bounds'][1])
            next_state, reward, done, _ = self.env._step(self.action_control, self.force)

            reward_add = reward + self.reward_decay * reward_add
        #reward *= 0.1
        reward_add/=self.sampling_skip
        reward_task = (reward_add+10*self.reward_action_bound(self.action, self.control_bound))*self.reward_scale

        reward_imitation = 10*self.reward_action_ref(self.action, self.action_reference)*self.reward_scale
        reward = 0.7*reward_task + 0.3*reward_imitation
        self.prev_action = np.array(self.action)
        self.info = dict([('task_reward', reward_task),
                          ('imitation_reward', reward_imitation),
                          ('total_reward', reward)])
        return np.array(next_state), reward, done, self.info


    def reset(self):
        self.info = []
        self.action = np.zeros((len(self.config.conf['actor-action-joints']),))
        self.prev_action = np.zeros((len(self.config.conf['actor-action-joints']),))
        self.action_interpolate = np.zeros((len(self.config.conf['actor-action-joints']),))
        self.action_control = np.zeros((len(self.config.conf['controlled-joints']),))

    def rescale(self, value, old_range, new_range):
        value = np.array(value)
        old_range = np.array(old_range)
        new_range = np.array(new_range)

        OldRange = old_range[1][:] - old_range[0][:]
        NewRange = new_range[1][:] - new_range[0][:]
        NewValue = (value - old_range[0][:]) * NewRange / OldRange + new_range[0][:]
        return NewValue

    def clip_action(self, state, action):

        return

    def reward_action_bound(self, action, bound):
        lower_bound = np.maximum(0.0, bound[0]-action)
        upper_bound = np.maximum(0.0, action-bound[1])
        reward = -np.mean(lower_bound+upper_bound)
        #print(reward)
        return reward

    def reward_action_ref(self, action, action_ref):
        alpha = 1e-2#1e-1
        action_ref_error = action-action_ref
        action_ref_error /= ((self.actor_bound[1]-self.actor_bound[0])/8)
        reward = np.mean(np.exp(np.log(alpha) * (action_ref_error) ** 2))
        #print(reward)
        return reward