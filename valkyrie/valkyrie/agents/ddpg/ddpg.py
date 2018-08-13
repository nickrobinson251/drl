import tensorflow as tf
import numpy as np

from actor_network import ActorNetwork
from critic_network import CriticNetwork
from grad_inverter import grad_inverter
from normalize2 import BatchNormalize, OnlineNormalize
from ou_noise import OUNoise
from prioritized_replay import Memory
from replay_buffer import ReplayBuffer
from utils import translate


class DDPG:
    """docstring for DDPG"""

    def __init__(self, env, config):
        self.name = 'DDPG'  # name for uploading results
        self.environment = env
        self.config = config
        # Randomly initialize actor network and critic network
        # with both their target networks
        self.state_dim = config['env']['state-dim']
        # waist, hip, knee, ankle env._actionDim
        self.action_dim = config['actor']['action-dim']

        self.replay_buffer_size = config['replay']['buffer-size']
        self.replay_start_size = config['replay']['start-size']
        self.batch_size = config['training']['batch-size']
        self.gamma = config['env']['gamma']
        self.action_bounds = config['actor']['action-bounds']
        self.actor_output_bounds = config['actor']['action-bounds']

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
        self.sess = tf.InteractiveSession(
            config=tf.ConfigProto(
                gpu_options=gpu_options))

        #self.sess = tf.InteractiveSession()

        self.actor_network = ActorNetwork(
            self.sess, self.state_dim, self.action_dim, config)
        self.critic_network = CriticNetwork(
            self.sess, self.state_dim, self.action_dim, config)

        # initialize replay buffer
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)
        self.memory = Memory(capacity=self.replay_buffer_size)
        self.replay_buffer_count = 0

        # Initialize a random process the Ornstein-Uhlenbeck process for action
        # exploration
        if self.config['noise']['use-normalization']:
            self.exploration_noise = OUNoise(
                self.action_dim,
                (self.config['noise']['OU-settings'][1]
                 * self.config['actor']['action-scale']),
                (self.config['noise']['OU-settings'][2]
                 * self.config['actor']['action-scale'])
            )
        else:
            self.exploration_noise = OUNoise(self.action_dim)

        #self.grad_inv = grad_inverter(self.actor_output_bounds, self.sess)
        self.grad_inv = grad_inverter(self.actor_output_bounds, self.sess)

        self.actor_network.perturb_policy()

        self.ob_normalize1 = BatchNormalize(
            self.state_dim,
            config['replay']['buffer-size'])  # TODO test observation normalization
        self.ob_normalize2 = OnlineNormalize(
            self.state_dim,
            config['replay']['buffer-size'])  # TODO test observation normalization

    def train(self):
        train_num = 1
        loss = 0
        for i in range(0, train_num):
            # Sample a random minibatch of N transitions from replay buffer
            tree_idx, batch_memory, ISWeights = [], [], []
            if self.config['replay']['use-prioritized-experience']:
                tree_idx, batch_memory, ISWeights = self.memory.sample(
                    self.batch_size)
            else:
                batch_memory = self.replay_buffer.get_batch(self.batch_size)

            state_batch = np.asarray([data[0] for data in batch_memory])
            action_batch = np.asarray([data[1] for data in batch_memory])
            reward_batch = np.asarray([data[2] for data in batch_memory])
            next_state_batch = np.asarray([data[3] for data in batch_memory])
            done_batch = np.asarray([data[4] for data in batch_memory])

            if self.config['training']['normalize-observations']:
                state_batch = self.ob_normalize1.normalize(state_batch)
                next_state_batch = self.ob_normalize1.normalize(
                    next_state_batch)

            # for action_dim = 1
            action_batch = np.resize(
                action_batch, [
                    self.batch_size, self.action_dim])

            # Calculate y_batch

            #next_action_batch = self.actor_network.actions_target(next_state_batch)
            # print('next_action_batch'+str(np.shape(next_action_batch)))
            next_action_batch = self.actions_target(next_state_batch)
            #print('next_action_batch' + str(np.shape(next_action_batch)))
            q_value_batch = self.critic_network.q_value_target(
                next_state_batch, next_action_batch)
            y_batch = []
            for i in range(self.batch_size):
                if done_batch[i]:
                    y_batch.append(reward_batch[i])
                else:
                    y_batch.append(
                        reward_batch[i] +
                        self.gamma *
                        q_value_batch[i])
            y_batch = np.resize(y_batch, [self.batch_size, 1])
            # Update critic by minimizing the loss L
            # print(self.memory.tree.data_pointer)
            # print(tree_idx)

            if self.config['replay']['use-prioritized-experience']:
                # print(ISWeights)
                ISWeights = np.asarray(ISWeights)
                ISWeights = np.resize(ISWeights, [self.batch_size, 1])
                # print(np.size(ISWeights))
                loss, abs_errors = self.critic_network.train(
                    y_batch, state_batch, action_batch, ISWeights)
                # print(abs_errors)
                self.memory.batch_update(tree_idx, abs_errors)
            else:
                loss = self.critic_network.train(
                    y_batch, state_batch, action_batch, [])

            # Update the actor policy using the sampled gradient:
            #action_batch_for_gradients = self.actor_network.actions(state_batch)
            #print('action_batch_for_gradients' + str(np.shape(action_batch_for_gradients)))
            action_batch_for_gradients = self.actions(state_batch)
            #print('action_batch_for_gradients' + str(np.shape(action_batch_for_gradients)))
            q_gradient_batch = self.critic_network.gradients(
                state_batch, action_batch_for_gradients)
            q_gradient_batch = self.grad_inv.invert(
                q_gradient_batch, action_batch_for_gradients)

            self.actor_network.train(q_gradient_batch, state_batch)

        # Update the target networks
        self.actor_network.update_target()
        self.critic_network.update_target()

        return loss

    def action_noise(self, state, epsilon=1, lamda=1):
        # Select action a_t according to the current policy and exploration
        # noise
        if self.config['noise']['use-param-noise']:
            action = self.actor_network.action_noise(state)
            # print(self.actor_network.action_noise(state)-self.actor_network.action(state))
        else:
            action = self.actor_network.action(state)
        # print(self.exploration_noise.noise(action))
        if self.config['noise']['use-OU']:
            ou_noise = epsilon * lamda * self.exploration_noise.noise(action)
            # normalizing noise for each action output
            # if self.config['noise-normalization'] == True:
            #     for i in range(self.action_dim):
            #         ou_noise[i] *= abs(self.config['action-bounds'][1][i]-self.config['action-bounds'][0][i])/2.0

            action = action + ou_noise

        # translate action from [-1.0,1.0] to joint range
        for i in range(self.action_dim):
            action[i] = translate(
                action[i], [
                    self.actor_output_bounds[0][i], self.actor_output_bounds[1][i]], [
                    self.action_bounds[0][i], self.action_bounds[1][i]])

        # clip action 2018-01-14
        action = np.clip(action, self.action_bounds[0], self.action_bounds[1])
        return action

    def action(self, state):
        action = self.actor_network.action(state)

        # translate action from [-1.0,1.0] to joint range
        for i in range(self.action_dim):
            action[i] = translate(
                action[i], [
                    self.actor_output_bounds[0][i], self.actor_output_bounds[1][i]], [
                    self.action_bounds[0][i], self.action_bounds[1][i]])

        # clip action 2018-01-14
        action = np.clip(action, self.action_bounds[0], self.action_bounds[1])
        return action

    def actions_target(self, state):
        actions = np.array(self.actor_network.actions_target(state))
        actions_scale = np.zeros(np.shape(actions))

        for i in range(self.action_dim):
            # translate action from [-1.0,1.0] to joint range
            action_temp = translate(
                actions[:, i],
                [self.actor_output_bounds[0][i],
                 self.actor_output_bounds[1][i]],
                [self.action_bounds[0][i],
                 self.action_bounds[1][i]])
            actions_scale[:, i] = action_temp  # .reshape(self.batch_size,1)
            # clip action 2018-01-14
            actions_scale[:, i] = np.clip(
                actions_scale[:, i], self.action_bounds[0][i], self.action_bounds[1][i])

        return actions_scale

    def actions(self, state):
        actions = np.array(self.actor_network.actions(state))
        actions_scale = np.zeros(np.shape(actions))

        for i in range(self.action_dim):
            # translate action from [-1.0,1.0] to joint range
            action_temp = translate(
                actions[:, i],
                [self.actor_output_bounds[0][i],
                 self.actor_output_bounds[1][i]],
                [self.action_bounds[0][i],
                 self.action_bounds[1][i]])
            actions_scale[:, i] = action_temp  # .reshape(self.batch_size,1)
            # clip action 2018-01-14
            actions_scale[:, i] = np.clip(
                actions_scale[:, i], self.action_bounds[0][i], self.action_bounds[1][i])

        return actions_scale

    def perceive(self):
        loss = 0
        if self.replay_buffer_count > self.replay_start_size:
            loss = self.train()

        return loss

    def reset(self):
        # Re-iniitialize the random process when an episode ends
        if self.config['noise']['use-param-noise']:
            self.param_noise()
            #print(conf['param-noise'] == True)
        if self.config['noise']['use-OU']:
            self.exploration_noise.reset()

    def save_weight(self, time_step, dir_path):
        print("Now we save model")
        self.actor_network.save_network(time_step, dir_path)
        self.critic_network.save_network(time_step, dir_path)

    def load_weight(self, dir_path):
        # Now load the weight
        print("Now we load the weight")
        self.actor_network.load_network(dir_path)
        self.critic_network.load_network(dir_path)

    def save_memory(self, filename):
        self.replay_buffer.save_menory(filename)

    def load_memory(self, filename):
        self.replay_buffer.load_memory(filename)

    def param_noise(self):
        # update parameter noise spec
        # self.actor_network.perturb_policy()
        distance = 0
        if self.config['replay']['use-prioritized-experience']:
            if self.replay_buffer_count > self.replay_start_size:
                #batch_memory = self.memory.sample_random(self.batch_size)
                tree_idx, batch_memory, ISWeights = self.memory.sample(
                    self.batch_size)

                state_batch = np.asarray([data[0] for data in batch_memory])
                distance = self.actor_network.adapt_param_noise(state_batch)

                # print(distance)
                # print(self.actor_network.param_noise.current_stddev)
        else:
            if self.replay_buffer_count > self.replay_start_size:
                batch_memory = self.replay_buffer.get_batch(self.batch_size)

                state_batch = np.asarray([data[0] for data in batch_memory])
                distance = self.actor_network.adapt_param_noise(state_batch)

                # print(distance)
                # print(self.actor_network.param_noise.current_stddev)

        self.actor_network.perturb_policy()

    def store_transition(self, s, a, r, s_, d):
        # Store transition (s_t,a_t,r_t,s_{t+1}) in replay buffer
        # TODO test observation normalization
        self.ob_normalize2.update(np.resize(s, [1, self.state_dim]))
        self.replay_buffer_count += 1
        self.replay_buffer_count = min(
            self.replay_buffer_count,
            self.replay_buffer_size)

        if self.config['replay']['use-prioritized-experience']:
            transition = (s, a, r, s_, d)
            # have high priority for newly arrived transition
            self.memory.store(transition)
            self.replay_buffer.add(s, a, r, s_, d)
        else:
            self.replay_buffer.add(s, a, r, s_, d)

        if self.replay_buffer_count > (self.batch_size*2):
            # Sample a random minibatch of N transitions from replay buffer
            batch_memory = []
            if self.config['replay']['use-prioritized-experience']:
                tree_idx, batch_memory, ISWeights = self.memory.sample(
                    self.batch_size)
                # print(self.memory.tree.data_pointer)
            else:
                batch_memory = self.replay_buffer.get_batch(self.batch_size)

            state_batch = np.asarray([data[0] for data in batch_memory])
            # TODO test observation normalization
            self.ob_normalize1.update(state_batch)
