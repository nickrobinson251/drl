from sklearn.utils import shuffle

from TRPO import TRPO
from util import *


class Agent:
    """docstring for TRPO"""

    def __init__(self, env, config):

        self.config = config
        self.state_dim = config.conf['state-dim']
        self.action_dim = config.conf['action-dim'] # waist, hip, knee, ankle env._actionDim

        self.agent = TRPO(config, self.state_dim, self.action_dim)

        # Randomly initialize actor network and critic network
        # with both their target networks

        self.batch_size = config.conf['critic-batch-size']
        self.gamma = config.conf['gamma']
        self.lamb = config.conf['lambda']
        self.action_bounds = config.conf['action-bounds']
        self.actor_output_bounds = config.conf['actor-output-bounds']#config.conf['normalized-action-bounds']

    def train_actor_TRPO(self,on_policy_state, on_policy_action, on_policy_advantage, mean, logstd):
        on_policy_state = np.vstack(on_policy_state)
        on_policy_action = np.vstack(on_policy_action)
        mean = np.vstack(mean)
        logstd = np.vstack(logstd)

        self.agent.check_network_update()
        self.agent.update_old_policy()
        self.agent.check_network_update()
        #print(cv.shape)
        b = np.zeros((len(on_policy_advantage), 1))
        #print(eta.shape)
        l = on_policy_advantage
        # mean_temp, logstd_temp = self.agent.get_actor_info(on_policy_state)
        # print(mean-mean_temp)
        #loss, kl = self.agent.train_actor_IPG(on_policy_state, on_policy_action, on_policy_state, l, b, DPG_flag)
        loss, kl = self.agent.train_actor_TRPO(on_policy_state, on_policy_action, on_policy_advantage, mean, logstd)

        logstd = self.agent.logstd()
        #print(logstd)
        bound = self.config.conf['actor-logstd-bounds']
        logstd = np.clip(logstd, bound[0], bound[1])
        #print(logstd)
        self.agent.set_logstd(logstd)

        return loss, kl

    def train_actor_PPO(self,on_policy_state, on_policy_action, on_policy_advantage, mean, logstd):
        on_policy_state = np.vstack(on_policy_state)
        on_policy_action = np.vstack(on_policy_action)
        mean = np.vstack(mean)
        logstd = np.vstack(logstd)
        
        self.agent.check_network_update()
        self.agent.update_old_policy()
        self.agent.check_network_update()
        # feed_dict = {self.input: obs, self.value: rns}
        iter_num = self.config.conf['PPO-method']['epoch']
        data_size, _ = on_policy_state.shape
        assert on_policy_state.shape[0] == on_policy_advantage.shape[0]
        batch_size = self.config.conf['PPO-method']['actor-batch-size']
        # n = obs.shape[0]
        # inds = np.arange(n)
        num_of_batch = data_size//batch_size#-(-data_size)//batch_size
        loss = 0
        kl = 0
        for i in range(int(iter_num)):
            idx = np.random.permutation(data_size)
            start = 0
            for i in np.arange(num_of_batch):
                start = i*batch_size
                end = start+batch_size
                minibatch_state = on_policy_state[idx[start:end]]
                minibatch_action = on_policy_action[idx[start:end]]
                minibatch_advantage = on_policy_advantage[idx[start:end]]
                minibatch_mean = mean[idx[start:end]]
                minibatch_logstd = logstd[idx[start:end]]
                #start += batch_size
                loss, kl = self.agent.train_actor_PPO(minibatch_state, minibatch_action, minibatch_advantage, minibatch_mean, minibatch_logstd)

        logstd = self.agent.logstd()
        #print(logstd)
        bound = self.config.conf['actor-logstd-bounds']
        logstd = np.clip(logstd, bound[0], bound[1])
        #print(logstd)
        self.agent.set_logstd(logstd)

        return loss, kl

    def train_V(self,state, discount_reward):
        # feed_dict = {self.input: obs, self.value: rns}
        iter_num = self.config.conf['critic-iteration']
        data_size, _ = state.shape
        assert state.shape[0] == discount_reward.shape[0]
        batch_size = self.config.conf['critic-batch-size']
        # n = obs.shape[0]
        # inds = np.arange(n)
        num_of_batch = data_size//batch_size#-(-data_size)//batch_size
        vloss = 0
        for i in range(int(iter_num)):
            idx = np.random.permutation(data_size)
            start = 0
            for i in np.arange(num_of_batch):
                start = i*batch_size
                end = start+batch_size
                minibatch_state = state[idx[start:end]]
                minibatch_reward = discount_reward[idx[start:end]]
                #start += batch_size
                loss = self.agent.train_critic_V(minibatch_state, minibatch_reward)
                vloss = loss
        return vloss #Final Value Network Loss

    def action_noise(self, state):
        action = self.agent.action(state)
        action = np.clip(action, self.actor_output_bounds[0], self.actor_output_bounds[1])

        return action

    def action(self, state):
        action = self.agent.action_mean(state)
        action = np.clip(action, self.actor_output_bounds[0], self.actor_output_bounds[1])

        return action

    def actions_noise(self, state):
        action = self.agent.action(state)
        action = np.clip(action, self.actor_output_bounds[0], self.actor_output_bounds[1])

        return action

    def actions(self, state):
        action = self.agent.action_mean(state)
        action = np.clip(action, self.actor_output_bounds[0], self.actor_output_bounds[1])

        return action

    def reset(self):
        #clear the buffer
        self.buffer = []

        #logstd = self.actor_network.logstd()
        #print(logstd)
        #range = self.config.conf['actor-logstd-range']
        #logstd = np.clip(logstd, range[0], range[1])
        #print(logstd)
        #self.actor_network.set_logstd(logstd)

    def GAE(self, state, next_state, reward, done):
        state = np.vstack(state)
        next_state = np.vstack(next_state)
        reward_batch = np.vstack(reward)
        done = np.vstack(done)
        value = self.agent.V(state)
        next_value = self.agent.V(next_state)

        delta_v = reward_batch + self.config.conf['gamma']*next_value - value
        size = value.shape[0]
        # if done[size-1] == True: # terminal state
        #     delta_v[size-1][0] = reward_batch[size-1][0] - value[size-1][0]
        if done[-1][0] == True: # terminal state
            delta_v[-1][0] = reward_batch[-1][0] - value[-1][0]

        GAE = discount(delta_v, self.config.conf['gamma']*self.config.conf['lambda'])
        GAE = np.vstack(GAE)
        return GAE

    def extract_joint_angle(self, state):
        joint = state
        return joint

    def get_actor_info(self, state):
        mu, logstd = self.agent.get_actor_info(state)
        actor_info = dict(mu=mu,logstd=logstd)
        return actor_info

    def save_weight(self, time_step, dir_path):
        print("Now we save model")
        self.agent.save_network(time_step, dir_path)
        self.agent.save_actor_variable(time_step, dir_path)
        self.agent.save_critic_variable(time_step, dir_path)

    def load_weight(self, dir_path):
        # Now load the weight
        print("Now we load the weight")
        self.agent.load_network(dir_path)

    def V_monte(self, done, reward, next_state):
        reward = np.array(reward)
        if done[-1][0] == False: #not terminal state
            reward[-1][0] += self.agent.V(next_state[-1]) #bootstrap if not terminal state
        V = discount(reward, self.config.conf['gamma'])
        return V


    def V_trace(self, state, action, next_state, reward, done, mean, logstd): #IMPALA: Scalable Deep-RL with Importance Weighted Actor-Learner Rchitectures
        rho_thresh = 1.0 #importance sampling
        c_thresh = 1.0 #trace cutting
        length = len(done)
        mean_beh = mean
        logstd_beh = logstd

        mean_pi = self.agent.action(state)
        logstd_pi = np.tile(self.agent.logstd(), (length, 1))

        log_prob_beh = log_likelihood(action, mean_beh, logstd_beh)
        log_prob_pi = log_likelihood(action, mean_pi, logstd_pi)
        rho = np.exp(log_prob_pi - log_prob_beh)
        rho = np.nan_to_num(rho)
        rho = np.clip(rho, 0.0, 10.0)
        rho = np.vstack(rho)

        #rho = rho[np.isnan(rho)] = 0.0
        #rho = rho[np.isinf(rho)] = 0.0
        rho = np.minimum(np.ones(np.shape(rho))*rho_thresh, rho)
        c = np.minimum(np.ones(np.shape(rho))*c_thresh, rho)*self.lamb

        v = self.agent.V(state)#self.agent.Q_pi_Est(state) #self.agent.V(state)
        delta = reward + self.gamma*self.agent.V(next_state) - self.agent.V(state)
        if done[length-1][0] == True:#terminal state
            s = next_state[length-1][:]
            delta[length-1][0] = reward[length-1][0] - self.agent.V(s)
        delta = rho*delta

        V_target = np.zeros(np.shape(reward))


        if done[length - 1][0]: #terminal state
            V_trace = 0
        else:
            s = next_state[-1][:]
            V_trace = self.agent.V(s)  # self.agent.Q_pi_Est(s)#self.agent.V(s)

        # print(c)
        v_next = self.agent.V(next_state)
        for i in range(length - 1, -1, -1):

            # s = next_state[i][:]
            # temp = V_trace - self.agent.V(s)
            temp = V_trace - v_next[i][0]

            V_trace = v[i][0] + delta[i][0] + self.gamma*c[i][0]*temp
            V_target[i][0] = V_trace

        V_target = np.vstack(V_target)
        return V_target

