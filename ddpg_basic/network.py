import tensorflow as tf
import numpy as np
from replay_buffer import ReplayBuffer
from configuration import *
from normalize2 import *
from util import *
import tensorflow.contrib as tc
from sklearn.utils import shuffle
from grad_inverter import *
from symmetry_op import *
from get_action_from_state_op import *
import random
from build_graph import *

class Network:
    def __init__(self, config, state_dim, action_dim):
        #        config = tf.ConfigProto()
        #        config.gpu_options.per_process_gpu_memory_fraction = 0.25
        #        self.sess = tf.InteractiveSession(config=config)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

        #self.sess = tf.InteractiveSession()

        self.config = config
        self.state_dim = state_dim  # env.state_dim#config.conf['state-dim']
        self.action_dim = action_dim  # env.action_dim#config.conf['action-dim'] # waist, hip, knee, ankle env._actionDim
        self.tau = self.config.conf['tau']
        self.critic_l2_reg = self.config.conf['critic-l2-reg']
        self.critic_lr = self.config.conf['critic-lr']
        self.is_prioritized_replay = self.config.conf['prioritized-exp-replay']

        self.opt_method = {}
        self.opt_method['actor'] = self.config.conf['actor-opt-method']
        self.opt_method['critic'] = self.config.conf['critic-opt-method']

        self.action_bounds = config.conf['action-bounds']
        self.actor_output_bounds = config.conf['actor-output-bounds']#config.conf['normalized-action-bounds']
        self.logstd_bounds = config.conf['actor-output-bounds']

        self.build_graph = Build_Graph(config, state_dim, action_dim)

        self.state_input = {}
        self.action_input = {}
        with tf.name_scope('inputs'):
            # define placeholder for inputs to network
            self.state_input['actor'] = tf.placeholder("float", [None, self.state_dim])
            self.state_input['critic'] = tf.placeholder("float", [None, self.state_dim])
            self.action_input['critic'] = tf.placeholder("float", [None, self.action_dim])
            self.stochastic_flag = tf.placeholder(tf.bool, shape=[], name="stochastic_flag")
            # # variables needed for calculating symmetry loss, needs to be created before reuse


        self.mirror_op = mirror_op(state_dim, action_dim) #create operation to mirror state and action
        self.get_action_from_state = get_action_from_state_op(state_dim, action_dim)
        self.grad_inv = grad_inverter(self.actor_output_bounds, self.sess)
        self.clip_action = clip_action(self.actor_output_bounds, self.sess)

        with tf.variable_scope("agent"):

            self.actor_network = {}
            self.action_norm_dist, \
            self.action_output, \
            self.action_mu, \
            self.action_pi, \
            self.action_logstd, \
            self.actor_network['vars'], \
            self.actor_network['trainable_vars'], \
            self.actor_network['perturbable_vars'],\
                = self.build_graph.create_actor_network(self.state_input['actor'], self.stochastic_flag, 'actor_network',\
                                                trainable=True, config=self.config,\
                                                state_dim=self.state_dim, action_dim=self.action_dim)

            self.target_actor_network = {}
            self.target_action_norm_dist, \
            self.target_action_output, \
            self.target_action_mu, \
            self.target_action_pi, \
            self.target_action_logstd, \
            self.target_actor_network['vars'], \
            self.target_actor_network['trainable_vars'], \
            self.target_actor_network['perturbable_vars'],\
                = self.build_graph.create_actor_network(self.state_input['actor'], self.stochastic_flag, 'target_actor_network',\
                                                trainable=False, config=self.config,\
                                            state_dim=self.state_dim, action_dim=self.action_dim)



            self.critic_network = {}
            self.Q_output, \
            self.critic_network['vars'], \
            self.critic_network['trainable_vars'], \
                = self.build_graph.create_critic_network(self.state_input['critic'], self.action_input['critic'], 'critic_network', \
                                             trainable=True, config=self.config,\
                                             state_dim=self.state_dim, action_dim=self.action_dim)

            self.target_critic_network = {}
            self.target_Q_output, \
            self.target_critic_network['vars'], \
            self.target_critic_network['trainable_vars'], \
                = self.build_graph.create_critic_network(self.state_input['critic'], self.action_input['critic'], 'target_critic_network', \
                                             trainable=False, config=self.config,\
                                             state_dim=self.state_dim, action_dim=self.action_dim)

            tf.get_variable_scope().reuse_variables()
            # symmetric network
            with tf.name_scope('mirror_actor_network'):
                self.mirror_state_input = self.mirror_op.mirror_state_op(self.state_input['actor'])
                _, \
                self.src_action_output, \
                self.src_action_mu, \
                self.src_action_pi,\
                _, \
                _, \
                _, \
                _, \
                    = self.build_graph.create_actor_network(self.mirror_state_input, self.stochastic_flag, 'actor_network',\
                                                    trainable=True, config=self.config,\
                                                state_dim=self.state_dim, action_dim=self.action_dim)

                self.mirror_action_pi = self.mirror_op.mirror_action_op(self.src_action_pi)
                self.mirror_action_mu = self.mirror_op.mirror_action_op(self.src_action_mu)

            # actor network for sending action to critic to calculate deterministic policy gradient
            # shares weight with actor network
            with tf.name_scope('actor_network_for_critic'):
                self.q_actor_network = {}
                self.q_action_norm_dist, \
                self.q_action_output, \
                self.q_action_mu, \
                self.q_action_pi,\
                self.q_action_logstd, \
                self.q_actor_network['vars'], \
                self.q_actor_network['trainable_vars'], \
                    = self.build_graph.create_actor_network(self.state_input['critic'], self.stochastic_flag, 'actor_network',\
                                                    trainable=True, config=self.config,\
                                                    state_dim=self.state_dim, action_dim=self.action_dim)
            with tf.name_scope('critic_network_for_pi'):
                self.critic_network_pi = {}
                self.Q_pi_output,\
                self.critic_network_pi['vars'], \
                self.critic_network_pi['trainable_vars'], \
                    = self.build_graph.create_critic_network(self.state_input['critic'], self.q_action_pi, 'critic_network', \
                                                 trainable=True, config=self.config,\
                                                 state_dim=self.state_dim, action_dim=self.action_dim)
            with tf.name_scope('critic_network_for_mu'):
                self.critic_network_mu = {}
                self.Q_mu_output,\
                self.critic_network_mu['vars'], \
                self.critic_network_mu['trainable_vars'], \
                    = self.build_graph.create_critic_network(self.state_input['critic'], self.q_action_mu, 'critic_network', \
                                                 trainable=True, config=self.config,\
                                                 state_dim=self.state_dim, action_dim=self.action_dim)

        # trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='agent')
        # for var in self.old_actor_network['vars']:
        #     print(var.name)
        with tf.name_scope('network_update_op'):
            self.setup_network_update()
        with tf.name_scope('actor_train_op'):
            self.setup_actor_training_method()
        with tf.name_scope('critic_train_op'):
            self.setup_critic_training_method()

        writer = tf.summary.FileWriter("logs/", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        self.sess.run(self.hard_copy['target_actor'])
        self.sess.run(self.hard_copy['target_critic'])

    def get_network_update(self, vars, target_vars, tau):  # copy from vcriticar into target_var
        soft_update = []
        hard_copy = []
        assert len(vars) == len(target_vars)
        for var, target_var in zip(vars, target_vars):
            hard_copy.append(tf.assign(target_var, var))
            soft_update.append(tf.assign(target_var, (1. - tau) * target_var + tau * var))
        assert len(hard_copy) == len(vars)
        assert len(soft_update) == len(vars)
        return tf.group(*hard_copy), tf.group(*soft_update)


    def setup_network_update(self):
        self.hard_copy = {}
        self.soft_update = {}
        #print(self.actor_network['vars'])

        self.hard_copy['target_actor'], self.soft_update['target_actor'] = \
            self.get_network_update(self.actor_network['vars'], self.target_actor_network['vars'], self.tau)

        self.hard_copy['target_critic'], self.soft_update['target_critic'] = \
            self.get_network_update(self.critic_network['vars'], self.target_critic_network['vars'], self.tau)

        self.logstd_input = tf.placeholder("float", [self.action_dim])
        self.set_logstd_op = self.action_logstd.assign(self.logstd_input)
        self.set_target_logstd_op = self.target_action_logstd.assign(self.logstd_input)

        return

    def setup_critic_training_method(self):
        self.Q_input = tf.placeholder("float", [None, 1])
        weight_decay = tf.add_n([self.critic_l2_reg * tf.nn.l2_loss(var) for var in self.critic_network['trainable_vars']])
        if self.is_prioritized_replay == True:
            self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')#importance sampling for experience replay
            self.abs_errors = tf.reduce_sum(tf.abs(self.Q_input - self.Q_output), axis=1)  # TD_error for updating Sumtree
            self.qloss = tf.reduce_mean(self.ISWeights * tf.square(self.Q_input - self.Q_output))
            self.qloss = self.qloss + weight_decay
            #self.cost = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.y_input, self.q_value_output)) + weight_decay
        else:
            self.qloss = tf.reduce_mean(tf.square(self.Q_input - self.Q_output))
            self.qloss = self.qloss + weight_decay

        self.Q_optimizer = tf.train.AdamOptimizer(self.critic_lr).minimize(self.qloss)
        self.q_action_gradients = tf.gradients(self.Q_output, self.action_input)

    def setup_actor_training_method(self):
        #input

        self.q_gradient_input = tf.placeholder("float", [None, self.action_dim])

        #mean range loss
        self.mean_bound_loss = self.grad_inv.invert_loss(self.action_mu)

        #entropy loss
        self.entropy = tf.reduce_mean(entropy(self.action_logstd) )#encourage exploration maximize entropy
        self.entropy_loss = -self.entropy #

        # symmetric loss
        self.sym_loss = tf.reduce_mean(tf.square(self.action_mu-self.mirror_action_mu))

        #kl
        #tf.stop_gradient can onpy be applied to tensors, self.old_action_norm_dist is not tensor
        target_dist_mean = tf.stop_gradient(self.target_action_mu) #stop gradient from back propagating to old network
        target_dist_logstd = tf.stop_gradient(self.target_action_logstd)
        new_dist_mean = self.action_mu
        new_dist_logstd = self.action_logstd
        kl = kl_sym(target_dist_mean, target_dist_logstd, new_dist_mean, new_dist_logstd)
        #kl = tf.reduce_mean(tf.distributions.kl_divergence(self.target_action_norm_dist, self.action_norm_dist), -1)#stop gradient from backpropagating to old network
        self.kl = tf.reduce_mean(kl, axis = -1)
        self.kl_mean = tf.reduce_mean(kl)

        self.kl_grad = tf.gradients(self.kl, self.actor_network['vars'])

        self.parameters_gradients = tf.gradients(self.action_output, self.actor_network['trainable_vars'], -self.q_gradient_input)
        #self.parameters_gradients, _ = tf.clip_by_global_norm(self.parameters_gradients, 1.0) #clipping gradient

        self.param_grad = self.parameters_gradients
        k_dot_g = tf.reduce
        #zip back to gradient value tuples
        self.actor_optimizer = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(self.parameters_gradients, self.actor_network['trainable_vars']))



    def update_old_policy(self):
        #self.sess.run(self.hard_copy['actor'])
        self.sess.run(self.soft_update['target_actor'])
        return

    def update_critic(self):
        self.sess.run(self.soft_update['target_critic'])
        return

    def train_critic_Q(self, state_batch, action_batch, Q_batch): # train_num=1
        loss = 0
        kl = 0
        train_num = self.opt_method['critic']['train-num']
        for _ in range(train_num):
            loss, _ = self.sess.run([self.qloss, self.Q_optimizer], feed_dict={
                self.Q_input: Q_batch,
                self.state_input['critic']: state_batch,
                self.action_input['critic']: action_batch,
            })
                #print(kl)
        #self.update_critic()
        return loss

    def action(self, state, flag):
        if state.ndim < 2:  # no batch
            state = state[np.newaxis, :]
            return self.sess.run(self.action_output, feed_dict={
                self.state_input['actor']: state,
                self.stochastic_flag: flag,
            })[0]
        else:
            state_batch = state
            return self.sess.run(self.action_output, feed_dict={
                self.state_input['actor']: state_batch,
                self.stochastic_flag: flag,
            })
    def logstd(self):
        return self.sess.run(self.action_logstd, feed_dict={})

    def set_logstd(self, logstd):
        self.sess.run(self.set_logstd_op, feed_dict={self.logstd_input: logstd})
        # self.sess.run([self.set_logstd_op,self.set_old_logstd_op], feed_dict={self.logstd_input:logstd})

    def get_actor_info(self,state):
        state = state[np.newaxis, :]
        mu = self.sess.run(self.action_mu, feed_dict={
                self.state_input['actor']: state,
        })[0]
        logstd = self.sess.run(self.action_logstd)
        return mu, logstd

    def q_action_gradients(self, state_batch, action_batch):
        grad_batch = self.sess.run(self.q_action_gradients, feed_dict={
            self.state_input['critic']: state_batch,
            self.action_input['critic']: action_batch,
        })[0]
        #grad_batch = self.grad_inv.invert(grad_batch, action_batch) #TODO
        return grad_batch

    def Q(self,state,action):
        if state.ndim<2: # no batch
            state = state[np.newaxis,:] # add new axis
            action = action[np.newaxis,:]
            return self.sess.run(self.Q_output, feed_dict={
                self.state_input['critic']: state,
                self.action_input['critic']: action,
            })[0]
        else:
            state_batch = state
            action_batch = action
            return self.sess.run(self.Q_output, feed_dict={
                self.state_input['critic']: state_batch,
                self.action_input['critic']: action_batch
            })
    def Q_mu(self, state):
        if state.ndim<2: # no batch
            state = state[np.newaxis,:] # add new axis
            return self.sess.run(self.Q_mu_output, feed_dict={
                self.state_input['critic']: state,
                self.state_input['actor']: state,
            })[0]
        else:
            state_batch = state
            return self.sess.run(self.Q_mu_output, feed_dict={
                self.state_input['critic']: state_batch,
                self.state_input['actor']: state_batch,
            })

    def Q_pi(self, state):
        if state.ndim<2: # no batch
            state = state[np.newaxis,:] # add new axis
            return self.sess.run(self.Q_pi_output, feed_dict={
                self.state_input['critic']: state,
                self.state_input['actor']: state,
                self.stochastic_flag: True,
            })[0]
        else:
            state_batch = state
            return self.sess.run(self.Q_pi_output, feed_dict={
                self.state_input['critic']: state_batch,
                self.state_input['actor']: state_batch,
                self.stochastic_flag: True,
            })

    def Q_pi_Est(self, state):
        if state.ndim<2: # no batch
            state = state[np.newaxis,:] # add new axis
            length = np.shape(state)[0]#first dimension
            Q_pi_Est = np.zeros((length,1))
            n = 30
            for i in range(n):
                Q_pi_Est += self.sess.run(self.Q_pi_output, feed_dict={
                self.state_input['critic']: state,
                self.state_input['actor']: state,
                self.stochastic_flag: True,
                })
            Q_pi_Est /= n
            return Q_pi_Est[0]
        else:
            state_batch = state
            length = np.shape(state)[0]#first dimension
            Q_pi_Est = np.zeros((length,1))
            n=30
            for i in range(n):
                Q_pi_Est += self.sess.run(self.Q_pi_output, feed_dict={
                self.state_input['critic']: state_batch,
                self.state_input['actor']: state_batch,
                self.stochastic_flag: True,
                })

            Q_pi_Est /= n
            return Q_pi_Est

    def load_network(self, dir_path):
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(dir_path + '/saved_networks')
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:" + checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    def save_network(self, time_step, dir_path):
        print('save network...' + str(time_step))
        self.saver.save(self.sess, dir_path + '/saved_networks/' + 'network')  # , global_step = time_step)


    def save_actor_variable(self, time_step, dir_path):
        var_dict = {}
        for var in self.actor_network['trainable_vars']:
            var_array = self.sess.run(var)
            var_dict[var.name] = var_array
            #print(var.name + '  shape: ')
            #print(np.shape(var_array))
        output = open(dir_path + '/actor_variable.obj', 'wb')
        pickle.dump(var_dict, output)
        output.close()

    def load_actor_variable(self, dir_path):
        var = {}
        pkl_file = open(dir_path + '/actor_variable.obj', 'rb')
        var_temp = pickle.load(pkl_file)
        var.update(var_temp)
        pkl_file.close()
        return var

    def transfer_actor_variable(self, var_dict):
        for var in self.actor_network['trainable_vars']:
            for key in var_dict:
                if key in var.name: # if variable name contains similar strings
                    self.sess.run(tf.assign(var, var_dict[var.name]))
                    print(var.name + ' transfered')

        self.sess.run(self.hard_copy['target_actor'])

    def save_critic_variable(self, time_step, dir_path):
        var_dict = {}
        for var in self.critic_network['trainable_vars']:
            var_array = self.sess.run(var)
            var_dict[var.name] = var_array
            #print(var.name + '  shape: ')
            #print(np.shape(var_array))
        output = open(dir_path + '/critic_variable.obj', 'wb')
        pickle.dump(var_dict, output)
        output.close()

    def load_critic_variable(self, dir_path):
        var = {}
        pkl_file = open(dir_path + '/critic_variable.obj', 'rb')
        var_temp = pickle.load(pkl_file)
        var.update(var_temp)
        pkl_file.close()
        return var

    def transfer_critic_variable(self, var_dict):
        for var in self.critic_network['trainable_vars']:
            for key in var_dict:
                if key in var.name: # if variable name contains similar strings
                    self.sess.run(tf.assign(var, var_dict[var.name]))
                    print(var.name + ' transfered')

        self.sess.run(self.hard_copy['target_critic'])