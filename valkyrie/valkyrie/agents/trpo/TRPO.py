import tensorflow as tf
import numpy as np
# from util import *
from actor import GaussianActor
from critic import Critic
from net import Fcnn
import os
import pickle


class TRPO:
    def __init__(self, config, state_dim, action_dim):
        #        config = tf.ConfigProto()
        #        config.gpu_options.per_process_gpu_memory_fraction = 0.25
        #        self.sess = tf.InteractiveSession(config=config)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
        self.sess = tf.InteractiveSession(
            config=tf.ConfigProto(gpu_options=gpu_options))

        self.name = 'TRPO'
        self.config = config
        self.state_dim = state_dim  # env.state_dim#config.conf['state-dim']
        # env.action_dim#config.conf['action-dim'] # waist, hip, knee, ankle
        # env._actionDim
        self.action_dim = action_dim
        self.tau = self.config.conf['tau']
        self.critic_l2 = self.config.conf['critic-opt-method']['critic-l2-reg']

        self.opt_method = {}
        self.opt_method['actor'] = self.config.conf['actor-opt-method']
        self.opt_method['critic'] = self.config.conf['critic-opt-method']

        self.action_bounds = config.conf['action-bounds']
        # config.conf['normalized-action-bounds']
        self.actor_output_bounds = config.conf['actor-output-bounds']
        self.logstd_bounds = config.conf['actor-output-bounds']

        self.state_input = {}
        self.action_input = {}
        with tf.name_scope('inputs'):
            # define placeholder for inputs to network
            self.state_input['actor'] = tf.placeholder(
                "float", [None, self.state_dim])
            self.action_input['actor'] = tf.placeholder(
                "float", [None, self.action_dim])
            self.state_input['critic'] = tf.placeholder(
                "float", [None, self.state_dim])
            self.action_input['critic'] = tf.placeholder(
                "float", [None, self.action_dim])
            # # variables needed for calculating symmetry loss, needs to be created before reuse

        # actor param
        self.actor_input_dim = self.state_dim
        self.actor_output_dim = self.action_dim
        self.actor_layer_dim = self.config.conf['actor-layer-size']
        self.actor_bias = [True]
        # np.sqrt(1.0/np.concatenate([[self.actor_input_dim],
        # self.actor_layer_dim],axis=0))#[0.1,0.1,0.1,0.01]
        self.actor_init_weight = [0.1, 0.1, 0.1, 0.01]
        self.actor_init_bias = [None, None, None, None]
        self.actor_activation = self.config.conf['actor-activation-fn']

        # critic_param
        self.critic_input_dim = self.state_dim
        self.critic_output_dim = 1
        self.critic_layer_dim = self.config.conf['critic-layer-size']
        self.critic_bias = [True]
        self.critic_init_weight = np.sqrt(
            2.0 / np.concatenate(
                [[self.critic_input_dim],
                 self.actor_layer_dim],
                axis=0))  # [0.1,0.1,0.1,0.1]
        self.critic_init_bias = [None, None, None, None]
        self.critic_activation = self.config.conf['critic-activation-fn']
        with tf.variable_scope("agent"):

            with tf.variable_scope('actor_network'):
                self.actor_network = Fcnn(
                    sess=self.sess,
                    input_dim=self.state_dim,
                    output_dim=self.action_dim,
                    layer_dim=self.actor_layer_dim,
                    config=self.config,
                    if_bias=self.actor_bias,
                    activation=self.actor_activation,
                    init_weight=self.actor_init_weight,
                    init_bias=self.actor_init_bias,
                    input_tf=self.state_input['actor'],
                    name='actor_network',
                    trainable=True,
                    reusable=False)
                self.actor = GaussianActor(
                    net=self.actor_network,
                    sess=self.sess,
                    config=self.config)
                self.action_mu = self.actor.action_mean
                self.action_pi = self.actor.action
                self.action_logstd = self.actor.action_logstd
                self.action_norm_dist = self.actor.action_norm_dist
                self.actor_vars = self.actor.var_list
                self.actor_trainable_vars = self.actor.trainable_var_list

            with tf.variable_scope('old_actor_network'):
                self.old_actor_network = Fcnn(
                    sess=self.sess,
                    input_dim=self.state_dim,
                    output_dim=self.action_dim,
                    layer_dim=self.actor_layer_dim,
                    config=self.config,
                    if_bias=self.actor_bias,
                    activation=self.actor_activation,
                    init_weight=self.actor_init_weight,
                    init_bias=self.actor_init_bias,
                    input_tf=self.state_input['actor'],
                    name='old_actor_network',
                    trainable=True,
                    reusable=False)
                self.old_actor = GaussianActor(
                    net=self.old_actor_network, sess=self.sess,
                    config=self.config)
                self.old_action_mu = self.actor.action_mean
                self.old_action_pi = self.actor.action
                self.old_action_logstd = self.actor.action_logstd
                self.old_action_norm_dist = self.actor.action_norm_dist
                self.old_actor_vars = self.old_actor.var_list
                self.old_actor_trainable_vars = self.old_actor.trainable_var_list

            with tf.variable_scope('critic_network'):
                self.critic_network = Fcnn(
                    sess=self.sess,
                    input_dim=self.state_dim,
                    output_dim=1,
                    layer_dim=self.critic_layer_dim,
                    config=self.config,
                    if_bias=self.critic_bias,
                    activation=self.critic_activation,
                    init_weight=self.critic_init_weight,
                    init_bias=self.critic_init_bias,
                    input_tf=self.state_input['critic'],
                    name='critic_network',
                    trainable=True,
                    reusable=False)
                self.critic = Critic(
                    net=self.critic_network,
                    sess=self.sess,
                    config=self.config)
                self.V_output = self.critic.output
                self.critic_vars = self.critic.var_list
                self.critic_trainable_vars = self.critic.trainable_var_list

        trainable_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='agent')
        for var in self.old_actor_trainable_vars:
            print(var.name)
        for var in self.old_actor_vars:
            print(var.name)
        for var in self.actor_trainable_vars:
            print(var.name)
        for var in self.actor_vars:
            print(var.name)
        for var in self.critic_trainable_vars:
            print(var.name)
        for var in self.critic_vars:
            print(var.name)

        with tf.name_scope('actor_train_op'):
            self.setup_actor_training_method()
        with tf.name_scope('critic_train_op'):
            self.setup_critic_training_method()
        with tf.name_scope('network_update_op'):
            self.setup_network_update()

        writer = tf.summary.FileWriter("logs/", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        self.sess.run(self.hard_copy['actor'])
        # self.sess.run(self.hard_copy['target_actor'])
        # self.sess.run(self.hard_copy['target_critic'])

    # copy from vcriticar into target_var
    def get_network_update(self, vars, target_vars, tau):
        soft_update = []
        hard_copy = []
        assert len(vars) == len(target_vars)
        for var, target_var in zip(vars, target_vars):
            hard_copy.append(tf.assign(target_var, var))
            soft_update.append(
                tf.assign(
                    target_var,
                    (1. - tau) * target_var + tau * var))
        assert len(hard_copy) == len(vars)
        assert len(soft_update) == len(vars)
        return tf.group(*hard_copy), tf.group(*soft_update)

    def setup_network_update(self):
        self.hard_copy = {}
        self.soft_update = {}
        # print(self.actor_network['vars'])
        self.hard_copy['actor'], self.soft_update['actor'] = \
            self.get_network_update(self.actor_vars, self.old_actor_vars, self.tau)

        self.set_logstd_op = self.action_logstd.assign(self.logstd_input)
        self.set_old_logstd_op = self.old_action_logstd.assign(
            self.logstd_input)

        # used to check whether the old network is correctly updated
        flat_new_var = flatten_var(self.actor_vars)
        flat_old_var = flatten_var(self.old_actor_vars)
        self.actor_var_diff = tf.reduce_sum(flat_new_var-flat_old_var)

        return

    def setup_critic_training_method(self):
        # Define training optimizer
        self.yV_input = tf.placeholder("float", [None, 1])
        # L2 regularization
        weight_decay = tf.add_n(
            [self.critic_l2 * tf.nn.l2_loss(var)
             for var in self.critic_trainable_vars])

        self.advantage = self.yV_input - self.V_output
        self.vloss = tf.reduce_mean(tf.square(self.advantage))
        self.vloss_l2 = self.vloss+weight_decay
        if self.opt_method['critic']['critic-l2']:
            self.V_optimizer = tf.train.AdamOptimizer(
                self.opt_method['critic']['critic-lr']).minimize(self.vloss_l2)
        else:
            self.V_optimizer = tf.train.AdamOptimizer(
                self.opt_method['critic']['critic-lr']).minimize(self.vloss)

    def setup_actor_training_method(self):
        # input
        self.mean_input = tf.placeholder("float", [None, self.action_dim])
        self.logstd_input = tf.placeholder("float", [None, self.action_dim])
        self.advantage_input = tf.placeholder(
            "float", [None], name='advantage_input')

        # entropy loss
        # encourage exploration maximize entropy
        self.entropy = tf.reduce_mean(entropy(self.action_logstd))
        self.entropy_loss = -self.entropy

        self.old_dist_mean = tf.placeholder(
            tf.float32, [None, self.action_dim],
            name='old_dist_mean')
        self.old_dist_logstd = tf.placeholder(
            tf.float32, [None, self.action_dim],
            name='old_dist_logstd')
        self.new_dist_mean = self.action_mu
        self.new_dist_logstd = self.action_logstd

        logli_new = log_likelihood_tf(
            self.action_input['actor'],
            self.new_dist_mean,
            self.new_dist_logstd)
        logli_old = log_likelihood_tf(
            self.action_input['actor'],
            self.old_dist_mean,
            self.old_dist_logstd)
        self.ratio = tf.exp(logli_new - logli_old)

        self.surr_loss1 = - self.ratio * self.advantage_input
        self.surr_loss2 = -self.advantage_input * tf.clip_by_value(
            self.ratio, 1.0 - self.config.conf['PPO-method']['epsilon'],
            1.0 + self.config.conf['PPO-method']['epsilon'])
        # + self.config.conf['loss-entropy-coeff']*self.entropy_loss
        self.surr_loss = tf.reduce_mean(
            tf.maximum(self.surr_loss1, self.surr_loss2))

        self.kl = tf.reduce_mean(
            kl_sym(
                self.old_dist_mean,
                self.old_dist_logstd,
                self.new_dist_mean,
                self.new_dist_logstd))
        # TRPO loss
        surr_grad = tf.gradients(self.surr_loss, self.actor_trainable_vars)
        self.flat_surr_grad = flatten_var(surr_grad)

        batchsize = tf.cast(tf.shape(self.state_input['actor'])[0], tf.float32)
        self.flat_tangent = tf.placeholder(
            tf.float32, shape=[None], name='flat_tangent')
        kl_firstfixed = kl_sym_firstfixed(
            self.new_dist_mean, self.new_dist_logstd)
        #grads = tf.gradients(kl_firstfixed, self.actor_trainable_vars)
        #grads = tf.gradients(self.kl, self.actor_trainable_vars)
        grads = tf.gradients(
            kl_sym(
                self.old_dist_mean,
                self.old_dist_logstd,
                self.new_dist_mean,
                self.new_dist_logstd),
            self.actor_trainable_vars)

        flat_grads = flatten_var(grads) / batchsize
        self.fvp = tf.gradients(
            tf.reduce_sum(
                flat_grads *
                self.flat_tangent),
            self.actor_trainable_vars)
        self.flat_fvp = flatten_var(self.fvp)

        self.weights_to_set = tf.placeholder(
            tf.float32, [None], name='weights_to_set')
        self.set_var_from_flat = set_from_flat(
            self.actor_trainable_vars, self.weights_to_set)
        self.flatten_var = flatten_var(self.actor_trainable_vars)

        # PPO loss
        self.PPO_loss = self.surr_loss + \
            self.config.conf['loss-entropy-coeff']*self.entropy_loss
        PPO_grads = tf.gradients(self.PPO_loss, self.actor_trainable_vars)
        PPO_grads = list(zip(PPO_grads, self.actor_trainable_vars))
        self.PPO_optimizer = tf.train.AdamOptimizer(
            learning_rate=self.config.conf['PPO-method']['actor-lr'])
        self.PPO_train_op = self.PPO_optimizer.apply_gradients(PPO_grads)

    def train_actor_TRPO(
            self,
            on_policy_state,
            on_policy_action,
            on_policy_advantage,
            old_dist_mean,
            old_dist_logstd):

        flat_theta_prev = self.sess.run(self.flatten_var)

        feed_dict = {self.state_input['actor']: on_policy_state,
                     self.advantage_input: on_policy_advantage,
                     self.action_input['actor']: on_policy_action,
                     self.old_dist_mean: old_dist_mean,  # [:,np.newaxis],
                     self.old_dist_logstd: old_dist_logstd,  # [:,np.newaxis]
                     }

        def fisher_vector_product(p):
            feed_dict[self.flat_tangent] = p
            return self.sess.run(
                self.flat_fvp,
                feed_dict=feed_dict) + self.config.conf['actor-opt-method']['cg-damping'] * p

        g = self.sess.run(self.flat_surr_grad, feed_dict=feed_dict)
        step_gradient = cg(
            fisher_vector_product, -g,
            cg_iters=self.config.conf['actor-opt-method']
            ['conjugate-gradient-iteration'])
        sAs = step_gradient.dot(fisher_vector_product(step_gradient))
        inv_stepsize = np.sqrt(
            sAs / (2. * self.config.conf['actor-opt-method']['epsilon']))
        stepsize = np.sqrt(
            (2. * self.config.conf['actor-opt-method']['epsilon']) /
            (sAs + 1e-8))
        print('g', sum(g))
        print('step graadient', sum(step_gradient))
        print('sAs', sAs)
        print('inv stepsize', inv_stepsize)
        fullstep_gradient = step_gradient / (inv_stepsize + 1e-8)
        # print('surr loss 1', self.sess.run(self.surr_loss1, feed_dict=feed_dict))
        # print('surr loss 2', self.sess.run(self.surr_loss2, feed_dict=feed_dict))
        # print('surr loss', self.sess.run(self.surr_loss, feed_dict=feed_dict))

        def loss_function(x):
            self.sess.run(
                self.set_var_from_flat, feed_dict={
                    self.weights_to_set: x})
            surr_loss, kl = self.sess.run(
                [self.surr_loss, self.kl],
                feed_dict=feed_dict)
            # self.sess.run(set_from_flat(self.actor.var_list, flat_theta_prev))
            return surr_loss, kl

        if self.config.conf['actor-opt-method']['line-search']:
            flat_theta_new = linesearch(
                loss_function,
                flat_theta_prev,
                fullstep_gradient,
                self.config.conf['actor-opt-method']['line-search-backtrack'],
                self.config.conf['actor-opt-method']['epsilon'])
        else:
            flat_theta_new = flat_theta_prev + fullstep_gradient
        self.sess.run(
            self.set_var_from_flat, feed_dict={
                self.weights_to_set: flat_theta_prev})
        update_gradient = flat_theta_new - flat_theta_prev
        print('sum of full step gradient', sum(fullstep_gradient))
        print('sum of change in parameter', sum(update_gradient))
        # step_gradients.append(flat_theta_new - flat_theta_prev)
        # flat_theta_new = flat_theta_prev + np.nanmean(step_gradients, axis = 0)
        surrogate_loss, kl_divergence = loss_function(flat_theta_new)
        print('surrogate loss', surrogate_loss, 'kl divergence', kl_divergence)
        return surrogate_loss, kl_divergence

    def train_actor_PPO(
            self,
            on_policy_state,
            on_policy_action,
            on_policy_advantage,
            old_dist_mean,
            old_dist_logstd):
        on_policy_advantage = np.squeeze(on_policy_advantage)
        feed_dict = {self.state_input['actor']: on_policy_state,
                     self.advantage_input: on_policy_advantage,
                     self.action_input['actor']: on_policy_action,
                     self.old_dist_mean: old_dist_mean,  # [:,np.newaxis],
                     self.old_dist_logstd: old_dist_logstd,  # [:,np.newaxis]
                     }
        self.sess.run(self.PPO_train_op, feed_dict=feed_dict)
        surrogate_loss, kl_divergence = self.sess.run(
            [self.surr_loss, self.kl], feed_dict=feed_dict)

        return surrogate_loss, kl_divergence

    def check_network_update(self):
        diff = self.sess.run(self.actor_var_diff)
        print('old and new network parameter diff:', diff)
        return

    def update_old_policy(self):
        self.sess.run(self.hard_copy['actor'])
        # self.sess.run(self.soft_update['actor'])
        return

    def update_critic(self):
        self.sess.run(self.soft_update['target_critic'])
        return

    def train_critic_V(self, state_batch, y_batch):  # train_num=1
        loss = 0
        kl = 0
        train_num = self.opt_method['critic']['train-num']
        for _ in range(train_num):
            loss,  _ = self.sess.run([self.vloss, self.V_optimizer], feed_dict={
                self.yV_input: y_batch,
                self.state_input['critic']: state_batch,
            })
            # print(kl)
        # self.update_critic()
        return loss

    def action_mean(self, state):
        if state.ndim < 2:  # no batch
            state = state[np.newaxis, :]
            return self.sess.run(self.action_mu, feed_dict={
                self.state_input['actor']: state,
            })[0]
        else:
            state_batch = state
            return self.sess.run(self.action_pi, feed_dict={
                self.state_input['actor']: state_batch,
            })

    def action(self, state):
        if state.ndim < 2:  # no batch
            state = state[np.newaxis, :]
            return self.sess.run(self.action_pi, feed_dict={
                self.state_input['actor']: state,
            })[0]
        else:
            state_batch = state
            return self.sess.run(self.action_mu, feed_dict={
                self.state_input['actor']: state_batch,
            })

    def get_action(self, state):
        return(self.actor.get_action(state))

    def logstd(self):
        return self.sess.run(self.action_logstd, feed_dict={})

    def set_logstd(self, logstd):
        if logstd.ndim < 2:
            logstd = logstd[np.newaxis, :]
        self.sess.run(self.set_logstd_op, feed_dict={self.logstd_input: logstd})
        # self.sess.run([self.set_logstd_op,self.set_old_logstd_op], feed_dict={self.logstd_input:logstd})

    def get_actor_info(self, state):
        if state.ndim < 2:  # no batch

            state = state[np.newaxis, :]
            mu = self.sess.run(self.action_mu, feed_dict={
                    self.state_input['actor']: state,
            })[0]
            logstd = self.sess.run(self.action_logstd)[0]
            return mu, logstd
        else:  # batch
            state_batch = state
            mu = self.sess.run(self.action_mu, feed_dict={
                    self.state_input['actor']: state_batch,
            })
            logstd = np.ones((np.shape(state_batch)[0], self.action_dim))
            logstd = logstd*self.sess.run(self.action_logstd)
            return mu, logstd

    def get_old_actor_info(self, state):
        if state.ndim < 2:  # no batch

            state = state[np.newaxis, :]
            mu = self.sess.run(self.old_action_mu, feed_dict={
                    self.state_input['actor']: state,
            })[0]
            logstd = self.sess.run(self.old_action_logstd)[0]
            return mu, logstd
        else:  # batch
            state_batch = state
            mu = self.sess.run(self.old_action_mu, feed_dict={
                    self.state_input['actor']: state_batch,
            })
            logstd = np.ones((np.shape(state_batch)[0], self.action_dim))
            logstd = logstd*self.sess.run(self.old_action_logstd)
            return mu, logstd

    def V(self, state):
        if state.ndim < 2:  # no batch
            state = state[np.newaxis, :]  # add new axis
            return self.sess.run(self.V_output, feed_dict={
                self.state_input['critic']: state,
            })[0]
        else:
            state_batch = state
            return self.sess.run(self.V_output, feed_dict={
                self.state_input['critic']: state_batch,
            })

    def load_network(self, dir_path):
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(dir_path + '/saved_networks')
        print(dir_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:" + checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    def save_network(self, time_step, dir_path):
        print('save network...' + str(time_step))
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        self.saver.save(
            self.sess,
            dir_path +
            '/saved_networks/' +
            'network')  # , global_step = time_step)

    def save_actor_variable(self, time_step, dir_path):
        var_dict = {}
        for var in self.actor_trainable_vars:
            var_array = self.sess.run(var)
            var_dict[var.name] = var_array
            #print(var.name + '  shape: ')
            # print(np.shape(var_array))
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
        for var in self.actor_trainable_vars:
            for key in var_dict:
                if key in var.name:  # if variable name contains similar strings
                    self.sess.run(tf.assign(var, var_dict[var.name]))
                    print(var.name + ' transfered')

        self.update_old_policy()

    def save_critic_variable(self, time_step, dir_path):
        var_dict = {}
        for var in self.critic_trainable_vars:
            var_array = self.sess.run(var)
            var_dict[var.name] = var_array
            #print(var.name + '  shape: ')
            # print(np.shape(var_array))
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
        for var in self.critic_trainable_vars:
            for key in var_dict:
                if key in var.name:  # if variable name contains similar strings
                    self.sess.run(tf.assign(var, var_dict[var.name]))
                    print(var.name + ' transfered')

        # self.update_critic()
