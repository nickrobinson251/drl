from datetime import datetime
import tensorflow as tf
import tensorflow.contrib as tc
import math
import numpy as np
import pickle

from parameter_noise import AdaptiveParamNoise


class ActorNetwork:
    """docstring for ActorNetwork"""

    def __init__(self, sess, state_dim, action_dim, config):
        self.param_noise = AdaptiveParamNoise(
            config['param-noise-settings'][0],
            config['param-noise-settings'][1],
            config['param-noise-settings'][2])
        self.layer1_size = config['actor-layer-size'][0]
        self.layer2_size = config['actor-layer-size'][1]
        self.learning_rate = config['actor-lr']
        self.tau = config['tau']
        self.is_param_noise = config['param-noise']
        self.is_layer_norm = config['actor-layer-norm']
        self.is_observation_norm = config['actor-observation-norm']

        self.activation_fn = config['actor-activation-fn']

        #self.param_noise_stddev = tf.placeholder(tf.float32, shape=(), name='param_noise_stddev')

        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        # create actor network
        self.actor_network = {}
        self.state_input, \
            self.action_output, \
            self.actor_network['vars'], \
            self.actor_network['trainable_vars'], \
            self.actor_network['perturbable_vars'] = self.create_network(state_dim, action_dim, 'actor_network')

        # create target actor network
        self.target_actor_network = {}
        self.target_state_input, \
            self.target_action_output, \
            self.target_actor_network['vars'], \
            self.target_actor_network['trainable_vars'], \
            self.target_actor_network['perturbable_vars'] = self.create_network(state_dim, action_dim, 'target_actor_network')

        # create perturbed actor network
        self.perturbed_actor_network = {}
        self.perturbed_state_input, \
            self.perturbed_action_output, \
            self.perturbed_actor_network['vars'], \
            self.perturbed_actor_network['trainable_vars'], \
            self.perturbed_actor_network['perturbable_vars'] = self.create_network(state_dim, action_dim, 'perturbed_actor_network')

        # define training rules
        self.create_training_method()
        self.setup_target_network_updates()
        self.sess.run(tf.global_variables_initializer())

        self.sess.run(self.target_init_updates)
        self.param_noise_stddev = self.setup_param_noise()

        # self.load_network()

    def create_training_method(self):
        self.q_gradient_input = tf.placeholder("float", [None, self.action_dim])
        self.parameters_gradients = tf.gradients(
            self.action_output, self.actor_network['trainable_vars'], -self.q_gradient_input)
        # self.parameters_gradients, _ =
        # tf.clip_by_global_norm(self.parameters_gradients, 1.0) #clipping
        # gradient

        # zip back to gradient value tuples
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(
            zip(self.parameters_gradients, self.actor_network['trainable_vars']))

    def create_network(self, state_dim, action_dim, name):
        layer1_size = self.layer1_size
        layer2_size = self.layer1_size

        with tf.variable_scope(name) as scope:
            state_input = tf.placeholder("float", [None, state_dim])
            W1 = self.variable([state_dim, layer1_size], state_dim)
            b1 = self.variable([layer1_size], state_dim)
            W2 = self.variable([layer1_size, layer2_size], layer1_size)
            b2 = self.variable([layer2_size], layer1_size)
            W3 = tf.Variable(tf.random_uniform(
                [layer2_size, action_dim], -3e-3, 3e-3))
            b3 = tf.Variable(tf.random_uniform([action_dim], -3e-3, 3e-3))
            if self.is_layer_norm:
                layer1 = tf.matmul(state_input, W1) + b1
                layer1_norm = tc.layers.layer_norm(
                    layer1, center=True, scale=True)  # , activation_fn=tf.nn.relu)
                #layer1_norm = tf.nn.relu(layer1_norm)
                layer1_norm = self.add_activation_fn(
                    layer1_norm, self.activation_fn)
                layer2 = tf.matmul(layer1_norm, W2) + b2
                layer2_norm = tc.layers.layer_norm(
                    layer2, center=True, scale=True)  # , activation_fn=tf.nn.relu)
                #layer2_norm = tf.nn.relu(layer2_norm)
                layer2_norm = self.add_activation_fn(
                    layer2_norm, self.activation_fn)
                action_output = tf.identity(tf.matmul(layer2_norm, W3) + b3)
            else:
                layer1 = tf.matmul(state_input, W1) + b1
                layer1 = self.add_activation_fn(
                    layer1, self.activation_fn)  # tf.nn.relu(layer1)
                layer2 = tf.matmul(layer1, W2) + b2
                layer2 = self.add_activation_fn(
                    layer2, self.activation_fn)  # tf.nn.relu(layer2)
                action_output = tf.identity(tf.matmul(layer2, W3) + b3)
            #action_output = action_output + tf.stop_gradient(ref_action)
        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        trainable_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        perturbable_vars = [var for var in trainable_vars
                            if 'LayerNorm' not in var.name]

        return state_input, action_output, vars, trainable_vars, perturbable_vars

    def update_target(self):
        self.sess.run(self.target_soft_updates)

    def train(self, q_gradient_batch, state_batch):
        self.sess.run(self.optimizer, feed_dict={
            self.q_gradient_input: q_gradient_batch,
            self.state_input: state_batch
        })

    def actions(self, state_batch):
        return self.sess.run(self.action_output, feed_dict={
            self.state_input: state_batch
        })

    def action(self, state):
        return self.sess.run(self.action_output, feed_dict={
            self.state_input: [state]
        })[0]

    def action_noise(self, state):
        return self.sess.run(self.perturbed_action_output, feed_dict={
            self.perturbed_state_input: [state]
        })[0]

    def actions_target(self, state_batch):
        return self.sess.run(self.target_action_output, feed_dict={
            self.target_state_input: state_batch
        })

    def get_target_updates(self, vars, target_vars, tau):
        soft_updates = []
        init_updates = []
        assert len(vars) == len(target_vars)
        for var, target_var in zip(vars, target_vars):
            init_updates.append(tf.assign(target_var, var))
            soft_updates.append(
                tf.assign(
                    target_var,
                    (1. - tau) * target_var + tau * var))
        assert len(init_updates) == len(vars)
        assert len(soft_updates) == len(vars)
        return tf.group(*init_updates), tf.group(*soft_updates)

    def setup_target_network_updates(self):
        actor_init_updates, actor_soft_updates = self.get_target_updates(
            self.actor_network['vars'], self.target_actor_network['vars'], self.tau)
        self.target_init_updates = actor_init_updates
        self.target_soft_updates = actor_soft_updates

    def get_perturbed_actor_updates(
            self,
            actor,
            perturbed_actor,
            param_noise_stddev):
        assert len(actor['vars']) == len(perturbed_actor['vars'])
        assert len(
            actor['perturbable_vars']) == len(
            perturbed_actor['perturbable_vars'])

        updates = []
        seed = int(datetime.now().strftime('%S%f'))
        for var, perturbed_var in zip(actor['vars'], perturbed_actor['vars']):
            if var in actor['perturbable_vars']:
                updates.append(
                    tf.assign(
                        perturbed_var,
                        var +
                        tf.random_normal(
                            tf.shape(var),
                            mean=0.,
                            stddev=param_noise_stddev,
                            seed=seed)))  # set seed to avoid repeating random
            else:
                updates.append(tf.assign(perturbed_var, var))
        assert len(updates) == len(actor['vars'])
        return tf.group(*updates)

    def setup_param_noise(self):
        param_noise_stddev = tf.placeholder(
            tf.float32, shape=(), name='param_noise_stddev')
        # Configure perturbed actor.
        perturbed_actor_init_updates, perturbed_actor_soft_updates = self.get_target_updates(
            self.actor_network['vars'], self.perturbed_actor_network['vars'], self.tau)
        self.sess.run(perturbed_actor_init_updates)

        self.perturb_policy_ops = self.get_perturbed_actor_updates(
            self.actor_network, self.perturbed_actor_network, param_noise_stddev)
        self.sess.run(self.perturb_policy_ops, feed_dict={
            param_noise_stddev: self.param_noise.current_stddev,
        })

        # Configure separate copy for stddev adoption.
        self.perturb_adaptive_policy_ops = self.get_perturbed_actor_updates(
            self.actor_network, self.perturbed_actor_network, param_noise_stddev)
        self.adaptive_policy_distance = tf.sqrt(
            tf.reduce_mean(
                tf.square(
                    self.action_output -
                    self.perturbed_action_output)))
        self.sess.run(self.perturb_adaptive_policy_ops, feed_dict={
            param_noise_stddev: self.param_noise.current_stddev,
        })
        return param_noise_stddev

    def adapt_param_noise(self, state_batch):
        # Perturb a separate copy of the policy to adjust the scale for the next
        # "real" perturbation.
        self.sess.run(self.perturb_adaptive_policy_ops, feed_dict={
            self.param_noise_stddev: self.param_noise.current_stddev,
        })
        # measure the distance
        distance = self.sess.run(self.adaptive_policy_distance, feed_dict={
            self.state_input: state_batch,
            self.perturbed_state_input: state_batch,
            self.param_noise_stddev: self.param_noise.current_stddev,
        })

        #mean_distance = mpi_mean(distance)
        self.param_noise.adapt(distance)
        return distance

    def perturb_policy(self):
        self.sess.run(self.perturb_policy_ops, feed_dict={
            self.param_noise_stddev: self.param_noise.current_stddev,
        })

    def batch_norm_layer(self, x, training_phase, scope_bn, activation=None):
        return tf.cond(
            training_phase,
            lambda: tc.layers.batch_norm(
                x,
                activation_fn=activation,
                center=True,
                scale=True,
                updates_collections=None,
                is_training=True,
                reuse=None,
                scope=scope_bn,
                decay=0.9,
                epsilon=1e-5),
            lambda: tc.layers.batch_norm(
                x,
                activation_fn=activation,
                center=True,
                scale=True,
                updates_collections=None,
                is_training=False,
                reuse=True,
                scope=scope_bn,
                decay=0.9,
                epsilon=1e-5))

    # f fan-in size
    def variable(self, shape, f):
        #seed = int(datetime.now().strftime('%S%f'))
        # return tf.Variable(tf.random_uniform(shape, -1 / math.sqrt(f), 1 /
        # math.sqrt(f), seed=seed))
        return tf.Variable(
            tf.random_uniform(
                shape, -1 / math.sqrt(f), 1 / math.sqrt(f)))
        # return tf.Variable(tf.random_normal(shape))

    def add_layer(
            self,
            inputs,
            in_size,
            out_size,
            normalization=None,
            activation_fn=None,
            dropout=None):
        #Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        #biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        #biases = tf.Variable(tf.random_normal([1, out_size]))
        Weights = self.variable([in_size, out_size], in_size)
        biases = self.variable([out_size], in_size)
        layer = tf.matmul(inputs, Weights) + biases

        if normalization:  # Layer Norm
            layer_norm = tc.layers.layer_norm(layer, center=True, scale=True)
        else:
            layer_norm = layer

        if activation_fn == 'elu':
            outputs = tf.nn.elu(layer_norm)
        elif activation_fn == 'relu':
            outputs = tf.nn.relu(layer_norm)
        elif activation_fn == 'leaky_relu':
            outputs = self.leaky_relu(layer_norm)
        else:
            outputs = tf.identity(layer_norm)
        return outputs

    def leaky_relu(self, x):
        alpha = 0.2
        return tf.maximum(x, alpha * x)

    def add_activation_fn(self, x, activation_fn='None'):
        if activation_fn == 'elu':
            outputs = tf.nn.elu(x)
        elif activation_fn == 'relu':
            outputs = tf.nn.relu(x)
        elif activation_fn == 'leaky_relu':
            outputs = self.leaky_relu(x)
        elif activation_fn == 'tanh':
            outputs = tf.nn.tanh(x)
        else:
            outputs = tf.identity(x)
        return outputs

    def load_network(self, dir_path):
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(
            dir_path + '/saved_actor_networks')
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:" + checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    def save_network(self, time_step, dir_path):
        print('save actor-network...' + str(time_step))
        self.saver.save(
            self.sess,
            dir_path +
            '/saved_actor_networks/' +
            'actor-network')  # , global_step = time_step)

    def save_variable(self, time_step, dir_path):
        var_dict = {}
        for var in self.actor_network['trainable_vars']:
            var_array = self.sess.run(var)
            var_dict[var.name] = var_array
            print(var.name + '  shape: ')
            print(np.shape(var_array))
        output = open(dir_path + '/actor_variable.obj', 'wb')
        pickle.dump(var_dict, output)
        output.close()

    def load_variable(self, dir_path):
        var = {}
        pkl_file = open(dir_path + '/actor_variable.obj', 'rb')
        var_temp = pickle.load(pkl_file)
        var.update(var_temp)
        pkl_file.close()
        return var

    def transfer_variable(self, var_dict):
        for var in self.actor_network['trainable_vars']:
            for key in var_dict:
                if key in var.name:  # if variable name contains similar strings
                    self.sess.run(tf.assign(var, var_dict[var.name]))
                    print(var.name + ' transfered')
        # copy network weights to target network
        self.sess.run(self.target_init_updates)
        # if var.name in var_dict:  # check if variable exist
        #    self.sess.run(tf.assign(var, var_dict[var.name]))
        #    print(var.name + ' transfered')
