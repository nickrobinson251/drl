import tensorflow as tf
import tensorflow.contrib as tc
import math
import numpy as np
import pickle


class CriticNetwork:
    """docstring for CriticNetwork"""

    def __init__(self, sess, state_dim, action_dim, config):
        self.layer1_size = config['critic-layer-size'][0]
        self.layer2_size = config['critic-layer-size'][1]
        self.learning_rate = config['critic-lr']
        self.tau = config['tau']
        self.l2 = config['critic-l2-reg']
        self.is_prioritized_replay = config['prioritized-exp-replay']
        self.is_layer_norm = config['critic-layer-norm']
        self.is_observation_norm = config['critic-observation-norm']

        self.activation_fn = config['critic-activation-fn']

        self.time_step = 0
        self.sess = sess
        # create q network
        self.critic_network = {}
        self.state_input, \
            self.action_input, \
            self.q_value_output, \
            self.critic_network['vars'], \
            self.critic_network['trainable_vars'], \
            self.critic_network['perturbable_vars'] = self.create_q_network(state_dim, action_dim, 'critic_network')

        # create target q network (the same structure with q network)
        self.target_critic_network = {}
        self.target_state_input, \
            self.target_action_input, \
            self.target_q_value_output, \
            self.target_critic_network['vars'], \
            self.target_critic_network['trainable_vars'], \
            self.target_critic_network['perturbable_vars'] = self.create_q_network(state_dim, action_dim, 'target_critic_network')

        self.create_training_method()
        self.setup_target_network_updates()
        self.sess.run(tf.global_variables_initializer())

        self.sess.run(self.target_init_updates)
        # self.load_network()

    def create_training_method(self):
        # Define training optimizer
        self.y_input = tf.placeholder("float", [None, 1])
        weight_decay = tf.add_n(
            [self.l2 * tf.nn.l2_loss(var)
             for var in self.critic_network['trainable_vars']])
        if self.is_prioritized_replay:
            # importance sampling for experience replay
            self.ISWeights = tf.placeholder(
                tf.float32, [None, 1],
                name='IS_weights')
            self.abs_errors = tf.reduce_sum(
                tf.abs(self.y_input - self.q_value_output),
                axis=1)  # TD_error for updating Sumtree
            self.loss = tf.reduce_mean(
                self.ISWeights *
                tf.square(
                    self.y_input -
                    self.q_value_output))
            self.cost = self.loss + weight_decay
            #self.cost = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.y_input, self.q_value_output)) + weight_decay
        else:
            self.loss = tf.reduce_mean(
                tf.square(
                    self.y_input -
                    self.q_value_output))
            self.cost = self.loss + weight_decay

        self.optimizer = tf.train.AdamOptimizer(
            self.learning_rate).minimize(
            self.cost)
        self.action_gradients = tf.gradients(
            self.q_value_output, self.action_input)

    def create_q_network(self, state_dim, action_dim, name):
        # the layer size could be changed
        layer1_size = self.layer1_size
        layer2_size = self.layer2_size
        with tf.variable_scope(name) as scope:
            state_input = tf.placeholder("float", [None, state_dim])
            action_input = tf.placeholder("float", [None, action_dim])

            W1 = self.variable([state_dim, layer1_size], state_dim)
            b1 = self.variable([layer1_size], state_dim)
            W2 = self.variable(
                [layer1_size, layer2_size],
                layer1_size + action_dim)
            W2_action = self.variable(
                [action_dim, layer2_size],
                layer1_size + action_dim)
            b2 = self.variable([layer2_size], layer1_size + action_dim)
            W3 = tf.Variable(tf.random_uniform([layer2_size, 1], -3e-3, 3e-3))
            b3 = tf.Variable(tf.random_uniform([1], -3e-3, 3e-3))

            if self.is_layer_norm:
                layer1 = tf.matmul(state_input, W1) + b1
                # , activation_fn=tf.nn.relu)#normalize layer before activation function?
                layer1_norm = tc.layers.layer_norm(
                    layer1, center=True, scale=True)
                # layer1_norm = tf.nn.relu(layer1_norm)seed =
                # int(datetime.now().strftime('%S%f'))
                layer1_norm = self.add_activation_fn(
                    layer1_norm, self.activation_fn)
                layer2 = (
                    tf.matmul(
                        layer1_norm,
                        W2) +
                    tf.matmul(
                        action_input,
                        W2_action) +
                    b2)
                layer2_norm = tc.layers.layer_norm(
                    layer2, center=True, scale=True)  # , activation_fn=tf.nn.relu)
                #layer2_norm = tf.nn.relu(layer2_norm)
                layer2_norm = self.add_activation_fn(
                    layer2_norm, self.activation_fn)
                q_value_output = tf.identity(tf.matmul(layer2_norm, W3) + b3)
            else:
                layer1 = tf.matmul(state_input, W1) + b1
                layer1 = self.add_activation_fn(
                    layer1, self.activation_fn)  # tf.nn.relu(layer1)
                layer2 = (
                    tf.matmul(
                        layer1,
                        W2) +
                    tf.matmul(
                        action_input,
                        W2_action) +
                    b2)
                layer2 = self.add_activation_fn(
                    layer2, self.activation_fn)  # tf.nn.relu(layer2)
                q_value_output = tf.identity(tf.matmul(layer2, W3) + b3)

        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        trainable_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        perturbable_vars = [var for var in trainable_vars
                            if 'LayerNorm' not in var.name]
        return state_input, action_input, q_value_output, vars, trainable_vars, perturbable_vars

    def update_target(self):
        self.sess.run(self.target_soft_updates)

    def train(self, y_batch, state_batch, action_batch, ISWeights):
        self.time_step += 1
        abs_errors = 0
        loss = 0
        if self.is_prioritized_replay:
            loss, abs_errors, _ = self.sess.run([self.loss, self.abs_errors, self.optimizer], feed_dict={
                self.y_input: y_batch,
                self.state_input: state_batch,
                self.action_input: action_batch,
                self.ISWeights: ISWeights
            })
        else:
            loss, _ = self.sess.run([self.loss, self.optimizer], feed_dict={
                self.y_input: y_batch,
                self.state_input: state_batch,
                self.action_input: action_batch
            })
        return loss, abs_errors

    def gradients(self, state_batch, action_batch):
        return self.sess.run(self.action_gradients, feed_dict={
            self.state_input: state_batch,
            self.action_input: action_batch
        })[0]

    def q_value_target(self, state_batch, action_batch):
        return self.sess.run(self.target_q_value_output, feed_dict={
            self.target_state_input: state_batch,
            self.target_action_input: action_batch
        })

    def q_value(self, state_batch, action_batch):
        return self.sess.run(self.q_value_output, feed_dict={
            self.state_input: state_batch,
            self.action_input: action_batch})

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
            self.critic_network['vars'], self.target_critic_network['vars'], self.tau)
        self.target_init_updates = actor_init_updates
        self.target_soft_updates = actor_soft_updates

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
        # Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        # biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        # biases = tf.Variable(tf.random_normal([1, out_size]))
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
            dir_path + '/saved_critic_networks')
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:"+checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    def save_network(self, time_step, dir_path):
        print('save critic-network...'+str(time_step))
        self.saver.save(
            self.sess,
            dir_path +
            '/saved_critic_networks/' +
            'critic-network')  # , global_step = time_step)
        #self.saver.save(self.sess, 'critic-network')

    def save_variable(self, time_step, dir_path):
        var_dict = {}
        for var in self.critic_network['trainable_vars']:
            var_array = self.sess.run(var)
            var_dict[var.name] = var_array
            print(var.name + '  shape: ')
            print(np.shape(var_array))
        output = open(dir_path + '/critic_variable.obj', 'wb')
        pickle.dump(var_dict, output)
        output.close()

    def load_variable(self, dir_path):
        var = {}
        pkl_file = open(dir_path + '/critic_variable.obj', 'rb')
        var_temp = pickle.load(pkl_file)
        var.update(var_temp)
        pkl_file.close()
        return var

    def transfer_variable(self, var_dict):
        for var in self.critic_network['trainable_vars']:
            for key in var_dict:
                if key in var.name:  # if variable name contains similar strings
                    self.sess.run(tf.assign(var, var_dict[var.name]))
                    print(var.name + ' transfered')
            # if var.name in var_dict:  # check if variable exist
            #    self.sess.run(tf.assign(var, var_dict[var.name]))
            #    print(var.name + ' transfered')
        # copy network weights to target network
        self.sess.run(self.target_init_updates)
