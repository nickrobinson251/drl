import tensorflow as tf
import numpy as np


class NeuralNetwork(object):
    def __init__(self,
                 sess,
                 input_dim,
                 output_dim,
                 layer_dim,
                 config,
                 name=None,
                 **kwargs):
        activation_functions = {
            'tanh': tf.nn.tanh,
            'relu': tf.nn.relu,
            'None': lambda x: x}

        self.config = config
        self.sess = sess
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_dim = layer_dim
        self.name = name
        self.all_layer_dim = np.concatenate(
            [[self.input_dim], layer_dim, [self.output_dim]],
            axis=0).astype(int)
        self.if_bias = kwargs.get(
            'if_bias', ([True] * len(layer_dim)) + [False])
        self.activation_names = kwargs.get(
            'activations', (['tanh'] * len(layer_dim)) + ['None'])
        self.activations = [activation_functions[name]
                            for name in self.activation_names]
        self.initialize_weight = kwargs.get('init_weight', None)
        self.initialize_bias = kwargs.get('init_bias', None)
        self.trainable = kwargs.get('trainable', True)
        self.reusable = kwargs.get('reusable', False)
        num_layers = len(self.layer_dim) + 1
        if len(self.if_bias) == 1:
            self.if_bias *= num_layers
        if len(self.activations) == 1:
            self.activations *= num_layers

        assert len(self.activations) == num_layers, "incorrect number activatns"
        assert len(self.if_bias) == num_layers, "incorrect number of bias terms"

    def build(self):
        raise NotImplementedError


class FullyConnectedNetwork(NeuralNetwork):
    def __init__(self,
                 sess,
                 input_dim,
                 output_dim,
                 layer_dim,
                 config,
                 name=None,
                 **kwargs):
        super(FullyConnectedNetwork, self).__init__(
            sess,
            input_dim,
            output_dim,
            layer_dim,
            config,
            name,
            **kwargs)

        # with tf.variable_scope(self.name):
        self.input = kwargs.get(
            'input_tf',
            tf.placeholder(tf.float32, [None, self.input_dim], name='input'))
        assert (self.input.get_shape()[-1].value == input_dim)
        self.output = self.build(self.input, self.name)

    def build(self, input_tf, name):
        net = input_tf
        weights = []
        if np.any(self.if_bias):
            biases = []

        for i, (dim_1, dim_2) in enumerate(
                zip(self.all_layer_dim[:-1], self.all_layer_dim[1:])):

            if self.initialize_weight:
                stddev = self.initialize_weight[i]
            else:
                stddev = 0.1
            weight = tf.get_variable(
                initializer=tf.truncated_normal([dim_1, dim_2], stddev=stddev),
                trainable=self.trainable,
                dtype=tf.float32,
                name='theta_{}'.format(i))
            weights.append(weight)

            if self.if_bias[i]:
                # zero initialization for bias
                if self.initialize_bias[i]:
                    initializer = tf.truncated_normal(
                        [dim_2], stddev=self.initialize_bias[i])
                else:
                    initializer = np.zeros([1, dim_2])
                bias = tf.get_variable(
                    initializer=initializer,
                    trainable=self.trainable,
                    dtype=tf.float32,
                    name='bias_{}'.format(i))
                biases.append(bias)

                net = self.activation_fns_call[i](
                    tf.matmul(net, weights[i]) + biases[-1])
            else:
                net = self.activation_fns_call[i](tf.matmul(net, weights[i]))

        return net
