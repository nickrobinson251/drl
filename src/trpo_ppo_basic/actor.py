import numpy as np
import tensorflow as tf

class Actor(object):

    def __init__(self, net, sess, config):


        self.net = net
        self.sess = sess
        self.config = config

        self.input_ph = self.net.input
        self.output_net = self.net.output
        self.var_list = []
        self.trainable_var_list = []

    def get_action(self, inputs):
        raise NotImplementedError

class GaussianActor(Actor):

    def __init__(self, net, sess, config):
        super(GaussianActor, self).__init__(net, sess, config)

        logstd_init = np.array(self.config.conf['actor-logstd-initial'], ndmin=2).astype(np.float32)
        #logstd_init = (-0.6*np.ones([1, self.net.output_dim])).astype(np.float32)
        #self.action_logstd = tf.Variable( (np.zeros([1, self.net.output_dim])).astype(np.float32) ,name = 'weights_logstd')
        #self.action_logstd = tf.Variable(logstd_init,name='weights_logstd')
        self.action_logstd = tf.get_variable(initializer=logstd_init, dtype=tf.float32, trainable=self.net.trainable, name='weights_logstd')
        # self.action_logstd = tf.tile(self.action_logstd_param, tf.stack( [tf.shape(self.output_net)[0] ,1] ) )
        self.action_std = tf.exp(self.action_logstd)
        self.logstd_bound = self.config.conf['actor-logstd-bounds']
        self.std_bound = np.exp(self.logstd_bound)
        # if isinstance(self.std_bound[0], float):
        #     min_std = np.ones((1, self.net.output_dim)).astype(np.float32) * self.config
        # else:
        print(self.std_bound[0])
        min_std = np.array(self.std_bound[0]).astype(np.float32)#.resize(1, self.net.output_dim)
        min_std.resize(1, self.net.output_dim)
        print(min_std)

        # if isinstance(self.std_bound[1], float):
        #     max_std = np.ones((1, self.net.output_dim)).astype(np.float32) * self.config
        # else:
        max_std = np.array(self.std_bound[1]).astype(np.float32)#.resize(1, self.net.output_dim)
        max_std.resize(1, self.net.output_dim)
        print(max_std)

        print(self.net.output_dim)

        self.action_std = tf.clip_by_value(self.action_std, min_std, max_std)
        self.action_mean = self.output_net
        self.action_norm_dist = tf.distributions.Normal(loc=self.action_mean, scale=self.action_std)  # normal distribution
        self.action = tf.squeeze(self.action_norm_dist.sample(1), axis=0)
        # self.action_std = tf.maximum(self.action_std, self.pms.min_std)
        # self.action_std = tf.minimum(self.action_std, self.pms.max_std)

        self.var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name)
        self.trainable_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=tf.get_variable_scope().name)
        # self.trainable_var_list = [v for v in tf.trainable_variables() if v.name.startswith('agent/'+self.net.name)]
        # self.var_list = [v for v in tf.trainable_variables() if v.name.startswith('agent/'+self.net.name)]
    def get_action(self, inputs):

        if len(inputs.shape) < 2:
            inputs = inputs[np.newaxis,:]

        feed_dict = {self.input_ph: inputs}
        a_mean, a_std = self.sess.run([self.output_net, self.action_std], feed_dict = feed_dict)
        a_mean, a_std = map(np.squeeze, [a_mean, a_std])
        a_logstd = np.log(a_std)
        # a_mean = np.tanh(a_mean) * self.pms.max_action
        action = self.sess.run(self.action, feed_dict)
        # action = np.squeeze(action)#remove single dimensional entries
        # print(a_mean, a_std)
        action = np.random.normal( a_mean,a_std )
        return action, dict(mean = a_mean, std = a_std,logstd = a_logstd)
