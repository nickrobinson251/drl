import numpy as np
import tensorflow as tf
import pickle

class BatchNormallize:
    def __init__(self, sess, size, config = None):
        self.size = size
        self.mean = tf.Variable(tf.zeros([1,self.size], tf.float32))#(1,39)
        self.std = tf.Variable(tf.ones([1,self.size], tf.float32))
        self.mean_ = tf.Variable(tf.zeros([1,self.size], tf.float32))
        self.std_ = tf.Variable(tf.ones([1,self.size], tf.float32))

        self.gamma = 0.9 #momentum

        self.x = tf.placeholder("float", [None,self.size])
        self.running_mean, self.running_std = tf.nn.moments(self.x,[0])#(39,)
        #print(self.running_mean.get_shape())
        #print(self.mean.get_shape())
        #initialize
        self.sess = sess
        self.create_operation()
        self.sess.run(tf.global_variables_initializer())

    def update(self,x):
        self.sess.run([self.running_mean,self.running_std],feed_dict={
			self.x:x})
        self.sess.run([self.update_mean_, self.update_std_])
        #self.sess.run([self.ema_mean_update,self.ema_std_update])
        self.sess.run([self.update_mean,self.update_std])

    def create_operation(self):
        self.normalize_op = tf.divide(tf.subtract(self.x,self.mean),self.std)
        self.denormalize_op = tf.add(tf.multiply(self.x,self.std),self.mean)
        self.update_mean_ = self.mean_.assign(tf.reshape(self.running_mean,[1,self.size]))
        self.update_std_ = self.mean_.assign(tf.reshape(self.running_std,[1,self.size]))
        self.update_mean = tf.assign(self.mean, self.gamma*self.mean + (1-self.gamma)*self.mean_)
        self.update_std = tf.assign(self.mean, self.gamma*self.std + (1 - self.gamma)*self.std_)
        #ema=tf.train.ExponentialMovingAverage(decay = self.gamma)
        #self.ema_mean_update = ema.apply(self.mean_)
        #self.ema_std_update = ema.apply(self.std_)
        #self.ema_mean = ema.average(self.mean_)
        #self.ema_std = ema.average(self.std_)
        #self.update_mean = self.mean.assign(self.ema_mean)
        #self.update_std = self.std.assign(self.ema_std)

        #tf.groups()
    def normalize(self,x):
        y = self.sess.run(self.normalize_op,feed_dict={self.x:x})[0]
        return y
        #(x - self.mean) / self.std

    def denormalize(self, x):
        y = self.sess.run(self.denormalize_op,feed_dict={self.x:x})[0]
        return y
        # x * self.std + self.mean

    def save_normalization(self, dir):
        m,v = self.sess.run([self.mean, self.std])
        output = open(dir + '/normalize.obj', 'wb')
        pickle.dump([m,v], output)
        output.close()

    def load_normalization(self, dir):
        pkl_file = open(dir + '/normalize.obj', 'rb')
        m,v = pickle.load(pkl_file)
        pkl_file.close()
        op = [
            self.mean.assign(m),
            self.std.assign(v)
            ]
        self.sess.run(op)

class OnlineNormalize:
    def __init__(self, sess, size, length):
        self.size = size
        self.mean = tf.Variable(tf.zeros([1,self.size], tf.float32))
        self.std = tf.Variable(tf.ones([1,self.size], tf.float32))
        self.M = tf.Variable(tf.constant(0.0,shape = [1,size]))
        self.delta1 = tf.Variable(tf.constant(0.0,shape = [1,size]))
        self.delta2 = tf.Variable(tf.constant(0.0,shape = [1,size]))

        self.x = tf.placeholder("float", [1,self.size])
        self.n = 0.0
        self.n_max = length

        #initialize
        self.sess = sess
        self.create_operation()
        self.sess.run(tf.global_variables_initializer())

    def update(self,x):
        self.n += 1
        self.n = np.minimum(self.n, self.n_max)
        self.sess.run(self.update_delta1, feed_dict = {self.x:x}) #delta with history mean
        self.sess.run(self.update_mean)
        self.sess.run(self.update_delta2, feed_dict = {self.x:x}) #delta with new mean
        self.sess.run(self.update_M)
        self.sess.run(self.update_std)

    def create_operation(self):
        self.cal_mean = tf.add(self.mean, tf.divide(self.delta1,self.n))
        self.update_mean = self.mean.assign(self.cal_mean)
        self.cal_delta = tf.subtract(self.x, self.mean)
        self.update_delta1 = self.delta1.assign(self.cal_delta)
        self.update_delta2 = self.delta2.assign(self.cal_delta)
        self.cal_M = tf.add(self.M, tf.multiply(self.delta1,self.delta2))
        self.update_M = self.M.assign(self.cal_M)
        self.update_std = self.std.assign(tf.sqrt(tf.divide(self.M,self.n)))
        self.normalize_op = tf.divide(tf.subtract(self.x,self.mean),self.std)
        self.denormalize_op = tf.add(tf.multiply(self.x,self.std),self.mean)

    def normalize(self,x):
        y = self.sess.run(self.normalize_op,feed_dict={self.x:x})
        return y
        #(x - self.mean) / self.std

    def denormalize(self, x):
        y = self.sess.run(self.denormalize_op,feed_dict={self.x:x})
        return y
        # x * self.std + self.mean

    def save_normalization(self, dir):
        m,v = self.sess.run([self.mean, self.std])
        output = open(dir + '/normalize.obj', 'wb')
        pickle.dump([m,v], output)
        output.close()

    def load_normalization(self, dir):
        pkl_file = open(dir + '/normalize.obj', 'rb')
        m,v = pickle.load(pkl_file)
        pkl_file.close()
        op = [
            self.mean.assign(m),
            self.std.assign(v)
            ]
        self.sess.run(op)
