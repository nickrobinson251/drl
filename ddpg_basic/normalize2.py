import numpy as np
import tensorflow as tf
import pickle

class BatchNormallize:
    def __init__(self, size, length):
        self.size = size
        self.mean = np.zeros((1,self.size))
        self.std = np.ones((1,self.size))
        self.mean_ = np.zeros((1,self.size))
        self.std_ = np.ones((1,self.size))

        self.gamma = 0.99 #momentum
        self.init = False
        self.n_max = length
        #initialize

    def update(self,x):
        if self.init == False:#upon first call
            self.mean = np.zeros((1, self.size))
            self.std = np.ones((1, self.size))
            #self.mean = np.mean(x, axis=0)
            #self.std = np.std(x, axis=0)
        else:
            self.mean_ = np.mean(x, axis = 0)
            self.std_ = np.std(x, axis = 0)
            self.mean = self.gamma * self.mean + (1 - self.gamma) * self.mean_
            self.std = self.gamma * self.std + (1 - self.gamma) * self.std_
        self.init = True

    def normalize(self,x):
        y = (x - self.mean) / self.std
        return y

    def denormalize(self, x):
        y = x * self.std + self.mean
        return y

    def save_normalization(self, dir):
        m,v = self.mean, self.std
        output = open(dir + '/BatchNormalize.obj', 'wb')
        pickle.dump([m,v], output)
        output.close()

    def load_normalization(self, dir):
        self.init = True
        pkl_file = open(dir + '/BatchNormalize.obj', 'rb')
        m,v = pickle.load(pkl_file)
        pkl_file.close()
        self.mean = m
        self.std = v

    def print_normalization(self):
        print('mean')
        print(self.mean)
        print('standard deviation')
        print(self.std)

class OnlineNormalize:
    def __init__(self, size, length):
        self.size = size
        self.mean = np.zeros((1, self.size))
        self.std = np.ones((1, self.size))
        self.M = np.zeros((1, self.size))
        self.delta1 = np.zeros((1, self.size))
        self.delta2 = np.zeros((1, self.size))

        self.n = 0.0
        self.n_max = length

        #initialize

    def update(self, x):
        self.n += 1
        self.n = np.minimum(self.n, self.n_max)
        delta1 = x - self.mean
        self.mean = self.mean + delta1/self.n
        delta2 = x-self.mean
        self.M = self.M + delta1*delta2
        self.std = np.sqrt(self.M/self.n)

    def normalize(self,x):
        y = (x - self.mean) / self.std
        return y

    def denormalize(self, x):
        y = x * self.std + self.mean
        return y

    def save_normalization(self, dir):
        m,v = self.mean, self.std
        output = open(dir + '/OnlineNormalize.obj', 'wb')
        pickle.dump([m,v], output)
        output.close()

    def load_normalization(self, dir):
        pkl_file = open(dir + '/OnlineNormalize.obj', 'rb')
        m,v = pickle.load(pkl_file)
        pkl_file.close()
        self.mean = m
        self.std = v

    def print_normalization(self):
        print('mean')
        print(self.mean)
        print('standard deviation')
        print(self.std)
