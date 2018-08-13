import pickle
import numpy as np

class Configuration:
    def __init__(self):
        self.conf={}
        self.conf['env-id']      = 'Gazebo'
        self.conf['render-eval'] = False

        self.conf['state-dim']  = 12
        self.conf['action-dim'] =  6
        self.conf['action-bounds']              =  np.zeros((2,self.conf['action-dim'])) # 
        self.conf['actor-logstd-initial']       =  np.ones((1, self.conf['action-dim']))*np.log(2.0)
        self.conf['actor-logstd-bounds']        =  np.ones((2,self.conf['action-dim']))
        self.conf['actor-logstd-bounds'][0]    *=  np.log(0.01)
        self.conf['actor-logstd-bounds'][1]    *=  np.log(2.0)
        self.conf['actor-output-bounds']        =  np.ones((2,self.conf['action-dim']))
        self.conf['actor-output-bounds'][0][:]  = -np.ones((self.conf['action-dim'],)) #STD
        self.conf['actor-output-bounds'][1][:]  =  np.ones((self.conf['action-dim'],))   #STD
        self.conf['actor-scale'] = 1 #unused basically Move to val_drl.py
        # self.conf['actor-output-bounds'] = self.conf['action-bounds']

        self.conf['critic-l2-reg']           = 1e-2
        self.conf['critic-lr']               = 1e-3
        # self.conf['critic-layer-size']       = [400,400]#[400,400]
        # self.conf['critic-activation-fn']    = ['tanh', 'tanh', 'None']

        self.conf['critic-layer-size']       = [100,50,25]#[400,400]
        self.conf['critic-activation-fn']    = ['relu', 'relu', 'relu', 'None']

        self.conf['critic-opt-method']       = dict([
                                            ('name', 'none'),
                                            ('train-num', 1),#10
                                            ('critic-lr', 3e-4),#3e-3#3e-4
                                            ('critic-l2-reg', 1e-3),#1e-5#1e-2
                                            ('critic-l2', False),#False
                                            ])
        self.conf['use-critic']                 = True
        self.conf['critic-batch-size']          = 128
        self.conf['critic-iteration']           = 40

        self.conf['actor-l2-reg']               = 1e-2
        self.conf['actor-lr']                   = 1e-4
        self.conf['actor-layer-size']           = [100,50,25]#[400,400]
        self.conf['actor-activation-fn']        = ['relu', 'relu', 'relu', 'None']
        # self.conf['actor-layer-size']           = [200,200]#[400,400]
        # self.conf['actor-activation-fn'] = ['relu', 'relu', 'None']
        # self.conf['actor-activation-fn']        = ['tanh', 'tanh', 'tanh']
        self.conf['actor-opt-method']           = dict([
                                                ('epsilon', 0.003),#0.1
                                                ('line-search', True), #True #TODO check seems to have huge effect
                                                ('conjugate-gradient-iteration', 30),
                                                ('line-search-backtrack', 10),
                                                ('cg-damping', 0.1),
                                                ])
        self.conf['center-advantage'] = False

        self.conf['IPG-method']     = dict(useCV = True, vu=0.2, DPG_flag=True)
        self.conf['QProp-method']   = ['adaptive','aggressive','conservative', 'none'][2]
        self.conf['PPO-method']     = dict([
                                    ('epsilon',0.2),
                                    ('actor-lr',3e-4),#1e-5 can not be too large
                                    ('actor-batch-size',256),#32#64
                                    ('epoch', 40)#too large will prevent PPO from learning, somewhat causes the ratio to exceed the clipping range
                                    ])
        self.conf['max-path-num']   = 20
        self.conf['max-path-step']  = 4096#512#1024

        self.conf['off-policy-update-num']          = 4  #8
        self.conf['off-policy-critic-update-num']   = 4  # 8
        self.conf['off-policy-actor-update-num']    = 0  # 8

        self.conf['loss-entropy-coeff']         = 0.01
        self.conf['loss-symmetry-coeff']        = 0.1
        self.conf['loss-output-bound-coeff']    = 0.01

        self.conf['GAE']    = True
        self.conf['lambda'] = 0.95#0.95#0.95#0.9
        self.conf['tau']    = 0.001
        self.conf['gamma']  = 0.95# gamma = (N-1)/N
        self.conf['render'] = False

        self.conf['record-start-size']  = 0

        # self.conf['reward-scale'] = 0.1#1.0
        self.conf['epoch-num']          = 10000
        self.conf['epoch-step-num']     = 5000000
        self.conf['total-step-num']     = 2500000000
        self.conf['max-train-time']     = 180 #second
        self.conf['max-test-time']      = 60 #second
        self.conf['test-num']           = 1
        self.conf['rollout-step-num']   = 1
        self.conf['train-step-num']     = 1
        self.conf['max-episode-num']    = 10000000000000#1000
        self.conf['max-step-num']       = 2500000000#2500000

    def save_configuration(self,dir):
        # write python dict to a file
        output = open(dir + '/configuration.obj', 'wb')
        pickle.dump(self.conf, output)
        output.close()

    def load_configuration(self,dir):
        # write python dict to a file
        pkl_file = open(dir + '/configuration.obj', 'rb')
        conf_temp = pickle.load(pkl_file)
        self.conf.update(conf_temp)
        pkl_file.close()

    def record_configuration(self,dir):
        # write python dict to a file
        output = open(dir + '/readme.txt', 'w')
        for key in self.conf:
            output.write("{}: {}\n".format(key,self.conf[key]))

    def print_configuration(self):
        for key in self.conf:
            print(key + ': ' + str(self.conf[key]))
