env-id: HumanoidBalanceFilter-v0
render-eval: False
joint-interpolation: True
controlled-joints: ['torsoPitch', 'rightHipRoll', 'rightHipPitch', 'rightKneePitch', 'rightAnklePitch', 'rightAnkleRoll', 'leftHipRoll', 'leftHipPitch', 'leftKneePitch', 'leftAnklePitch', 'leftAnkleRoll']
actor-action-joints: ['torsoPitch', 'rightHipRoll', 'rightHipPitch', 'rightKneePitch', 'rightAnklePitch', 'rightAnkleRoll', 'leftHipRoll', 'leftHipPitch', 'leftKneePitch', 'leftAnklePitch', 'leftAnkleRoll']
state-dim: 51
action-dim: 11
action-bounds: [[-0.13   -0.5515 -2.42   -0.083  -0.93   -0.4    -0.467  -2.42   -0.083
  -0.93   -0.4   ]
 [ 0.67    0.467   1.619   2.057   0.65    0.4     0.5515  1.619   2.057
   0.65    0.4   ]]
action-scale: [0.4     0.50925 1.045   1.045   0.79    0.4     0.50925 1.045   1.045
 0.79    0.4    ]
normalized-action-bounds: [[-0.325  -1.083  -1.1983 -0.0776 -1.1772 -1.     -0.917  -1.1983 -0.0776
  -1.1772 -1.    ]
 [ 1.675   0.917   0.8017  1.9224  0.8228  1.      1.083   0.8017  1.9224
   0.8228  1.    ]]
actor-logstd-initial: [[-1.60943791 -1.3679634  -0.6491303  -0.6491303  -0.92886951 -1.60943791
  -1.3679634  -0.6491303  -0.6491303  -0.92886951 -1.60943791]]
actor-logstd-bounds: [[-3.21887582 -2.97740132 -2.25856821 -2.25856821 -2.53830743 -3.21887582
  -2.97740132 -2.25856821 -2.25856821 -2.53830743 -3.21887582]
 [-1.60943791 -1.3679634  -0.6491303  -0.6491303  -0.92886951 -1.60943791
  -1.3679634  -0.6491303  -0.6491303  -0.92886951 -1.60943791]]
Physics-frequency: 1000
LLC-frequency: 500
HLC-frequency: 25
bullet-default-PD: False
Kp: {'torsoYaw': 4500, 'torsoPitch': 3000, 'torsoRoll': 4500, 'rightHipYaw': 500, 'rightHipRoll': 1500, 'rightHipPitch': 2000, 'rightKneePitch': 2000, 'rightAnklePitch': 2000, 'rightAnkleRoll': 1000, 'leftHipYaw': 500, 'leftHipRoll': 1500, 'leftHipPitch': 2000, 'leftKneePitch': 2000, 'leftAnklePitch': 2000, 'leftAnkleRoll': 1000, 'rightShoulderPitch': 700, 'rightShoulderRoll': 1500, 'rightShoulderYaw': 200, 'rightElbowPitch': 200, 'leftShoulderPitch': 700, 'leftShoulderRoll': 1500, 'leftShoulderYaw': 200, 'leftElbowPitch': 200}
Kd: {'torsoYaw': 30, 'torsoPitch': 300, 'torsoRoll': 30, 'rightHipYaw': 50, 'rightHipRoll': 150, 'rightHipPitch': 180, 'rightKneePitch': 120, 'rightAnklePitch': 120, 'rightAnkleRoll': 100, 'leftHipYaw': 50, 'leftHipRoll': 150, 'leftHipPitch': 180, 'leftKneePitch': 120, 'leftAnklePitch': 120, 'leftAnkleRoll': 100, 'rightShoulderPitch': 10, 'rightShoulderRoll': 30, 'rightShoulderYaw': 2, 'rightElbowPitch': 5, 'leftShoulderPitch': 10, 'leftShoulderRoll': 30, 'leftShoulderYaw': 2, 'leftElbowPitch': 5}
batch-size: 256
critic-layer-norm: False
critic-observation-norm: False
critic-l2-reg: 0.01
critic-lr: 0.001
critic-layer-size: [400, 400]
critic-activation-fn: relu
critic-dropout: False
critic-dropout-rate: 0.5
actor-layer-norm: False
actor-observation-norm: False
actor-l2-reg: 0.01
actor-lr: 0.0001
actor-layer-size: [400, 400]
actor-activation-fn: relu
actor-dropout: False
actor-dropout-rate: 0.5
actor-output-bound-method: grad-invert
actor-output-bound: [[-0.13   -0.5515 -2.42   -0.083  -0.93   -0.4    -0.467  -2.42   -0.083
  -0.93   -0.4   ]
 [ 0.67    0.467   1.619   2.057   0.65    0.4     0.5515  1.619   2.057
   0.65    0.4   ]]
tau: 0.001
gamma: 0.99
popart: False
render: False
normalize-returns: False
normalize-observations: False
prioritized-exp-replay: True
replay-buffer-size: 2000000
replay-start-size: 10000
record-start-size: 10100.0
reward-scale: 0.1
epoch-num: 50000
epoch-step-num: 50000
total-step-num: 250000000
max-train-time: 16
max-test-time: 30
test-num: 1
rollout-step-num: 1
train-step-num: 1
param-noise: False
param-noise-settings: [0.05, 0.2, 1.01]
OU-noise: True
OU-noise-settings: [0.0, 0.15, 0.3]
noise-normalization: True
