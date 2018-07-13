actor-dropout: False
joint-interpolation: True
Physics-frequency: 1000
actor-logstd-bounds: [[-3.21887582 -2.97740132 -2.25856821 -2.25856821 -2.53830743 -3.21887582
  -2.97740132 -2.25856821 -2.25856821 -2.53830743 -3.21887582]
 [-1.60943791 -1.3679634  -0.6491303  -0.6491303  -0.92886951 -1.60943791
  -1.3679634  -0.6491303  -0.6491303  -0.92886951 -1.60943791]]
critic-layer-size: [400, 400]
max-train-time: 16
Kp: {'rightAnkleRoll': 1000, 'rightShoulderPitch': 700, 'rightShoulderRoll': 1500, 'rightElbowPitch': 200, 'leftHipPitch': 2000, 'rightHipRoll': 1500, 'leftHipYaw': 500, 'leftShoulderRoll': 1500, 'leftHipRoll': 1500, 'rightHipYaw': 500, 'leftKneePitch': 2000, 'torsoPitch': 3000, 'torsoRoll': 4500, 'torsoYaw': 4500, 'rightKneePitch': 2000, 'rightHipPitch': 2000, 'leftShoulderYaw': 200, 'rightShoulderYaw': 200, 'leftAnkleRoll': 1000, 'leftAnklePitch': 2000, 'leftShoulderPitch': 700, 'rightAnklePitch': 2000, 'leftElbowPitch': 200}
controlled-joints: ['torsoPitch', 'rightHipRoll', 'rightHipPitch', 'rightKneePitch', 'rightAnklePitch', 'rightAnkleRoll', 'leftHipRoll', 'leftHipPitch', 'leftKneePitch', 'leftAnklePitch', 'leftAnkleRoll']
popart: False
action-bounds: [[-0.13   -0.5515 -2.42   -0.083  -0.93   -0.4    -0.467  -2.42   -0.083
  -0.93   -0.4   ]
 [ 0.67    0.467   1.619   2.057   0.65    0.4     0.5515  1.619   2.057
   0.65    0.4   ]]
prioritized-exp-replay: True
actor-logstd-initial: [[-1.60943791 -1.3679634  -0.6491303  -0.6491303  -0.92886951 -1.60943791
  -1.3679634  -0.6491303  -0.6491303  -0.92886951 -1.60943791]]
actor-output-bound: [[-0.13   -0.5515 -2.42   -0.083  -0.93   -0.4    -0.467  -2.42   -0.083
  -0.93   -0.4   ]
 [ 0.67    0.467   1.619   2.057   0.65    0.4     0.5515  1.619   2.057
   0.65    0.4   ]]
critic-dropout: False
render: False
state-dim: 51
critic-lr: 0.001
actor-action-joints: ['torsoPitch', 'rightHipRoll', 'rightHipPitch', 'rightKneePitch', 'rightAnklePitch', 'rightAnkleRoll', 'leftHipRoll', 'leftHipPitch', 'leftKneePitch', 'leftAnklePitch', 'leftAnkleRoll']
param-noise: False
critic-observation-norm: False
test-num: 1
record-start-size: 10100.0
batch-size: 256
critic-layer-norm: False
actor-output-bound-method: grad-invert
action-dim: 11
LLC-frequency: 500
normalized-action-bounds: [[-0.325  -1.083  -1.1983 -0.0776 -1.1772 -1.     -0.917  -1.1983 -0.0776
  -1.1772 -1.    ]
 [ 1.675   0.917   0.8017  1.9224  0.8228  1.      1.083   0.8017  1.9224
   0.8228  1.    ]]
action-scale: [0.4     0.50925 1.045   1.045   0.79    0.4     0.50925 1.045   1.045
 0.79    0.4    ]
render-eval: False
actor-l2-reg: 0.01
tau: 0.001
OU-noise-settings: [0.0, 0.15, 0.3]
rollout-step-num: 1
replay-start-size: 10000
param-noise-settings: [0.05, 0.2, 1.01]
gamma: 0.95
bullet-default-PD: False
Kd: {'rightAnkleRoll': 100, 'rightShoulderPitch': 10, 'rightShoulderRoll': 30, 'rightElbowPitch': 5, 'leftHipPitch': 180, 'rightHipRoll': 150, 'leftHipYaw': 50, 'leftShoulderRoll': 30, 'leftHipRoll': 150, 'rightHipYaw': 50, 'leftKneePitch': 120, 'torsoPitch': 300, 'torsoRoll': 30, 'torsoYaw': 30, 'rightKneePitch': 120, 'rightHipPitch': 180, 'leftShoulderYaw': 2, 'rightShoulderYaw': 2, 'leftAnkleRoll': 100, 'leftAnklePitch': 120, 'leftShoulderPitch': 10, 'rightAnklePitch': 120, 'leftElbowPitch': 5}
actor-activation-fn: relu
actor-layer-norm: False
total-step-num: 250000000
OU-noise: True
replay-buffer-size: 2000000
HLC-frequency: 25
train-step-num: 1
normalize-returns: False
normalize-observations: False
actor-observation-norm: False
critic-dropout-rate: 0.5
epoch-num: 50000
critic-activation-fn: relu
actor-dropout-rate: 0.5
max-test-time: 30
noise-normalization: True
epoch-step-num: 50000
actor-layer-size: [400, 400]
reward-scale: 0.1
actor-lr: 0.0001
env-id: HumanoidBalanceFilter-v0
critic-l2-reg: 0.01
