import numpy as np
import pickle
from pprint import pprint


ACTION_DIM = 11
NUM_CONTROLLED_JOINTS = 11
REPLAY_START_SIZE = 10000
TAU = 0.001  # weight given to the target model

ACTION_BOUNDS = {
    "leftAnklePitch": [-0.93, 0.65],
    "leftAnkleRoll": [-0.4, 0.4],
    "leftHipPitch": [-2.42, 1.619],
    "leftHipRoll": [-0.467, 0.5515],
    "leftHipYaw": [-0.4141, 1.1],
    "leftKneePitch": [-0.083, 2.057],
    "rightAnklePitch": [-0.93, 0.65],
    "rightAnkleRoll": [-0.4, 0.4],
    "rightHipPitch": [-2.42, 1.619],
    "rightHipRoll": [-0.5515, 0.467],
    "rightHipYaw": [-1.1, 0.4141],
    "rightKneePitch": [-0.083, 2.057],
    "torsoPitch": [-0.13, 0.67],
    "torsoRoll": [-0.23, 0.255],
}

ACTION_BOUNDS_NORMALISED = {
    "leftAnklePitch": [-1.1772, 0.8228],
    "leftAnkleRoll": [-1.0, 1.0],
    "leftHipPitch": [-1.1983, 0.8017],
    "leftHipRoll": [-0.917, 1.083],
    "leftHipYaw": [-0.2735, 0.7265],
    "leftKneePitch": [-0.0776, 1.9224],
    "rightAnklePitch": [-1.1772, 0.8228],
    "rightAnkleRoll": [-1.0, 1.0],
    "rightHipPitch": [-1.1983, 0.8017],
    "rightHipRoll": [-1.083, 0.917],
    "rightHipYaw": [-0.7265, 0.2735],
    "rightKneePitch": [-0.0776, 1.9224],
    "torsoPitch": [-0.325, 1.675],
    "torsoRoll": [-0.4742, 0.5258],
}


def record_configuration(config, dir):
    """Write configuration to dr/'configuration.txt'."""
    txt_file = open(dir + "/configuration.txt", "w")
    pprint(config, stream=txt_file)
    txt_file.close()


def save_configuration(config, dir):
    """Write cnfiguration dict to dir/'configuration.pickle'."""
    output = open(dir + "/configuration.pickle", "wb")
    pickle.dump(config, output)
    output.close()


def update_configuration(config, dir):
    """Update curretn config with dict from file dir/'configuration.pickle'."""
    pkl_file = open(dir + "/configuration.pickle", "rb")
    conf_temp = pickle.load(pkl_file)
    config.update(conf_temp)
    pkl_file.close()


def load_configuration(dir):
    """Load configuration dict from file dir/'configuration.pickle'."""
    pkl_file = open(dir + "/configuration.pickle", "rb")
    config = pickle.load(pkl_file)
    pkl_file.close()
    return config


config = {}
config['env'] = {
    "HLC-frequency": 25,
    "LLC-frequency": 500,
    "gamma": 0.99,
    "id": "HumanoidBalanceFilter-v0",
    "use-joint-interpolation": True,
    "physics-frequency": 1000,
    "render": False,
    "render-eval": False,
    "state-dim": 60,
    "use-bullet-default-PD": False,
}
config['env']["controlled-joints"] = [  # default to controlling 7 pitch joints
    "leftAnklePitch",
    "leftHipPitch",
    "leftKneePitch",
    "rightAnklePitch",
    "rightHipPitch",
    "rightKneePitch",
    "torsoPitch"]
if NUM_CONTROLLED_JOINTS >= 11:
    config['env']["controlled-joints"].extend([
        "leftAnkleRoll",
        "leftHipPitch",
        "rightAnkleRoll",
        "rightHipRoll"])
if NUM_CONTROLLED_JOINTS >= 12:
    config['env']["controlled-joints"].append("torsoRoll")
if NUM_CONTROLLED_JOINTS >= 13:
    config['env']["controlled-joints"].append("rightHipYaw")

config['env']["Kp"] = {
    "leftAnklePitch": 2000,  # -0.71
    "leftAnkleRoll": 1000,  # -0.71
    "leftElbowPitch": 200,
    "leftHipPitch": 2000,  # -0.491
    "leftHipRoll": 1500,  # -0.49
    "leftHipYaw": 500,
    "leftKneePitch": 2000,  # 1.205
    "leftShoulderPitch": 700,
    "leftShoulderRoll": 1500,
    "leftShoulderYaw": 200,
    "rightAnklePitch": 2000,  # -0.71
    "rightAnkleRoll": 1000,  # -0.71
    "rightElbowPitch": 200,
    "rightHipPitch": 2000,  # -0.49
    "rightHipRoll": 1500,  # -0.49
    "rightHipYaw": 500,
    "rightKneePitch": 2000,  # 1.205
    "rightShoulderPitch": 700,
    "rightShoulderRoll": 1500,
    "rightShoulderYaw": 200,
    "torsoPitch": 3000,
    "torsoRoll": 4500,
    "torsoYaw": 4500,
}

config['env']["Kd"] = {
    "leftAnklePitch": 120,  # -0.71
    "leftAnkleRoll": 100,  # -0.71
    "leftElbowPitch": 5,
    "leftHipPitch": 180,  # -0.49
    "leftHipRoll": 150,  # -0.49
    "leftHipYaw": 50,
    "leftKneePitch": 120,  # 1.205
    "leftShoulderPitch": 10,
    "leftShoulderRoll": 30,
    "leftShoulderYaw": 2,
    "rightAnklePitch": 120,  # -0.71
    "rightAnkleRoll": 100,  # -0.71
    "rightElbowPitch": 5,
    "rightHipPitch": 180,  # -0.49
    "rightHipRoll": 150,  # -0.49
    "rightHipYaw": 50,
    "rightKneePitch": 120,  # 1.205
    "rightShoulderPitch": 10,
    "rightShoulderRoll": 30,
    "rightShoulderYaw": 2,
    "torsoPitch": 300,
    "torsoRoll": 30,
    "torsoYaw": 30,
}

config['train'] = {
    "batch-size": 256,
    "num-episodes": 50000,
    "epoch-step-num": 50000,
    "max-time": 16,  # seconds,
    "normalize-observations": False,
    "normalize-returns": False,
    "popart": False,
    "reward-scale": 0.1,  # 1.0
    "num-rollout-steps": 1,
    "num-test-": 1,
    "num-episode-steps": 250000000,
}

config['test'] = {
    "max-time": 30,  # seconds,
    "num-episodes": 1,
    "num-episode-steps": 250000000,
}

config["replay"] = {
    "buffer-size": 2000000,
    "record-start-size": REPLAY_START_SIZE*1.01,
    "start-size": REPLAY_START_SIZE,
    "use-prioritized-experience": True
}

config["noise"] = {
    "OU-settings": [0.0, 0.15, 0.3],  # mu, theta, sigma
    "param-noise-settings": [0.05, 0.2, 1.01],
    "use-OU": True,
    "use-normalization": True,
    "use-param-noise": False
}

config["critic"] = {
    "activation-fn": "relu",
    "dropout-rate": 0.5,
    "l2-reg": 1e-2,
    "layer-size": [400, 400],
    "lr": 1e-3,
    "tau": TAU,
    "use-dropout": False,
    "use-layer-norm": False,
    "use-observation-norm": False  # use batch norm to normalize observations
}

config["actor"] = {
    "action-dim": ACTION_DIM,
    "activation-fn": "relu",
    "dropout-rate": 0.5,
    "l2-reg": 1e-2,
    "layer-size": [400, 400],
    "lr": 1e-4,
    "output-bound-method": "grad-invert",  # "tanh"
    "tau": TAU,
    "use-dropout": False,
    "use-layer-norm": False,
    "use-observation-norm": False
}
config["actor"]["action-joints"] = [  # default to 11 pitch and roll joints
        "leftAnklePitch",
        "leftAnkleRoll",
        "leftHipPitch",
        "leftHipRoll",
        "leftKneePitch",
        "rightAnklePitch",
        "rightAnkleRoll",
        "rightHipPitch",
        "rightHipRoll",
        "rightKneePitch",
        "torsoPitch"]
if ACTION_DIM >= 12:
    config["actor"]["action-joints"].append("torsoRoll")
if ACTION_DIM >= 13:
    config["actor"]["action-joints"].append("rightHipYaw")

# bound actions. transpose so  array[0] are lowwer bound, [1] are upper bounds
config["actor"]['action-bounds'] = np.array(
    [ACTION_BOUNDS[joint] for joint in config["actor"]["action-joints"]]).T
config["actor"]['action-bounds-normalised'] = np.array(
    [ACTION_BOUNDS_NORMALISED[joint]
        for joint in config["actor"]["action-joints"]]).T

# clip exploration range for joints with large limits
config["actor"]["action-scale"] = 0.5 * np.clip(
    config["actor"]['action-bounds'][1] - config["actor"]['action-bounds'][0],
    0,
    2.09)
config['actor']['logstd-initial'] = np.log(config["actor"]["action-scale"] / 2)
config['actor']['logstd-bounds'] = np.zeros_like(
    config["actor"]['action-bounds'])
config['actor']['logstd-bounds'][0] = np.log(config["actor"]["action-scale"]/10)
config['actor']['logstd-bounds'][1] = np.log(config["actor"]["action-scale"]/2)
