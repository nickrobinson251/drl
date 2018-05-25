import numpy as np
import pickle
from pprint import pprint


ACTION_DIM = 11
NUM_CONTROLLED_JOINTS = 11
REPLAY_START_SIZE = 10000

ACTION_BOUNDS = {
    "torsoPitch": [-0.13, 0.67],
    "torsoRoll": [-0.23, 0.255],
    "rightHipYaw": [-1.1, 0.4141],
    "rightHipRoll": [-0.5515, 0.467],
    "rightHipPitch": [-2.42, 1.619],
    "rightKneePitch": [-0.083, 2.057],
    "rightAnklePitch": [-0.93, 0.65],
    "rightAnkleRoll": [-0.4, 0.4],
    "leftHipYaw": [-0.4141, 1.1],
    "leftHipRoll": [-0.467, 0.5515],
    "leftHipPitch": [-2.42, 1.619],
    "leftKneePitch": [-0.083, 2.057],
    "leftAnklePitch": [-0.93, 0.65],
    "leftAnkleRoll": [-0.4, 0.4],
}

ACTION_BOUNDS_NORMALISED = {
    "torsoPitch": [-0.325, 1.675],
    "torsoRoll": [-0.4742, 0.5258],
    "rightHipYaw": [-0.7265, 0.2735],
    "rightHipRoll": [-1.083, 0.917],
    "rightHipPitch": [-1.1983, 0.8017],
    "rightKneePitch": [-0.0776, 1.9224],
    "rightAnklePitch": [-1.1772, 0.8228],
    "rightAnkleRoll": [-1.0, 1.0],
    "leftHipYaw": [-0.2735, 0.7265],
    "leftHipRoll": [-0.917, 1.083],
    "leftHipPitch": [-1.1983, 0.8017],
    "leftKneePitch": [-0.0776, 1.9224],
    "leftAnklePitch": [-1.1772, 0.8228],
    "leftAnkleRoll": [-1.0, 1.0],
}

config = {}
config['env'] = {
    "id": "HumanoidBalanceFilter-v0",
    "render-eval": False,
    "render": False,
    "joint-interpolation": True,
    "physics-frequency": 1000,
    "LLC-frequency": 500,
    "HLC-frequency": 25,
    "use-bullet-default-PD": False,
    "state-dim": 60
}

config['training'] = {
    "action-dim": ACTION_DIM,
    "batch-size": 256,
    "tau": 0.001,
    "gamma": 0.99,
    "popart": False,
    "normalize-returns": False,
    "normalize-observations": False,
    "reward-scale": 0.1,  # 1.0
    "epoch-num": 50000,
    "epoch-step-num": 50000,
    "total-step-num": 250000000,
    "max-train-time": 16,  # seconds,
    "max-test-time": 30,  # seconds,
    "test-num": 1,
    "rollout-step-num": 1,
    "train-step-num": 1
}

config["replay"] = {
    "use-prioritized-experience": True,
    "buffer-size": 2000000,
    "start-size": REPLAY_START_SIZE,
    "record-start-size": REPLAY_START_SIZE*1.01
}

config["noise"] = {
    "use-param-noise": False,
    "param-noise-settings": [0.05, 0.2, 1.01],
    "use-OU": True,
    "OU-noise-settings": [0.0, 0.15, 0.3],  # mu, theta, sigma
    "use-normalization": True
}

config["critic"] = {
    "use-layer-norm": False,
    "use-observation-norm": False,  # use batch norm to normalize observations
    "use-dropout": False,
    "dropout-rate": 0.5,
    "l2-reg": 1e-2,
    "lr": 1e-3,
    "layer-size": [400, 400],
    "activation-fn": "relu"
}

config["actor"] = {
    "use-layer-norm": False,
    "use-observation-norm": False,
    "use-dropout": False,
    "dropout-rate": 0.5,
    "l2-reg": 1e-2,
    "lr": 1e-4,
    "layer-size": [400, 400],
    "activation-fn": "relu",
    "output-bound-method": "grad-invert",  # "tanh"
    # "output-bound": config["action-bounds"]
}
config["actor"]["action-joints"] = [  # default to 11 pitch and roll joints
        "torsoPitch",
        "rightHipPitch",
        "rightHipRoll",
        "leftHipPitch",
        "leftHipRoll",
        "rightKneePitch",
        "leftKneePitch",
        "rightAnklePitch",
        "rightAnkleRoll",
        "leftAnklePitch",
        "leftAnkleRoll"]
if ACTION_DIM >= 12:
    config["actor"]["action-joints"].append("torsoRoll")
if ACTION_DIM >= 13:
    config["actor"]["action-joints"].append("rightHipYaw")

# bound actions. transpose so  array[0] are lowwer bound, [1] are upper bounds
config['action-bounds'] = np.array(
    [ACTION_BOUNDS[joint] for joint in config["actor"]["action-joints"]]).T
config['action-bounds-normalised'] = np.array(
    [ACTION_BOUNDS_NORMALISED[joint]
        for joint in config["actor"]["action-joints"]]).T

# clip exploration range for joints with large limits
config["action-scale"] = 0.5 * np.clip(
    config['action-bounds'][1] - config['action-bounds'][0], 0, 2.09)
config['actor']['logstd-initial'] = np.log(config["action-scale"]/2)
config['actor']['logstd-bounds'] = np.zeros_like(config['action-bounds'])
config['actor']['logstd-bounds'][0] = np.log(config["action-scale"]/10)
config['actor']['logstd-bounds'][1] = np.log(config["action-scale"]/2)

config["controlled-joints"] = [  # default to controlling 7 pitch joints
    "torsoPitch",
    "rightHipPitch",
    "leftHipPitch",
    "rightKneePitch",
    "leftKneePitch",
    "rightAnklePitch",
    "leftAnklePitch"]
if NUM_CONTROLLED_JOINTS >= 11:
    config["controlled-joints"].extend([
        "rightHipRoll",
        "leftHipPitch",
        "rightAnkleRoll",
        "leftAnkleRoll"])
if NUM_CONTROLLED_JOINTS >= 12:
    config["controlled-joints"].append("torsoRoll")
if NUM_CONTROLLED_JOINTS >= 13:
    config["controlled-joints"].append("rightHipYaw")

config["Kp"] = {
    "torsoYaw": 4500,
    "torsoPitch": 3000,
    "torsoRoll": 4500,
    "rightHipYaw": 500,
    "rightHipRoll": 1500,  # -0.49
    "rightHipPitch": 2000,  # -0.49
    "rightKneePitch": 2000,  # 1.205
    "rightAnklePitch": 2000,  # -0.71
    "rightAnkleRoll": 1000,  # -0.71
    "leftHipYaw": 500,
    "leftHipRoll": 1500,  # -0.49
    "leftHipPitch": 2000,  # -0.491
    "leftKneePitch": 2000,  # 1.205
    "leftAnklePitch": 2000,  # -0.71
    "leftAnkleRoll": 1000,  # -0.71
    "rightShoulderPitch": 700,
    "rightShoulderRoll": 1500,
    "rightShoulderYaw": 200,
    "rightElbowPitch": 200,
    "leftShoulderPitch": 700,
    "leftShoulderRoll": 1500,
    "leftShoulderYaw": 200,
    "leftElbowPitch": 200,
}

config["Kd"] = {
    "torsoYaw": 30,
    "torsoPitch": 300,
    "torsoRoll": 30,
    "rightHipYaw": 50,
    "rightHipRoll": 150,  # -0.49
    "rightHipPitch": 180,  # -0.49
    "rightKneePitch": 120,  # 1.205
    "rightAnklePitch": 120,  # -0.71
    "rightAnkleRoll": 100,  # -0.71
    "leftHipYaw": 50,
    "leftHipRoll": 150,  # -0.49
    "leftHipPitch": 180,  # -0.49
    "leftKneePitch": 120,  # 1.205
    "leftAnklePitch": 120,  # -0.71
    "leftAnkleRoll": 100,  # -0.71
    "rightShoulderPitch": 10,
    "rightShoulderRoll": 30,
    "rightShoulderYaw": 2,
    "rightElbowPitch": 5,
    "leftShoulderPitch": 10,
    "leftShoulderRoll": 30,
    "leftShoulderYaw": 2,
    "leftElbowPitch": 5,
}


def save_configuration(config, dir):
    """Write cnfiguration dict to dir/'configuration.pickle'."""
    output = open(dir + "/configuration.pickle", "wb")
    pickle.dump(config, output)
    output.close()


def load_configuration(config, dir):
    """Load configuration dict from file dir/'configuration.pickle'."""
    pkl_file = open(dir + "/configuration.pickle", "rb")
    conf_temp = pickle.load(pkl_file)
    config.update(conf_temp)
    pkl_file.close()


def record_configuration(config, dir):
    """Write configuration to dr/'configuration.txt'."""
    txt_file = open(dir + "/configuration.txt", "w")
    pprint(config, stream=txt_file)
    txt_file.close()


def print_configuration(config):
    pprint(config)
