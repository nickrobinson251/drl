import gym
from gym import spaces
from gym.utils import seeding
import math
import numpy as np
import os
import pybullet as p

from valkyrie.envs.filter import FilterClass
from valkyrie.envs.pd_controller import PDController
from valkyrie.envs.sensor_signal_process import calCOP


CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

KP_DEFAULTS = dict([
    ("torsoYaw", 4500),
    ("torsoPitch", 4500),
    ("torsoRoll", 4500),
    ("rightHipYaw", 500),
    ("rightHipRoll", 1000),  # -0.49
    ("rightHipPitch", 2000),  # -0.49
    ("rightKneePitch", 2000),  # 1.205
    ("rightAnklePitch", 3000),  # -0.71
    ("rightAnkleRoll", 300),  # -0.71
    ("leftHipYaw", 500),
    ("leftHipRoll", 1000),  # -0.49
    ("leftHipPitch", 2000),  # -0.49
    ("leftKneePitch", 2000),  # 1.205
    ("leftAnklePitch", 3000),  # -0.71
    ("leftAnkleRoll", 300),  # -0.71
    ("rightShoulderPitch", 700),
    ("rightShoulderRoll", 1500),
    ("rightShoulderYaw", 200),
    ("rightElbowPitch", 200),
    ("leftShoulderPitch", 700),
    ("leftShoulderRoll", 1500),
    ("leftShoulderYaw", 200),
    ("leftElbowPitch", 200),
])

KD_DEFAULTS = dict([
    ("torsoYaw", 30),
    ("torsoPitch", 30),
    ("torsoRoll", 30),
    ("rightHipYaw", 20),
    ("rightHipRoll", 30),  # -0.49
    ("rightHipPitch", 30),  # -0.49
    ("rightKneePitch", 30),  # 1.205
    ("rightAnklePitch", 3),  # -0.71
    ("rightAnkleRoll", 3),  # -0.71
    ("leftHipYaw", 20),
    ("leftHipRoll", 30),  # -0.49
    ("leftHipPitch", 30),  # -0.49
    ("leftKneePitch", 30),  # 1.205
    ("leftAnklePitch", 3),  # -0.71
    ("leftAnkleRoll", 3),  # -0.71
    ("rightShoulderPitch", 10),
    ("rightShoulderRoll", 30),
    ("rightShoulderYaw", 2),
    ("rightElbowPitch", 5),
    ("leftShoulderPitch", 10),
    ("leftShoulderRoll", 30),
    ("leftShoulderYaw", 2),
    ("leftElbowPitch", 5),
])


class ValkyrieEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 max_time=16,  # in seconds
                 initial_gap_time=0.01,  # in seconds
                 isEnableSelfCollision=True,
                 renders=True,
                 PD_freq=500.0,
                 physics_freq=1000.0,
                 Kp=KP_DEFAULTS,
                 Kd=KD_DEFAULTS,
                 bullet_default_PD=True,
                 logFileName=None,
                 controlled_joints_list=None,
                 ):
        self.bullet_default_PD = bullet_default_PD

        self.jointLowerLimit = []
        self.jointUpperLimit = []

        self._p = p
        self._seed()
        self._envStepCounter = 0
        self._renders = renders
        if logFileName is None:
            self._logFileName = CURRENT_DIR
        else:
            self._logFileName = logFileName

        if self._renders:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        if controlled_joints_list is None:
            self.controlled_joints = ["torsoPitch",
                                      "rightHipPitch",
                                      "rightKneePitch",
                                      "rightAnklePitch",
                                      "rightAnkleRoll",
                                      "leftHipPitch",
                                      "leftKneePitch",
                                      "leftAnklePitch",
                                      "leftAnkleRoll"]
        else:
            self.controlled_joints = controlled_joints_list
        # TODO add control for roll joints
        # self.controlled_joints = ["torsoPitch",
        # "rightHipPitch", "rightHipRoll", "rightKneePitch",
        # "rightAnklePitch", "rightAnkleRoll",
        # "leftHipPitch", "leftHipRoll", "leftKneePitch",i
        # "leftAnklePitch", "leftAnkleRoll",]

        self.nu = len(self.controlled_joints)
        self.r = -1
        self.PD_freq = PD_freq
        self.physics_freq = physics_freq
        self._actionRepeat = int(physics_freq/PD_freq)
        self._dt_physics = (1. / self.physics_freq)
        self._dt_PD = (1. / self.PD_freq)
        self._dt = self._dt_physics  # PD control loop timestep
        self._dt_filter = self._dt_PD  # filter time step
        self.g = 9.81

        self.max_steps = max_time * self.PD_freq  # PD control loop timestep
        # Simulation reset timestep in PD control loop
        self.initial_gap_steps = initial_gap_time * self.PD_freq

        self.u_max = dict([("torsoYaw", 190),
                           ("torsoPitch", 150),
                           ("torsoRoll", 150),
                           ("rightShoulderPitch", 190),
                           ("rightShoulderRoll", 190),
                           ("rightShoulderYaw", 65),
                           ("rightElbowPitch", 65),
                           ("rightForearmYaw", 26),
                           ("rightWristRoll", 14),
                           ("rightWristPitch", 14),
                           ("leftShoulderPitch", 190),
                           ("leftShoulderRoll", 190),
                           ("leftShoulderYaw", 65),
                           ("leftElbowPitch", 65),
                           ("leftForearmYaw", 26),
                           ("leftWristRoll", 14),
                           ("leftWristPitch", 14),
                           ("rightHipYaw", 190),
                           ("rightHipRoll", 350),
                           ("rightHipPitch", 350),
                           ("rightKneePitch", 350),
                           ("rightAnklePitch", 205),
                           ("rightAnkleRoll", 205),
                           ("leftHipYaw", 190),
                           ("leftHipRoll", 350),
                           ("leftHipPitch", 350),
                           ("leftKneePitch", 350),
                           ("leftAnklePitch", 205),
                           ("leftAnkleRoll", 205),
                           ("lowerNeckPitch", 50),
                           ("upperNeckPitch", 50),
                           ("neckYaw", 50)])

        self.v_max = dict([("torsoYaw", 5.89),
                           ("torsoPitch", 9),
                           ("torsoRoll", 9),
                           ("rightShoulderPitch", 5.89),
                           ("rightShoulderRoll", 5.89),
                           ("rightShoulderYaw", 11.5),
                           ("rightElbowPitch", 11.5),
                           ("leftShoulderPitch", 5.89),
                           ("leftShoulderRoll", 5.89),
                           ("leftShoulderYaw", 11.5),
                           ("leftElbowPitch", 11.5),
                           ("rightHipYaw", 5.89),
                           ("rightHipRoll", 7),
                           ("rightHipPitch", 6.11),
                           ("rightKneePitch", 6.11),
                           ("rightAnklePitch", 11),
                           ("rightAnkleRoll", 11),
                           ("leftHipYaw", 5.89),
                           ("leftHipRoll", 7),
                           ("leftHipPitch", 6.11),
                           ("leftKneePitch", 6.11),
                           ("leftAnklePitch", 11),
                           ("leftAnkleRoll", 11),
                           ("lowerNeckPitch", 5),
                           ("upperNeckPitch", 5),
                           ("neckYaw", 5)])
        # nominal joint configuration
        self.q_nom = dict([("torsoYaw", 0.0),
                           ("torsoPitch", 0.0),
                           ("torsoRoll", 0.0),
                           ("lowerNeckPitch", 0.0),
                           ("neckYaw", 0.0),
                           ("upperNeckPitch", 0.0),
                           ("rightShoulderPitch", 0.300196631343),
                           ("rightShoulderRoll", 1.25),
                           ("rightShoulderYaw", 0.0),
                           ("rightElbowPitch", 0.785398163397),
                           # ("rightForearmYaw", 1.571),
                           # ("rightWristRoll", 0.0),self.logId
                           # ("rightWristPitch", 0.0),
                           ("leftShoulderPitch", 0.300196631343),
                           ("leftShoulderRoll", -1.25),
                           ("leftShoulderYaw", 0.0),
                           ("leftElbowPitch", -0.785398163397),
                           # ("leftForearmYaw", 1.571),
                           # ("leftWristRoll", 0.0),
                           # ("leftWristPitch", 0.0),
                           ("rightHipYaw", 0.0),
                           ("rightHipRoll", 0.0),
                           ("rightHipPitch", 0.0),  # -0.49
                           ("rightKneePitch", 0.0),  # 1.205
                           ("rightAnklePitch", 0.0),  # -0.71
                           ("rightAnkleRoll", 0.0),
                           ("leftHipYaw", 0.0),
                           ("leftHipRoll", 0.0),
                           ("leftHipPitch", 0.0),  # -0.49
                           ("leftKneePitch", 0.0),  # 1.205
                           ("leftAnklePitch", 0.0),  # -0.71
                           ("leftAnkleRoll", 0.0)])

        self.Kp = Kp
        self.Kd = Kd

        self.pd_controller = dict()  # TODO Add self defined PD controller
        self.action = dict()
        # Historical joint torque and target joint torque values, for PD control
        self.hist_torque = dict()
        self.hist_tar_torque = dict()
        self.PD_torque_filtered = dict()
        self.PD_torque_unfiltered = dict()
        self.PD_torque_adjusted = dict()

        for key in self.controlled_joints:
            self.hist_torque.update({key: 0.0})
            self.hist_tar_torque.update({key: 0.0})

        self.controllable_joints = [
            "torsoYaw", "torsoPitch", "torsoRoll", "lowerNeckPitch", "neckYaw",
            "upperNeckPitch", "rightShoulderPitch", "rightShoulderRoll",
            "rightShoulderYaw", "rightElbowPitch", "leftShoulderPitch",
            "leftShoulderRoll", "leftShoulderYaw", "leftElbowPitch",
            "rightHipYaw", "rightHipRoll", "rightHipPitch", "rightKneePitch",
            "rightAnklePitch", "rightAnkleRoll", "leftHipYaw", "leftHipRoll",
            "leftHipPitch", "leftKneePitch", "leftAnklePitch", "leftAnkleRoll"]
        self.uncontrolled_joints = [
            a for a in self.controllable_joints
            if a not in self.controlled_joints]
        self.jointIdx = dict()
        self.jointNameIdx = dict()

        # link information
        self.linkMass = dict()
        self.linkCOMPos = dict()
        self.linkCOMVel = dict()
        self.total_mass = 0.0

        self.base_pos_nom = np.array([0, 0, 1.175])  # straight #1.025 bend
        self.base_orn_nom = np.array([0, 0, 0, 1])  # x,y,z,w
        self.q_nom_array = np.array([self.q_nom[i]
                                     for i in self.controllable_joints])

        # global coordinate of COM
        self.COM_pos = np.array([0.0, 0.0, 0.0])
        self.COM_pos_his = np.array([0.0, 0.0, 0.0])
        self.COM_vel = np.array([0.0, 0.0, 0.0])
        # local coordinate of COM w.r.t center of mass of foot link
        # robot operates solely on the sagittal plane, the orientation of global
        # frame and local frame is aligned
        self.COM_pos_local = np.array([0.0, 0.0, 0.0])
        self.COM_pos_local_filter = np.array([0.0, 0.0, 0.0])
        self.support_polygon_center = np.array([[0.0, 0.0, 0.0]])
        self.COM_pos_local_surrogate = np.array([0.0, 0.0, 0.0])
        self.support_polygon_center_surrogate = np.array([[0.0, 0.0, 0.0]])
        self.hull = []  # hull of support polygon
        self.contact_points = []

        # TODO pelvis acceleration
        self.pelvis_acc_gap_step = 10  # 30
        self.pelvis_acc = np.array([0.0, 0.0, 0.0])
        self.pelvis_acc_base = np.array([0.0, 0.0, 0.0])
        self.pelvis_vel_his_array = np.zeros((self.pelvis_acc_gap_step, 3))
        self.pelvis_vel_his = np.array(
            [0.0, 0.0, 0.0])  # history pelvis velocity
        self.pelvis_vel_his_base = np.array([0.0, 0.0, 0.0])

        self.stateNumber = 51  # 60  # 2 * 26 + 2 * 6

        # Setup Simulation
        self._setupSimulation()

        observationDim = self.stateNumber

        observation_high = np.array([np.finfo(np.float32).max] * observationDim)
        self.action_space = spaces.Discrete(self.nu)
        self.observation_space = spaces.Box(-observation_high, observation_high)
        self._observationDim = observationDim
        self._actionDim = len(self.controlled_joints)

        print(
            "observationDim",
            self._observationDim,
            "actionDim",
            self._actionDim)
        self.viewer = None

    def __del__(self):
        p.disconnect()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(
            self,
            Kp=KP_DEFAULTS,
            Kd=KD_DEFAULTS,
            base_pos_nom=None,
            base_orn_nom=None,
            fixed_base=False):
        self.Kp = Kp
        self.Kd = Kd

        self._setupSimulation(base_pos_nom, base_orn_nom, fixed_base)
        self._envStepCounter = 0

        self._observation = self.getExtendedObservation()
        # self._reading = self.getReading()
        return np.array(self._observation)

    def _step(self, action, pelvis_push_force=[0, 0, 0]):
        # print(p.getPhysicsEngineParameters())
        # self.setControl(action)
        # p.applyExternalForce(self.r, -1, forceObj=[pelvis_push_force, 0, 0],
        #                      posObj=[0, 0, 0], flags=p.LINK_FRAME)
        # base_pos, base_quat = p.getBasePositionAndOrientation(self.r)
        # pos = np.array(base_pos)+np.array([0, 0.0035, 0])
        # p.applyExternalForce(self.r, -1, forceObj=[pelvis_push_force, 0, 0],
        #                      posObj=pos, flags=p.WORLD_FRAME)

        applied_torque_dict = dict()
        torque_dict = dict(self.calPDTorque(action))
        applied_torque_dict = dict(torque_dict)
        prev_torque_dict = dict(self.hist_tar_torque)
        self.setControl(action)
        # higher frequency for physics simulation
        for i in range(int(self._actionRepeat)):
            # displacement from COM to the center of the pelvis
            p.applyExternalForce(
                self.r, -1, forceObj=pelvis_push_force, posObj=[0, 0.0035, 0],
                flags=p.LINK_FRAME)
            for key, value in torque_dict.items():
                applied_torque_dict[key] = np.clip(
                    torque_dict[key] + torque_dict[key] - prev_torque_dict[key],
                    -self.u_max[key], self.u_max[key])
                # applied_torque_dict[key] = self.pd_controller[key].u_adj
                # add momentum turn to reduce delay
                self.PD_torque_filtered[key] = self.pd_controller[key].u
                self.PD_torque_unfiltered[key] = self.pd_controller[key].u_raw
                self.PD_torque_adjusted[key] = applied_torque_dict[key]

            self.setDefaultControl(action)
            self.setPDVelocityControl(torque_dict)
            # self.setPDVelocityControl(applied_torque_dict)
            # self.setPDPositionControl(applied_torque_dict, action)
            # self.setPDTorqueControl(torque_dict)
            p.stepSimulation()

        prev_torque_dict.update(torque_dict)
        self.hist_tar_torque.update(prev_torque_dict)

        # update COM information
        self.getLinkCOMPos()
        self.getLinkCOMVel()
        self.calCOMPos()
        self.calCOMVel()
        # TODO calculate pelvis acceleration
        self.calPelvisAcc()
        self.calCOP()
        self.getGroundContactPoint()
        # perform filtering
        self.performFiltering()

        # if self._renders:
        #     time.sleep(self._dt)
        self._observation = self.getExtendedObservation()  # filtered
        self._reading = self.getReading()

        self._envStepCounter += 1
        reward, _ = self._reward()  # balancing
        done = self._termination()

        for key in self.controlled_joints:  # TODO
            joint_state = p.getJointState(self.r, self.jointIdx[key])
            # update history_torque
            self.hist_torque.update({key: joint_state[3]})

        # for key, value in torque_dict.items():
        #   # update history target torque for PD control
        #     self.hist_tar_torque.update({key: value})
        # b1=p.getBasePositionAndOrientation(self.r)
        self.COM_pos_his = np.array(self.COM_pos)
        return np.array(self._observation), reward, done, {}

    def _reward(self):
        base_pos, base_quat = p.getBasePositionAndOrientation(self.r)
        base_orn = p.getEulerFromQuaternion(base_quat)

        chest_link_state = p.getLinkState(
            self.r, self.jointIdx['torsoRoll'],
            computeLinkVelocity=1)

        # Nominal COM x and z position [0.0,1.104]
        # Nominal COM x and z position w.r.t COM of foot [0.0,1.064] #TODO test
        # filtering for reward
        x_pos_err = 0.0 - self.COM_pos_local[0]
        y_pos_err = 0.0 - self.COM_pos_local[1]
        z_pos_err = 1.104 * 1.02 - self.COM_pos_local[2]

        xy_pos_tar = np.array([0.0, 0.0])
        xy_pos = np.array([self.COM_pos_local[0], self.COM_pos_local[1]])
        xy_pos_err = np.linalg.norm(xy_pos-xy_pos_tar)

        x_vel_err = self.targetCOMVel(0.0, 'x') - self.COM_vel[0]
        y_vel_err = self.targetCOMVel(0.0, 'y') - self.COM_vel[1]
        z_vel_err = 0.0 - self.COM_vel[2]

        xy_vel_tar = np.array([self.targetCOMVel(0.0, 'x'),
                               self.targetCOMVel(0.0, 'y')])
        xy_vel = np.array([self.COM_vel[0], self.COM_vel[1]])
        xy_vel_err = np.linalg.norm(xy_vel - xy_vel_tar)

        # print(x_vel_err, y_vel_err, z_vel_err)
        torso_pitch_err = chest_link_state[1][1]
        pelvis_pitch_err = base_orn[1]
        torso_roll_err = chest_link_state[1][0]
        pelvis_roll_err = base_orn[0]

        # force distribution
        (COP, contact_force, _, right_COP, right_contact_force, _,
         left_COP, left_contact_force, _) = self.COP_info_filtered
        force_targ = self.total_mass * self.g / 2.0
        # Z contact force
        left_foot_force_err = force_targ - left_contact_force[2]
        right_foot_force_err = force_targ - right_contact_force[2]

        # foot roll
        right_foot_link_state = p.getLinkState(self.r,
                                               self.jointIdx['rightAnkleRoll'],
                                               computeLinkVelocity=0)
        right_foot_quat = right_foot_link_state[1]
        right_foot_orn = p.getEulerFromQuaternion(right_foot_quat)
        right_foot_roll_err = right_foot_orn[2]
        left_foot_link_state = p.getLinkState(self.r,
                                              self.jointIdx['leftAnkleRoll'],
                                              computeLinkVelocity=0)
        left_foot_quat = left_foot_link_state[1]
        left_foot_orn = p.getEulerFromQuaternion(left_foot_quat)
        left_foot_roll_err = left_foot_orn[2]

        x_pos_reward = math.exp(-19.51 * x_pos_err ** 2)
        y_pos_reward = math.exp(-19.51 * y_pos_err ** 2)
        z_pos_reward = math.exp(-113.84 * z_pos_err ** 2)  # -79.73
        xy_pos_reward = math.exp(-19.51 * xy_pos_err ** 2)

        x_vel_reward = math.exp(-0.57 * (x_vel_err) ** 2)
        y_vel_reward = math.exp(-0.57 * (y_vel_err) ** 2)
        z_vel_reward = math.exp(-3.69 * (z_vel_err) ** 2)  # -1.85
        xy_vel_reward = math.exp(-0.57 * (xy_vel_err) ** 2)

        torso_pitch_reward = math.exp(-4.68 * (torso_pitch_err) ** 2)
        pelvis_pitch_reward = math.exp(-4.68 * (pelvis_pitch_err) ** 2)
        torso_roll_reward = math.exp(-4.68 * (torso_roll_err) ** 2)
        pelvis_roll_reward = math.exp(-4.68 * (pelvis_roll_err) ** 2)

        # -0.0185#11.513/800/800
        left_foot_force_reward = math.exp(-2e-5 * (left_foot_force_err) ** 2)
        right_foot_force_reward = math.exp(-2e-5 * (right_foot_force_err) ** 2)
        left_foot_roll_reward = math.exp(-4.68 * (left_foot_roll_err) ** 2)
        right_foot_roll_reward = math.exp(-4.68 * (right_foot_roll_err) ** 2)

        # reward = (1.0 * x_pos_reward + 1.0 * y_pos_reward + 3.0 * z_pos_reward
        #   + 1.0 * x_vel_reward + 1.0 * y_vel_reward + 1.0 * z_vel_reward
        #   + 1.0 * torso_pitch_reward + 1.0 * pelvis_pitch_reward
        #   + 1.0 * torso_roll_reward + 1.0 * pelvis_roll_reward) \
        #   * 10 / (1.0 + 1.0 + 3.0 + 1.0 + 1.0 + 1.0 + 1.0 + 1.0 + 1.0 + 1.0)

        reward = (2.0 * xy_pos_reward + 3.0 * z_pos_reward
                  + 2.0 * xy_vel_reward + 1.0 * z_vel_reward
                  + 1.0 * torso_pitch_reward + 1.0 * pelvis_pitch_reward
                  + 1.0 * torso_roll_reward + 1.0 * pelvis_roll_reward
                  + 1.0 * left_foot_force_reward + 1.0 * right_foot_force_reward
                  # +1.0 * left_foot_roll_reward + 1.0 * right_foot_roll_reward
                  ) * 10 / (
                      2.0 + 3.0 + 2.0 + 1.0 + 1.0 + 1.0 + 1.0 + 1.0 + 1.0 + 1.0)
        # + 1.0 + 1.0)  # 5.0
        # reward = (
        #         4.0 * z_pos_reward\
        #         +1.0 * z_vel_reward \
        #         +1.0 * torso_pitch_reward + 1.0 * pelvis_pitch_reward \
        #         +1.0 * torso_roll_reward + 1.0 * pelvis_roll_reward \
        #         +1.0 * left_foot_force_reward + 1.0 * right_foot_force_reward
        #         +1.0 * left_foot_roll_reward + 1.0 * right_foot_roll_reward
        #           ) \ * 10 / (
        #   3.0 + 1.0 + 1.0 + 1.0 + 1.0 + 1.0 + 1.0 + 1.0)# + 1.0 + 1.0)  # 5.0
        # reward = (1.0 * x_pos_reward + 2.0 * z_pos_reward + \
        #           1.0 * x_vel_reward + 1.0 * z_vel_reward + \
        #           1.0 * torso_pitch_reward + 1.0 * pelvis_pitch_reward)\
        #           *10/(1.0+2.0+1.0+1.0+1.0+1.0) #5.0

        foot_contact_term = 0
        fall_term = 0
        if not (self.checkGroundContact('right') or
                self.checkGroundContact('left')):  # both feet lost contact
            # 1 TODO increase penalty for losing contact with the ground
            foot_contact_term -= 1
        if self.checkFall():
            fall_term -= 10

        reward = reward + fall_term + foot_contact_term
        # reward + fall_term+ foot_contact_term
        # penalize reward when the target position is hard to achieve
        position_follow_penalty = 0
        for key in self.controlled_joints:  # TODO
            joint_state = p.getJointState(self.r, self.jointIdx[key])
            position_follow_penalty -= 1.0 * abs(
                joint_state[0] - self.action[key])
            # penalize reward when joint is moving too fast
        velocity_penalty = 0
        for key in self.controlled_joints:  # TODO
            joint_state = p.getJointState(self.r, self.jointIdx[key])
            # velocity_penalty -= 1.0 * abs(joint_state[1] / self.v_max[key])
            velocity_penalty -= (joint_state[1] / self.v_max[key])**2
        # penalize reward when torque
        torque_penalty = 0
        for key in self.controlled_joints:  # TODO
            joint_state = p.getJointState(self.r, self.jointIdx[key])
            torque_penalty -= 1.0 * abs(joint_state[3] / self.u_max[key])
        # penalize power rate of joint motoUntitled Folderr
        power_penalty = 0
        for key in self.controlled_joints:  # TODO
            joint_state = p.getJointState(self.r, self.jointIdx[key])
            power_penalty -= 1 * abs(joint_state[3] / self.u_max[key]) * abs(
                joint_state[1] / self.v_max[key])
        # penalize change in torque
        torque_change_penalty = 0
        for key in self.controlled_joints:
            joint_state = p.getJointState(self.r, self.jointIdx[key])
            torque_change_penalty -= 1.0 * abs(
                self.hist_torque[key] - joint_state[3]) / self.u_max[key]

        # reward += 2*velocity_penalty
        # reward += 30 *velocity_penalty/len(self.controlled_joints)
        # reward += 30 * power_penalty / len(self.controlled_joints) #30
        # reward += 5 * power_penalty

        reward_term = dict([
            ("x_pos_reward", x_pos_reward),
            ("y_pos_reward", y_pos_reward),
            ("z_pos_reward", z_pos_reward),
            ("x_vel_reward", x_vel_reward),
            ("y_vel_reward", y_vel_reward),
            ("z_vel_reward", z_vel_reward),
            ("torso_pitch_reward", torso_pitch_reward),
            ("pelvis_pitch_reward", pelvis_pitch_reward),
            ("torso_roll_reward", torso_roll_reward),
            ("pelvis_roll_reward", pelvis_roll_reward),
            ("power_penalty", power_penalty),
            ("torque_change_penalty", torque_change_penalty),
            ("velocity_penalty", velocity_penalty),
            ("torque_penalty", torque_penalty),
            ("position_follow_penalty", position_follow_penalty),
            ("xy_pos_reward", xy_pos_reward),
            ("xy_vel_reward", xy_vel_reward),
            ("left_foot_force_reward", left_foot_force_reward),
            ("right_foot_force_reward", right_foot_force_reward),
            ("left_foot_roll_reward", left_foot_roll_reward),
            ("right_foot_roll_reward", right_foot_roll_reward),
            ("foot_contact_term", foot_contact_term),
            ("fall_term", fall_term)
        ])
        return reward, reward_term

    def _setDynamics(self):
        # set damping
        #    for jointNo in range(p.getNumJoints(self.r)):
        #        info = p.getJointInfo(self.r, jointNo)
        #        p.changeDynamics(self.r, info[0], linearDamping = 0.0,
        #                         angularDamping = 0.0)

        # set plane and foot friction to 1 and restitution as default
        # p.changeDynamics(self.r, self.jointIdx['leftAnklePitch'],
        # lateralFriction=1.0, spinningFriction=0.03, rollingFriction=0.03,
        # restitution=0.0)
        # p.changeDynamics(self.r, self.jointIdx['rightAnklePitch'],
        # lateralFriction=1.0, spinningFriction=0.03, rollingFriction=0.03,
        # restitution=0.0)
        # p.changeDynamics(self.r, self.jointIdx['leftAnkleRoll'],
        # lateralFriction=1.0, spinningFriction=0.03, rollingFriction=0.03,
        # restitution=0.0)
        # p.changeDynamics(self.r, self.jointIdx['rightAnkleRoll'],
        # lateralFriction=1.0, spinningFriction=0.03, rollingFriction=0.03,
        # restitution=0.0)
        # p.changeDynamics(self.plane, -1, lateralFriction=1.0, restitution=0.0)
        restitution = 0.0
        # spinningFriction = 0.03
        # rollingFriction = 0.03
        lateralFriction = 1.0

        p.changeDynamics(self.r, self.jointIdx['leftAnklePitch'],
                         restitution=restitution,
                         lateralFriction=lateralFriction,
                         # spinningFriction=spinningFriction,
                         # rollingFriction=rollingFriction
                         )
        p.changeDynamics(self.r, self.jointIdx['rightAnklePitch'],
                         restitution=restitution,
                         lateralFriction=lateralFriction,
                         # spinningFriction=spinningFriction,
                         # rollingFriction=rollingFriction
                         )
        p.changeDynamics(self.r, self.jointIdx['leftAnkleRoll'],
                         restitution=restitution,
                         lateralFriction=lateralFriction,
                         # spinningFriction=spinningFriction,
                         # rollingFriction=rollingFriction
                         )
        p.changeDynamics(self.r, self.jointIdx['rightAnkleRoll'],
                         restitution=restitution,
                         lateralFriction=lateralFriction,
                         # spinningFriction=spinningFriction,
                         # rollingFriction=rollingFriction
                         )
        p.changeDynamics(self.plane, -1, restitution=restitution,
                         lateralFriction=lateralFriction,
                         # spinningFriction=spinningFriction,
                         # rollingFriction=rollingFriction
                         )

        return 0

    def _render(self, mode='human', close=False):
        p.addUserDebugLine(
            self.COM_pos + np.array([0, 0, 2]),
            self.COM_pos + np.array([0, 0, -2]),
            [1, 0, 0],
            5, 0.1)  # TODO rendering to draw COM
        p.addUserDebugLine(
            self.support_polygon_center[0] + np.array([0, 0, 2]),
            self.support_polygon_center[0] + np.array([0, 0, -2]),
            [0, 1, 0],
            5, 0.1)  # TODO rendering to draw support polygon
        p.addUserDebugLine(
            self.support_polygon_center[0] + np.array([2, 0, 0]),
            self.support_polygon_center[0] + np.array([-2, 0, 0]),
            [0, 1, 0],
            5, 0.1)  # TODO rendering to draw support polygon
        return

    def _termination(self):
        return self.checkFall()
        # return (self._envStepCounter > self.max_steps) or (self.checkFall())

    # TODO create function to log video
    def _startLoggingVideo(self):
        self.logId = p.startStateLogging(
            loggingType=p.STATE_LOGGING_VIDEO_MP4,
            fileName=self._logFileName + '/video.mp4')

    def _stopLoggingVideo(self):
        # p.startStateLogging(self.logId)
        p.stopStateLogging(self.logId)

    def _setupFilter(self):
        # filtering for COP
        # TODO test calculation and filtering for COP
        self.COP_filter_method = calCOP(10, 10, self._dt_filter, 1)

        # filtering states
        self.state_filter_method = {}
        # TODO test filtering for state and reward

        # setup filter
        for i in range(self.stateNumber):
            # TODO experiment with different cutoff frequencies
            self.state_filter_method[i] = FilterClass()
            self.state_filter_method[i].butterworth(
                self._dt_filter, 10, 1)  # sample period, cutoff freq, order

        for key in self.controlled_joints:
            # TODO Add self defined PD controller
            if key in self.Kp:
                self.pd_controller.update({
                                           key:
                                           PDController(
                                               [self.Kp[key],
                                                self.Kd[key]],
                                               self.u_max[key],
                                               self.v_max[key],
                                               key, [True, True, False],
                                               self._dt_filter, [10, 10, 10],
                                               1)})  # 250
            else:  # PD gains not defined
                continue

        self.COM_pos_local_filter_method = {}
        self.COM_pos_local_filter_method[0] = FilterClass()
        self.COM_pos_local_filter_method[0].butterworth(self._dt_filter, 10, 1)
        self.COM_pos_local_filter_method[1] = FilterClass()
        self.COM_pos_local_filter_method[1].butterworth(self._dt_filter, 10, 1)
        self.COM_pos_local_filter_method[2] = FilterClass()
        self.COM_pos_local_filter_method[2].butterworth(self._dt_filter, 10, 1)

    def _setupCamera(
        self,
        cameraDistance=1.5,
        cameraYaw=0,
        cameraPitch=0,
        cameraTargetPosition=[
            0,
            0,
            0.7]):
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        p.resetDebugVisualizerCamera(
            cameraDistance=cameraDistance,
            cameraYaw=cameraYaw,
            cameraPitch=cameraPitch,
            cameraTargetPosition=cameraTargetPosition)

    def _setupSimulation(
            self,
            base_pos_nom=None,
            base_orn_nom=None,
            fixed_base=False):
        if base_pos_nom is None:
            base_pos_nom = self.base_pos_nom
        if base_orn_nom is None:
            base_orn_nom = self.base_orn_nom
        # p.setPhysicsEngineParameter(numSolverIterations=10, erp=0.2) # Physics
        # engine parameter default solver iteration = 50
        self._setupFilter()

        p.resetSimulation()

        p.setRealTimeSimulation(0)
        p.setGravity(0, 0, -self.g)  # TODO set gravity
        p.setTimeStep(self._dt)

        self.dir_path = CURRENT_DIR

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        plane_urdf = os.path.join(
            self.dir_path, "assets", "plane", "plane.urdf")

        self.plane = p.loadURDF(
            plane_urdf, basePosition=[
                0, 0, 0], useFixedBase=True)
        valkyrie_urdf = os.path.join(
            self.dir_path, "assets",
            "valkyrie_bullet_mass_sims_modified_foot_"
            "collision_box_modified_self_collision.urdf")

        self.r = p.loadURDF(
            fileName=valkyrie_urdf, basePosition=base_pos_nom,
            baseOrientation=base_orn_nom,
            flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION,
            # flags=p.URDF_USE_INERTIA_FROM_FILE,
            # flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
            # flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS,
            useFixedBase=fixed_base)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        # p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0,
        # cameraPitch=0, cameraTargetPosition=[0, 0, 0.7])
        # print(p.getDebugVisualizerCamera())

        # Set up joint mapping
        # TODO fix this
        for jointNo in range(p.getNumJoints(self.r)):
            info = p.getJointInfo(self.r, jointNo)
            joint_name = info[1].decode("utf-8")
            self.jointIdx.update({joint_name: info[0]})
            self.jointNameIdx.update({info[0]: joint_name})
        self.nq = len(self.controllable_joints)

        for joint in self.controllable_joints:
            info = p.getJointInfo(self.r, self.jointIdx[joint])
            self.jointLowerLimit.append(info[8])
            self.jointUpperLimit.append(info[9])

        self._setDynamics()

        self.resetJointStates(base_pos_nom, base_orn_nom)
        self.setZeroOrderHoldNominalPose()
        self.getLinkMass()

        # TODO test joint reaction force torque
        # p.enableJointForceTorqueSensor(self.r,
        # self.jointIdx['leftAnklePitch'],True)
        # p.enableJointForceTorqueSensor(self.r,
        # self.jointIdx['rightAnklePitch'],True)
        p.enableJointForceTorqueSensor(
            self.r, self.jointIdx['leftAnkleRoll'], True)
        p.enableJointForceTorqueSensor(
            self.r, self.jointIdx['rightAnkleRoll'], True)

        for _ in range(int(self.initial_gap_steps)):  # PD loop time steps
            for _ in range(int(self._actionRepeat)):
                p.stepSimulation()

            # update information
            self.getLinkCOMPos()
            self.getLinkCOMVel()
            self.calCOMPos()
            self.calCOMVel()
            # TODO calculate pelvis acceleration
            self.calPelvisAcc()
            self.calCOP()
            # initialize filter value
            self.performFiltering()
            # self.initializeFiltering()

        # record history joint torque output
        for key in self.controlled_joints:  # TODO
            joint_state = p.getJointState(self.r, self.jointIdx[key])
            # update history_torque
            self.hist_torque.update({key: joint_state[3]})

            self.action.update({key: 0.0})  # initialize PD control input

    def getExtendedObservation(self):
        # self._observation = self.getObservation()
        # filtered observation
        self._observation = self.getFilteredObservation()
        return self._observation

    def toggleRendering(self):
        if self._renders:  # It's on, so turn it off
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            self._renders = False
        else:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            self._renders = True

    def resetJointStates(self, base_pos_nom=None, base_orn_nom=None):
        if base_pos_nom is None:
            base_pos_nom = self.base_pos_nom
        if base_orn_nom is None:
            base_orn_nom = self.base_orn_nom

        for jointName in self.q_nom:
            p.resetJointState(self.r,
                              self.jointIdx[jointName],
                              targetValue=self.q_nom[jointName],
                              targetVelocity=0)
        p.resetBasePositionAndOrientation(self.r, base_pos_nom, base_orn_nom)
        p.resetBaseVelocity(self.r, [0, 0, 0], [0, 0, 0])

    def getObservation(self):
        x_observation = np.zeros((self.stateNumber,))

        # This may not be the Pelvis CoM position _exactly_ but should be fine,
        # otherwise can apply local transformation
        base_pos, base_quat = p.getBasePositionAndOrientation(self.r)
        base_orn = p.getEulerFromQuaternion(base_quat)
        base_pos_vel, base_orn_vel = p.getBaseVelocity(self.r)

        # Gravitational acceleration acts as a reference for pitch and roll but
        # not for yaw.

        # overwrite base_pos with the calculated base COM position
        # base_pos is tuple, tuple is immutable
        # change tuple to array
        base_pos = np.array(base_pos)
        base_pos[0] = self.linkCOMPos['pelvisBase'][0]
        base_pos[1] = self.linkCOMPos['pelvisBase'][1]
        base_pos[2] = self.linkCOMPos['pelvisBase'][2]
        # base_pos = self.linkCOMPos['pelvisBase']

        Rz = self.rotZ(base_orn[2])
        Rz_i = np.linalg.inv(Rz)
        # R = self.transform(base_quat)
        # R_i = np.linalg.inv(R)

        # pelvis positional velocity
        base_pos_vel = np.array(base_pos_vel)
        base_pos_vel.resize(1, 3)
        # base velocity in base (pelvis) frame
        # base_pos_vel_base = np.transpose(R_i @ base_pos_vel.transpose())
        # base velocity in adjusted yaw frame
        base_pos_vel_yaw = np.transpose(Rz_i @ base_pos_vel.transpose())
        # base_pos_vel_yaw = base_pos_vel
        x_observation[0] = base_pos_vel_yaw[0][0]  # pelvis_x_dot
        x_observation[1] = base_pos_vel_yaw[0][1]  # pelvis_y_dot
        x_observation[2] = base_pos_vel_yaw[0][2]  # pelvis_z_dot
        # pelvis

        x_observation[3] = base_orn[0]  # pelvis_roll
        x_observation[4] = base_orn[1]  # pelvis_pitch

        base_orn_vel = np.array(base_orn_vel)
        base_orn_vel.resize(1, 3)
        # base_orn_vel_base = np.transpose(R_i @ base_orn_vel.transpose())
        base_orn_vel_yaw = np.transpose(Rz_i @ base_orn_vel.transpose())
        x_observation[5] = base_orn_vel_yaw[0][0]  # pelvis_roll_dot
        x_observation[6] = base_orn_vel_yaw[0][1]  # pelvis_pitch_dot
        x_observation[7] = base_orn_vel_yaw[0][2]  # pelvis_yaw_dot

        # chest
        chest_link_state = p.getLinkState(
            self.r, self.jointIdx['torsoRoll'],
            computeLinkVelocity=1)
        chest_link_dis = np.array(chest_link_state[0]) - np.array(base_pos)
        chest_link_dis.resize(1, 3)
        # chest_link_dis_base = np.transpose(R_i @ chest_link_dis.transpose())
        chest_link_dis_yaw = np.transpose(Rz_i @ chest_link_dis.transpose())
        # chest_link_dis_yaw = chest_link_dis
        # chest_com_position_x - pelvis_com_position_x
        x_observation[8] = chest_link_dis_yaw[0][0]
        # chest_com_position_y - pelvis_com_position_y
        x_observation[9] = chest_link_dis_yaw[0][1]
        # chest_com_position_z - pelvis_com_position_z
        x_observation[10] = chest_link_dis_yaw[0][2]

        torso_pitch_joint_state = p.getJointState(
            self.r, self.jointIdx['torsoPitch'])
        x_observation[11] = torso_pitch_joint_state[0]
        x_observation[12] = (torso_pitch_joint_state[1]
                             / self.v_max['torsoPitch'])

        right_hip_roll_joint_state = p.getJointState(
            self.r, self.jointIdx['rightHipRoll'])
        x_observation[13] = right_hip_roll_joint_state[0]  # position
        # velocity
        x_observation[14] = (right_hip_roll_joint_state[1]
                             / self.v_max['rightHipRoll'])
        right_hip_pitch_joint_state = p.getJointState(
            self.r, self.jointIdx['rightHipPitch'])
        x_observation[15] = right_hip_pitch_joint_state[0]  # position
        # velocity
        x_observation[16] = (right_hip_pitch_joint_state[1]
                             / self.v_max['rightHipPitch'])
        right_knee_pitch_joint_state = p.getJointState(
            self.r, self.jointIdx['rightKneePitch'])
        x_observation[17] = right_knee_pitch_joint_state[0]  # position
        # velocity
        x_observation[18] = (right_knee_pitch_joint_state[1]
                             / self.v_max['rightKneePitch'])
        right_ankle_pitch_joint_state = p.getJointState(
            self.r, self.jointIdx['rightAnklePitch'])
        x_observation[19] = right_ankle_pitch_joint_state[0]
        x_observation[20] = (right_ankle_pitch_joint_state[1]
                             / self.v_max['rightAnklePitch'])
        right_ankle_roll_joint_state = p.getJointState(
            self.r, self.jointIdx['rightAnkleRoll'])
        x_observation[21] = right_ankle_roll_joint_state[0]
        x_observation[22] = (right_ankle_roll_joint_state[1]
                             / self.v_max['rightAnkleRoll'])

        right_foot_link_state = p.getLinkState(
            self.r, self.jointIdx['rightAnkleRoll'],
            computeLinkVelocity=0)
        right_foot_link_dis = np.array(right_foot_link_state[0]) - base_pos
        right_foot_link_dis.resize(1, 3)
        # right_foot_link_dis_base = np.transpose(
        #     R_i @right_foot_link_dis.transpose())
        right_foot_link_dis_yaw = np.transpose(
            Rz_i @right_foot_link_dis.transpose())
        # right_foot_link_dis_yaw = right_foot_link_dis
        # foot_com_position_x - pelvis_com_position_x
        x_observation[23] = right_foot_link_dis_yaw[0][0]
        # foot_com_position_y - pelvis_com_position_y
        x_observation[24] = right_foot_link_dis_yaw[0][1]
        # foot_com_position_z - pelvis_com_position_z
        x_observation[25] = right_foot_link_dis_yaw[0][2]

        left_hip_roll_joint_state = p.getJointState(
            self.r, self.jointIdx['leftHipRoll'])
        x_observation[26] = left_hip_roll_joint_state[0]  # position
        # velocity
        x_observation[27] = (left_hip_roll_joint_state[1]
                             / self.v_max['leftHipRoll'])
        left_hip_pitch_joint_state = p.getJointState(
            self.r, self.jointIdx['leftHipPitch'])
        x_observation[28] = left_hip_pitch_joint_state[0]  # position
        # velocity
        x_observation[29] = (left_hip_pitch_joint_state[1]
                             / self.v_max['leftHipPitch'])
        left_knee_pitch_joint_state = p.getJointState(
            self.r, self.jointIdx['leftKneePitch'])
        x_observation[30] = left_knee_pitch_joint_state[0]  # position
        # velocity
        x_observation[31] = (left_knee_pitch_joint_state[1]
                             / self.v_max['leftKneePitch'])
        left_ankle_pitch_joint_state = p.getJointState(
            self.r, self.jointIdx['leftAnklePitch'])
        x_observation[32] = left_ankle_pitch_joint_state[0]
        x_observation[33] = (left_ankle_pitch_joint_state[1]
                             / self.v_max['leftAnklePitch'])
        left_ankle_roll_joint_state = p.getJointState(
            self.r, self.jointIdx['leftAnkleRoll'])
        x_observation[34] = left_ankle_roll_joint_state[0]
        x_observation[35] = (left_ankle_roll_joint_state[1]
                             / self.v_max['leftAnkleRoll'])

        left_foot_link_state = p.getLinkState(
            self.r, self.jointIdx['leftAnkleRoll'],
            computeLinkVelocity=0)
        left_foot_link_dis = np.array(left_foot_link_state[0]) - base_pos
        left_foot_link_dis.resize(1, 3)
        # left_foot_link_dis_base = np.transpose(
        #     R_i @left_foot_link_dis.transpose())
        left_foot_link_dis_yaw = np.transpose(
            Rz_i @left_foot_link_dis.transpose())
        # left_foot_link_dis_yaw = left_foot_link_dis
        # foot_com_position_x - pelvis_com_position_x
        x_observation[36] = left_foot_link_dis_yaw[0][0]
        # foot_com_position_y - pelvis_com_position_y
        x_observation[37] = left_foot_link_dis_yaw[0][1]
        # foot_com_position_z - pelvis_com_position_z
        x_observation[38] = left_foot_link_dis_yaw[0][2]

        COM_dis = np.array(self.COM_pos-np.array(base_pos))
        COM_dis.resize(1, 3)
        # COM_dis_base = np.transpose(R_i @ COM_dis.transpose())
        COM_dis_yaw = np.transpose(Rz_i @ COM_dis.transpose())
        # COM_dis_yaw = COM_dis
        x_observation[39] = COM_dis_yaw[0][0]
        x_observation[40] = COM_dis_yaw[0][1]
        x_observation[41] = COM_dis_yaw[0][2]

        cp = self.capturePoint()
        cp_dis = cp+COM_dis  # capture point w.r.t pelvis TODO
        cp_dis.resize(1, 3)
        # cp_dis_base = np.transpose(R_i @ cp_dis.transpose())
        cp_dis_yaw = np.transpose(Rz_i @ cp_dis.transpose())
        # cp_dis_yaw = cp_dis
        x_observation[42] = cp_dis_yaw[0][0]
        x_observation[43] = cp_dis_yaw[0][1]
        # x_observation[57] = cp_yaw[2]

        COM_vel = np.array(self.COM_vel)
        COM_vel.resize(1, 3)
        # COM_vel_base = np.transpose(R_i @ COM_vel.transpose())
        COM_vel_yaw = np.transpose(Rz_i @ COM_vel.transpose())
        # COM_vel_yaw = COM_vel
        x_observation[44] = COM_vel_yaw[0][0]
        x_observation[45] = COM_vel_yaw[0][1]
        x_observation[46] = COM_vel_yaw[0][2]

        (COP, contact_force, _, right_COP, right_contact_force, _, left_COP,
         left_contact_force, _) = self.COP_info_filtered
        COP_dis = np.array(COP) - np.array(base_pos)
        COP_dis.resize(1, 3)
        # COP_dis_base = np.transpose(R_i @ COP_dis.transpose())
        COP_dis_yaw = np.transpose(Rz_i @ COP_dis.transpose())
        # COP_dis_yaw  =COP_dis
        x_observation[47] = COP_dis_yaw[0][0]
        x_observation[48] = COP_dis_yaw[0][1]
        # x_observation[57] = COP_dis_yaw[2]

        x_observation[49] = right_contact_force[2]/800.0
        x_observation[50] = left_contact_force[2]/800.0

        return x_observation

    def getFilteredObservation(self):
        observation = self.getObservation()
        observation_filtered = np.zeros(np.size(observation))

        for i in range(self.stateNumber):
            observation_filtered[i] = self.state_filter_method[i].y[0]
        # TODO binary values should not be filtered.
        # observation_filtered[48] = observation[48]
        # observation_filtered[49] = observation[49]
        # observation_filtered[39]=self.left_contact_filter_method.y
        # observation_filtered[59]=self.right_contact_filter_method.y
        return observation_filtered

    def getReading(self):
        readings = dict()
        readings.update({'leftGroundContact': self.checkGroundContact('left')})
        readings.update(
            {'rightGroundContact': self.checkGroundContact('right')})

        readings.update({'leftGroundContactReactionForce':
                         self.checkGroundContactReactionForce('left')})
        readings.update({'rightGroundContactReactionForce':
                         self.checkGroundContactReactionForce('right')})
        readings.update({'leftGroundReactionForce':
                         self.jointReactionForce('left')})
        readings.update({'rightGroundReactionForce':
                         self.jointReactionForce('right')})

        torso_pitch_joint_state = p.getJointState(self.r,
                                                  self.jointIdx['torsoPitch'])
        readings.update({'torsoPitchAngle': torso_pitch_joint_state[0]})
        readings.update({'torsoPitchVelocity': torso_pitch_joint_state[1]})
        readings.update({'torsoPitchTorque': torso_pitch_joint_state[3]})

        left_hip_pitch_joint_state = p.getJointState(
            self.r, self.jointIdx['leftHipPitch'])
        left_knee_pitch_joint_state = p.getJointState(
            self.r, self.jointIdx['leftKneePitch'])
        left_ankle_pitch_joint_state = p.getJointState(
            self.r, self.jointIdx['leftAnklePitch'])
        readings.update({'leftHipPitchAngle': left_hip_pitch_joint_state[0]})
        readings.update({'leftHipPitchVelocity': left_hip_pitch_joint_state[1]})
        readings.update({'leftHipPitchTorque': left_hip_pitch_joint_state[3]})
        readings.update({'leftKneePitchAngle': left_knee_pitch_joint_state[0]})
        readings.update({'leftKneePitchVelocity':
                         left_knee_pitch_joint_state[1]})
        readings.update({'leftKneePitchTorque': left_knee_pitch_joint_state[3]})
        readings.update({'leftAnklePitchAngle':
                         left_ankle_pitch_joint_state[0]})
        readings.update({'leftAnklePitchVelocity':
                         left_ankle_pitch_joint_state[1]})
        readings.update({'leftAnklePitchTorque':
                         left_ankle_pitch_joint_state[3]})

        right_hip_pitch_joint_state = p.getJointState(
            self.r, self.jointIdx['rightHipPitch'])
        right_knee_pitch_joint_state = p.getJointState(
            self.r, self.jointIdx['rightKneePitch'])
        right_ankle_pitch_joint_state = p.getJointState(
            self.r, self.jointIdx['rightAnklePitch'])
        readings.update({'rightHipPitchAngle': right_hip_pitch_joint_state[0]})
        readings.update({'rightHipPitchVelocity':
                         right_hip_pitch_joint_state[1]})
        readings.update({'rightHipPitchTorque': right_hip_pitch_joint_state[3]})
        readings.update({'rightKneePitchAngle':
                         right_knee_pitch_joint_state[0]})
        readings.update({'rightKneePitchVelocity':
                         right_knee_pitch_joint_state[1]})
        readings.update({'rightKneePitchTorque':
                         right_knee_pitch_joint_state[3]})
        readings.update({'rightAnklePitchAngle':
                         right_ankle_pitch_joint_state[0]})
        readings.update({'rightAnklePitchVelocity':
                         right_ankle_pitch_joint_state[1]})
        readings.update({'rightAnklePitchTorque':
                         right_ankle_pitch_joint_state[3]})

        left_foot_link_state = p.getLinkState(
            self.r, self.jointIdx['leftAnkleRoll'],
            computeLinkVelocity=0)
        readings.update({'leftFootPitch': left_foot_link_state[1][1]})
        right_foot_link_state = p.getLinkState(
            self.r, self.jointIdx['rightAnkleRoll'],
            computeLinkVelocity=0)
        readings.update({'rightFootPitch': right_foot_link_state[1][1]})

        # right_COP,right_contact_force,_ = self.calFootCOP('right')
        # left_COP,left_contact_force,_ = self.calFootCOP('left')
        (COP, contact_force, _, right_COP, right_contact_force, _, left_COP,
         left_contact_force, _) = self.COP_info
        readings.update({'rightCOP': right_COP})
        readings.update({'leftCOP': left_COP})
        readings.update({'right_contact_force': right_contact_force})
        readings.update({'left_contact_force': left_contact_force})
        readings.update({'contact_force': contact_force})
        readings.update({'COP': COP})

        (COP, contact_force, _, right_COP, right_contact_force, _, left_COP,
         left_contact_force, _) = self.COP_info_filtered
        readings.update({'rightCOP_filtered': right_COP})
        readings.update({'leftCOP_filtered': left_COP})
        readings.update({'right_contact_force_filtered': right_contact_force})
        readings.update({'left_contact_force_filtered': left_contact_force})
        readings.update({'contact_force_filtered': contact_force})
        readings.update({'COP_filtered': COP})

        readings.update({'COM_vel': np.array(self.COM_vel)})
        readings.update({'COM_pos': np.array(self.COM_pos)})
        readings.update({'COM_pos_local': np.array(self.COM_pos_local)})

        readings.update({'COM_pos_local_filter':
                         np.array(self.COM_pos_local_filter)})
        readings.update({'support_polygon_center':
                         self.support_polygon_center[0]})
        readings.update({'support_polygon_center_surrogate':
                         self.support_polygon_center_surrogate[0]})

        # record individual terms in reward
        _, reward_term = self._reward()

        readings.update(reward_term)

        # record joint torque calculated by self defined PD controller
        for key in self.controlled_joints:
            if key in self.Kp:  # if PD gains are defined
                position = self.pd_controller[key].filtered_position
                velocity = self.pd_controller[key].filtered_velocity
                # torque = self.pd_controller[key].control(self.action[key], 0)
                torque = self.pd_controller[key].u
                P_torque = self.pd_controller[key].u_e
                D_torque = self.pd_controller[key].u_de
                readings.update({key + 'PDPosition': position})
                readings.update({key + 'PDVelocity': velocity})
                readings.update({key + 'PDPositionAdjusted':
                                 self.pd_controller[key].adjusted_position})
                readings.update({key + 'PDVelocityAdjusted':
                                 self.pd_controller[key].adjusted_velocity})
                readings.update({key + 'PDTorque': torque})
                readings.update({key + 'PDTorque_Kp': P_torque})
                readings.update({key + 'PDTorque_Kd': D_torque})
                readings.update(
                    {key + 'PDPositionKalman':
                     self.pd_controller[key].kalman_filtered_position})
                readings.update(
                    {key + 'PDVelocityKalman':
                     self.pd_controller[key].kalman_filtered_velocity})
                readings.update({key + 'PDFilteredTorque':
                                 self.PD_torque_filtered[key]})
                readings.update({key + 'PDUnfilteredTorque':
                                 self.PD_torque_unfiltered[key]})
                readings.update({key + 'PDAdjustedTorque':
                                 self.PD_torque_adjusted[key]})
                readings.update({key + 'PDTorqueWithAdjustedFeedback':
                                 self.pd_controller[key].u_adj})

        readings.update({'torsoPitchAngleTarget': self.action['torsoPitch']})

        readings.update(
            {'leftHipPitchAngleTarget': self.action['leftHipPitch']})
        readings.update(
            {'leftKneePitchAngleTarget': self.action['leftKneePitch']})
        readings.update(
            {'leftAnklePitchAngleTarget': self.action['leftAnklePitch']})

        readings.update(
            {'rightHipPitchAngleTarget': self.action['rightHipPitch']})
        readings.update(
            {'rightKneePitchAngleTarget': self.action['rightKneePitch']})
        readings.update(
            {'rightAnklePitchAngleTarget': self.action['rightAnklePitch']})

        # return filterred joint angles
        observation = self.getObservation()
        readings.update({'torsoPitchAngle10HzFiltered': observation[11]})

        readings.update({'leftHipPitchAngle10HzFiltered': observation[19]})
        readings.update({'leftKneePitchAngle10HzFiltered': observation[21]})
        readings.update({'leftAnklePitchAngle10HzFiltered': observation[23]})

        readings.update({'rightHipPitchAngle10HzFiltered': observation[32]})
        readings.update({'rightKneePitchAngle10HzFiltered': observation[34]})
        readings.update({'rightAnklePitchAngle10HzFiltered': observation[36]})

        # readings.update({'torsoPitchAngle50HzFiltered':
        # self.pd_controller['torsoPitch'].measured_position})
        #
        # readings.update({'leftHipPitchAngle50HzFiltered':
        # self.pd_controller['leftHipPitch'].measured_position})
        # readings.update({'leftKneePitchAngle50HzFiltered':
        # self.pd_controller['leftKneePitch'].measured_position})
        # readings.update({'leftAnklePitchAngle50HzFiltered':
        # self.pd_controller['leftAnklePitch'].measured_position})
        #
        # readings.update({'rightHipPitchAngle50HzFiltered':
        # self.pd_controller['rightHipPitch'].measured_position})
        # readings.update({'rightKneePitchAngle50HzFiltered':
        # self.pd_controller['rightKneePitch'].measured_position})
        # readings.update({'rightAnklePitchAngle50HzFiltered':
        # self.pd_controller['rightAnklePitch'].measured_position})

        readings.update({'pelvis_acc_global': self.pelvis_acc})
        readings.update({'pelvis_acc_base': self.pelvis_acc_base})

        readings.update({'pelvis_vel_global': self.pelvis_vel_his})
        readings.update({'pelvis_vel_base': self.pelvis_vel_his_base})

        return readings

    def getExtendedReading(self):
        return self._reading

    def initializeFiltering(self):
        observation = self.getObservation()
        for i in range(self.stateNumber):
            self.state_filter_method[i].initializeFilter(observation[i])

        # TODO filtering for PD controller
        for key in self.controlled_joints:
            joint_state = p.getJointState(self.r, self.jointIdx[key])
            position = joint_state[0]  # position
            velocity = joint_state[1]  # velocity
            torque = joint_state[3]
            if key in self.Kp:  # if PD gains is defined
                self.pd_controller[key].reset(position, velocity, torque)

        # Filtering for COM position
        self.COM_pos_local_filter_method[0].initializeFilter(
            self.COM_pos_local[0])
        self.COM_pos_local_filter_method[1].initializeFilter(
            self.COM_pos_local[1])
        self.COM_pos_local_filter_method[2].initializeFilter(
            self.COM_pos_local[2])
        self.COM_pos_local_filter[0] = self.COM_pos_local_filter_method[0].y[0]
        self.COM_pos_local_filter[1] = self.COM_pos_local_filter_method[1].y[0]
        self.COM_pos_local_filter[2] = self.COM_pos_local_filter_method[2].y[0]

    def performFiltering(self):  # TODO test filtering
        observation = self.getObservation()
        for i in range(self.stateNumber):
            self.state_filter_method[i].applyFilter(observation[i])
        # Binay state filter
        # self.left_contact_filter_method.applyFilter(observation[39])
        # self.right_contact_filter_method.applyFilter(observation[59])

        # TODO filtering for PD controller
        for key in self.controlled_joints:
            joint_state = p.getJointState(self.r, self.jointIdx[key])
            position = joint_state[0]  # position
            velocity = joint_state[1]  # velocity
            if key in self.Kp:  # if PD gains is defined
                self.pd_controller[key].updateMeasurements(position, velocity)

        self.COM_pos_local_filter_method[0].applyFilter(self.COM_pos_local[0])
        self.COM_pos_local_filter_method[1].applyFilter(self.COM_pos_local[1])
        self.COM_pos_local_filter_method[2].applyFilter(self.COM_pos_local[2])
        self.COM_pos_local_filter[0] = self.COM_pos_local_filter_method[0].y[0]
        self.COM_pos_local_filter[1] = self.COM_pos_local_filter_method[1].y[0]
        self.COM_pos_local_filter[2] = self.COM_pos_local_filter_method[2].y[0]

    def debug(self):  # for debugging
        right_info = p.getLinkState(
            self.r, self.jointIdx['rightAnkleRoll'],
            computeLinkVelocity=0)
        # left_info = p.getLinkState(
        #     self.r, self.jointIdx['leftAnkleRoll'], computeLinkVelocity=0)
        # Transformation from the link frame position to geometry center w.r.t
        # link frame
        T = np.array([[0.066], [0], [-0.056]])
        # Transformation from the link frame position to center of bottom of
        # foot w.r.t link frame
        T1 = np.array([[0.066], [0], [-0.088]])
        quat = right_info[5]
        T_ = (self.transform(quat)) @ T
        T1_ = (self.transform(quat)) @ T1
        foot_center = right_info[4] + T_.T
        foot_bottom_center = right_info[4] + T1_.T
        print(foot_center)
        print(foot_bottom_center)

        print('right foot')
        print('linkWorldPosition')
        print([right_info[0][0], right_info[0][1], right_info[0][2]])
        print('localInertialFramePosition')
        print([right_info[2][0], right_info[2][1], right_info[2][2]])
        print('worldLinkFramePosition')
        print([right_info[4][0], right_info[4][1], right_info[4][2]])

        print('left foot')
        print('linkWorldPosition')
        print([right_info[0][0], right_info[0][1], right_info[0][2]])
        print('localInertialFramePosition')
        print([right_info[2][0], right_info[2][1], right_info[2][2]])
        print('worldLinkFramePosition')
        print([right_info[4][0], right_info[4][1], right_info[4][2]])

    def calCOMPos(self):
        summ = np.zeros((1, 3))
        for key, value in self.linkMass.items():
            summ += np.array(self.linkCOMPos[key]) * value
        summ /= self.total_mass
        # update global COM position
        self.COM_pos[0:3] = np.array(summ)
        # update local COM position w.r.t center of support polygon

        right_foot_info = p.getLinkState(
            self.r, self.jointIdx['rightAnkleRoll'],
            computeLinkVelocity=0)
        left_foot_info = p.getLinkState(
            self.r, self.jointIdx['leftAnkleRoll'],
            computeLinkVelocity=0)

        # T = np.array([[0.066],[0],[-0.056]])#Transformation from the link
        # frame position to geometry center w.r.t link frame
        # Transformation from the link frame position to center of bottom of
        # foot w.r.t link frame
        T = np.array([[0.045], [0], [-0.088]])
        right_quat = right_foot_info[1]
        left_quat = left_foot_info[1]
        right_T1 = (self.transform(right_quat)) @ T
        left_T1 = (self.transform(left_quat)) @ T
        # print(right_T1)
        # print(left_T1)
        right_foot_bottom_center = right_foot_info[4] + right_T1.T
        left_foot_bottom_center = left_foot_info[4] + left_T1.T

        # support polygon changes with foot contact
        if (self.checkGroundContact('right') and
                not self.checkGroundContact('left')):
            self.support_polygon_center = right_foot_bottom_center
        elif (not self.checkGroundContact('right') and
                self.checkGroundContact('left')):
            self.support_polygon_center = left_foot_bottom_center
        elif (self.checkGroundContact('right') and
                self.checkGroundContact('left')):
            self.support_polygon_center = (
                right_foot_bottom_center + left_foot_bottom_center) / 2.0
        else:  # both feet not in contact
            self.support_polygon_center = np.array(
                self.support_polygon_center)  # maintain previous value
            # Take the foot closet to the ground
            # if right_foot_bottom_center[0][2] < left_foot_bottom_center[0][2]:
            #     self.support_polygon_center = right_foot_bottom_center
            # elif (left_foot_bottom_center[0][2]
            #       < right_foot_bottom_center[0][2]):
            #     self.support_polygon_center = left_foot_bottom_center
            # else:
            #     self.support_polygon_center = (
            # right_foot_bottom_center + left_foot_bottom_center) / 2.0

        self.support_polygon_center_surrogate = (
            right_foot_bottom_center + left_foot_bottom_center) / 2.0
        self.support_polygon_center_surrogate[0][2] = min(
            right_foot_bottom_center[0][2], left_foot_bottom_center[0][2])
        self.COM_pos_local[0:3] = self.COM_pos - self.support_polygon_center
        return summ

    def calCOMVel(self):
        summ = np.zeros((1, 3))
        for key, value in self.linkMass.items():
            summ += np.array(self.linkCOMVel[key]) * value
        summ /= self.total_mass
        self.COM_vel[0:3] = np.array(summ)
        return summ

    def getLinkMass(self):
        self.total_mass = 0
        info = p.getDynamicsInfo(self.r, -1)  # for base link
        self.linkMass.update({"pelvisBase": info[0]})
        self.total_mass += info[0]
        for key, value in self.jointIdx.items():
            info = p.getDynamicsInfo(self.r, value)
            self.linkMass.update({key: info[0]})
            self.total_mass += info[0]

    def getLinkCOMPos(self):
        base_pos, base_quat = p.getBasePositionAndOrientation(self.r)
        # base_orn = p.getEulerFromQuaternion(base_quat)
        # Transformation from the pelvis base position to pelvis COM w.r.t
        # local pelvis frame
        # T = np.array([[-0.00532], [-0.003512],
        #               [-0.0036]])
        # T1 = (self.transform(base_quat)) @ T  # using quaternion
        # T1 = (self.rotZ(base_orn[2]))@(self.rotY(base_orn[1]))
        # @(self.rotX(base_orn[0]))@T  # using Euler angles
        # base_com = base_pos + T1.T  # [[]]
        # base_com = base_com[0]  # []
        # self.linkCOMPos.update({"pelvisBase": base_com})
        # TODO check if base position is the COM of the pelvis
        # base position is the COM of the pelvis
        self.linkCOMPos.update({"pelvisBase": np.array(base_pos)})
        for key, value in self.jointIdx.items():
            info = p.getLinkState(self.r, value, computeLinkVelocity=0)
            self.linkCOMPos.update({key: info[0]})

    def getLinkCOMVel(self):
        base_pos_vel, base_orn_vel = p.getBaseVelocity(self.r)

        self.linkCOMVel.update({"pelvisBase": base_pos_vel})
        for key, value in self.jointIdx.items():
            info = p.getLinkState(self.r, value, computeLinkVelocity=1)
            self.linkCOMVel.update({key: info[6]})

    # physical properties derived from capture point theory
    # def targetCOMVel(self, x, axis='x'):
    #     # input x is the target displacement w.r.t center of foot
    #     if axis == 'x':
    #         x_tar = self.COM_pos_local[0]
    #     elif axis == 'y':
    #         x_tar = self.COM_pos_local[1]
    #     else:
    #         x_tar = self.COM_pos_local[0]
    #     dist = x - x_tar
    #     z = max(0.05, x_tar)  # Make sure denominator is not zero
    #     tau = np.sqrt(z / 9.8)
    #     V = dist / tau
    #     return V

    def capturePoint(self):
        # input x is the target displacement w.r.t center of foot
        h = self.COM_pos_local[2]  # height of COM w.r.t bottom of foot
        # h = self.COM_pos[2] - self.support_polygon_center[0][2]
        z = max(0.05, h)  # Make sure denominator is not zero
        COM_vel = self.COM_vel
        cp = COM_vel*np.sqrt(z/self.g)

        return np.array(cp)

    # input x is the target displacement w.r.t center of foot
    def targetCOMVel(self, tar, axis='x'):
        # COM_pos_local = self.COM_pos_local_filter
        COM_pos_local = self.COM_pos_local
        h = COM_pos_local[2]  # height of COM w.r.t bottom of foot
        z = max(0.05, h)  # Make sure denominator is not zero
        tau = np.sqrt(z / self.g)

        if axis == 'x':
            x_tar = tar
            x_dist = x_tar - COM_pos_local[0]  # COM pos w.r.t to bottom foot
            x_vel = x_dist / tau
            return x_vel

        elif axis == 'y':
            y_tar = tar
            y_dist = y_tar - COM_pos_local[1]
            y_vel = y_dist / tau
            return y_vel

    def rejectableForce(self, t):  # t is impulse lasting period
        foot_length = 0.26  # TODO check length of foot
        # TODO check whether COM is at the center of foot
        V = np.array([self.targetCOMVel(-foot_length / 2, 'x'),
                      self.targetCOMVel(foot_length / 2, 'x')])
        F = V * self.total_mass / t
        return np.array([F[0], F[1]])

    def rejectableForce_xy(self, t):  # t is impulse lasting period
        base_pos, base_quat = p.getBasePositionAndOrientation(self.r)
        base_orn = p.getEulerFromQuaternion(base_quat)
        Rz = self.rotZ(base_orn[2])
        Rz_i = np.linalg.inv(Rz)

        hull = np.array(self.contact_points)
        # hull = np.array(self.hull) # (i,2)

        if np.shape(hull)[0] < 3:  # first axis
            return 0, 0, 0, 0  # X max X min Y max Y min
        hull = hull - self.COM_pos[0:2]  # support polygon hull w.r.t COM xy pos

        # add z coordinate
        hull_full = np.zeros((np.shape(hull)[0], 3))
        hull_full[:, 0:2] = hull[:, 0:2]
        # transform
        # base velocity in adjusted yaw frame
        hull_full_yaw = np.transpose(Rz_i @ hull_full.transpose())
        hull = hull_full_yaw[:, 0:2]

        x_pos = hull[:, 0]
        y_pos = hull[:, 1]
        x_pos_sort = np.sort(x_pos)  # small to large
        x_min = x_pos_sort[0]
        x_max = x_pos_sort[-1]
        # x_min = (x_pos_sort[0]+x_pos_sort[1])/2.0
        # x_max = (x_pos_sort[-1]+x_pos_sort[-2])/2.0
        y_pos_sort = np.sort(y_pos)  # small to large
        y_min = y_pos_sort[0]
        y_max = y_pos_sort[-1]
        # y_min = (y_pos_sort[0]+y_pos_sort[1])/2.0
        # y_max = (y_pos_sort[-1]+y_pos_sort[-2])/2.0
        h = self.COM_pos_local[2]  # height of COM w.r.t bottom of foot
        z = max(0.05, h)  # Make sure denominator is not zero
        tau = np.sqrt(z / self.g)
        Fx_min = self.total_mass/tau*x_min/t
        Fx_max = self.total_mass/tau*x_max/t
        Fy_min = self.total_mass/tau*y_min/t
        Fy_max = self.total_mass/tau*y_max/t

        return Fx_min, Fx_max, Fy_min, Fy_max

    def rotX(self, theta):
        R = np.array([
            [1.0, 0.0, 0.0],
            [0.0, math.cos(theta), -math.sin(theta)],
            [0.0, math.sin(theta), math.cos(theta)]])
        return R

    def rotY(self, theta):
        R = np.array([
            [math.cos(theta), 0.0, math.sin(theta)],
            [0.0, 1.0, 0.0],
            [-math.sin(theta), 0.0, math.cos(theta)]])
        return R

    def rotZ(self, theta):
        R = np.array([
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0]])
        return R

    def transform(self, qs):  # transform quaternion into rotation matrix
        qx = qs[0]
        qy = qs[1]
        qz = qs[2]
        qw = qs[3]

        x2 = qx + qx
        y2 = qy + qy
        z2 = qz + qz
        xx = qx * x2
        yy = qy * y2
        wx = qw * x2
        xy = qx * y2
        yz = qy * z2
        wy = qw * y2
        xz = qx * z2
        zz = qz * z2
        wz = qw * z2

        m = np.empty([3, 3])
        m[0, 0] = 1.0 - (yy + zz)
        m[0, 1] = xy - wz
        m[0, 2] = xz + wy
        m[1, 0] = xy + wz
        m[1, 1] = 1.0 - (xx + zz)
        m[1, 2] = yz - wx
        m[2, 0] = xz - wy
        m[2, 1] = yz + wx
        m[2, 2] = 1.0 - (xx + yy)

        return m

    def checkGroundContact(self, side):
        assert side in ['left', 'right']
        # TODO ground contact detection using contact point
        ground_contact = len(p.getContactPoints(
            self.r, self.plane, self.jointIdx[side + 'AnkleRoll'], -1)) > 0
        return 1.0 if ground_contact else 0.0

    def checkGroundContactReactionForce(self, side):
        # TODO ground contact detection using ground reaction force
        torque = self.jointReactionForce(side)
        # TODO adjust sigmoid function to represent actual contact as close
        # as possible
        # return 1/(1+np.exp(-(torque-9)))#sigmoid function
        return 1 / (1 + np.exp(-0.3 * (torque - 4)))

    def calFootCOP(self, side):
        assert side in ['left', 'right']
        footGroundContact = []
        # TODO ground contact detection using contact point
        ankleRollContact = p.getContactPoints(
            self.r, self.plane, self.jointIdx[side + 'AnkleRoll'], -1)
        anklePitchContact = p.getContactPoints(
            self.r, self.plane, self.jointIdx[side + 'AnklePitch'], -1)

        footGroundContact.extend(ankleRollContact)
        footGroundContact.extend(anklePitchContact)
        if not self.checkGroundContact(side):  # no contact
            return [
                0.0, 0.0, 0.0], [
                0.0, 0.0, 0.0], False  # COP does not exists
        # print(len(ankleRollContact))
        pCOP = np.array([0, 0, 0])  # position of Center of pressure
        # force among the x,y,z axis of world frame
        contactForce = np.array([0, 0, 0])
        # print(len(footGroundContact))
        for i in range(len(footGroundContact)):
            # contact normal of foot pointing towards plane
            contactNormal = np.array(footGroundContact[i][7])
            # contact normal of plane pointing towards foot
            contactNormal = -contactNormal
            contactNormalForce = np.array(footGroundContact[i][9])
            contactPosition = np.array(
                footGroundContact[i][5])  # position on plane
            forceX = contactNormal[0] * contactNormalForce
            forceY = contactNormal[1] * contactNormalForce
            # force along the z axis is larger than zero
            forceZ = max(abs(contactNormal[2] * contactNormalForce), 1e-6)
            # sum of contact force
            contactForce = contactForce + np.array([forceX, forceY, forceZ])
            # integration of contact point times vertical force
            pCOP = pCOP + contactPosition * forceZ
        pCOP = pCOP / contactForce[2]
        # pCOP[2] = 0.0  # z position is 0, on the plane
        # contactForce = contactForce / len(footGroundContact)
        return pCOP, contactForce, True

    def calFootCOP2(self, side):
        assert side in ['left', 'right']
        joint_state = p.getJointState(self.r, self.jointIdx[side+'AnkleRoll'])
        joint_reaction_force = joint_state[2]  # [Fx, Fy, Fz, Mx, My, Mz]
        Fx, Fy, Fz, Mx, My, Mz = -np.array(joint_reaction_force)
        foot_info = p.getLinkState(
            self.r, self.jointIdx[side + 'AnkleRoll'],
            computeLinkVelocity=0)
        # link_frame_pos = foot_info[4]
        link_frame_pos = foot_info[0]
        # Transformation from link frame position to geometry center w.r.t
        # link frame
        # T = np.array([[0.066],[0],[-0.056]])
        # Transformation from link frame position to center of bottom of foot
        # w.r.t link frame
        # T = np.array([[0.044], [0], [ -0.088]])
        # right_quat = right_foot_info[1]
        # left_quat = left_foot_info[1]
        # right_T1 = (self.transform(right_quat)) @ T
        # left_T1 = (self.transform(left_quat)) @ T
        # print(right_T1)
        # print(left_T1)
        # right_foot_bottom_center = right_foot_info[4] + right_T1.T
        # left_foot_bottom_center = left_foot_info[4] + left_T1.T

        # TODO ground contact detection using contact point
        if not self.checkGroundContact(side):  # no contact
            return [
                0.0, 0.0, 0.0], [
                0.0, 0.0, 0.0], False  # COP does not exists
        if Fz <= 1:  # z force
            return [
                0.0, 0.0, 0.0], [
                0.0, 0.0, 0.0], False  # COP does not exists
        d = link_frame_pos[2]  # z position
        px = (-My-Fx*d)/Fz
        py = (Mx - Fy*d)/Fz
        # position of Center of pressure
        pCOP = np.array(link_frame_pos) + np.array([px, py, -d])
        # force among the x,y,z axis of world frame
        contactForce = np.array([Fx, Fy, Fz])
        # print(len(footGroundContact))
        return pCOP, contactForce, True

    def calCOP(self):

        footGroundContact = []
        # TODO ground contact detection using contact point
        ankleRollContact = p.getContactPoints(
            self.r, self.plane, self.jointIdx['leftAnkleRoll'], -1)
        anklePitchContact = p.getContactPoints(
            self.r, self.plane, self.jointIdx['leftAnklePitch'], -1)
        footGroundContact.extend(ankleRollContact)
        footGroundContact.extend(anklePitchContact)
        left_contact_info = footGroundContact

        footGroundContact = []
        # TODO ground contact detection using contact point
        ankleRollContact = p.getContactPoints(
            self.r, self.plane, self.jointIdx['rightAnkleRoll'], -1)
        anklePitchContact = p.getContactPoints(
            self.r, self.plane, self.jointIdx['rightAnklePitch'], -1)
        footGroundContact.extend(ankleRollContact)
        footGroundContact.extend(anklePitchContact)
        right_contact_info = footGroundContact

        result = self.COP_filter_method(
            right_contact_info,
            left_contact_info)  # right then left
        self.COP_info = result
        result = self.COP_filter_method.getFilteredCOP()
        self.COP_info_filtered = result

        return result

    def calPelvisAcc(self):
        # TODO add pelvis acceleration
        base_pos, base_quat = p.getBasePositionAndOrientation(self.r)
        # base_orn = p.getEulerFromQuaternion(base_quat)
        R = self.transform(base_quat)
        R_i = np.linalg.inv(R)
        pelvis_vel, _ = p.getBaseVelocity(self.r)
        stop = len(self.pelvis_vel_his_array) - 2
        self.pelvis_vel_his_array[0:stop, :, None] = self.pelvis_vel_his_array[
            1:stop+1, :, None]  # shift element
        for i in range(len(self.pelvis_vel_his_array)-1):

            self.pelvis_vel_his_array[i] = self.pelvis_vel_his_array[i+1]

        self.pelvis_vel_his_array[len(
            self.pelvis_vel_his_array) - 1, :] = np.array(pelvis_vel)
        x = self.pelvis_vel_his_array[:, 0]
        # print(x)
        y = self.pelvis_vel_his_array[:, 1]

        z = self.pelvis_vel_his_array[:, 2]

        t = np.arange(len(self.pelvis_vel_his_array)) * self._dt  # time step
        x_acc = np.polyfit(t, x, 1)  # use slope of velocity as acceleration
        y_acc = np.polyfit(t, y, 1)
        z_acc = np.polyfit(t, z, 1)
        self.pelvis_acc = np.array([x_acc[0], y_acc[0], z_acc[0]])
        self.pelvis_acc_base = np.transpose(R_i @ self.pelvis_acc.transpose())

        self.pelvis_vel_his = np.array(pelvis_vel)
        self.pelvis_vel_his_base = np.transpose(
            R_i @self.pelvis_vel_his.transpose())

    def drawCOP(self):
        (COP, contact, rightFootCOP, rightFootContactForce, rightContact,
         leftFootCOP, leftFootContactForce, leftContact) = self.calCOP()
        if contact:
            p.addUserDebugLine(
                COP + [0, 0, -10],
                COP + [0, 0, 10],
                [0, 1, 0],
                5, 0.1)  # TODO rendering to draw support polygon
        if rightContact:
            p.addUserDebugLine(
                rightFootCOP + [0, 0, -10],
                rightFootCOP + [0, 0, 10],
                [1, 0, 0],
                5, 0.1)  # TODO rendering to draw support polygon
        if leftContact:
            p.addUserDebugLine(
                leftFootCOP + [0, 0, -10],
                leftFootCOP + [0, 0, 10],
                [0, 0, 1],
                5, 0.1)  # TODO rendering to draw support polygon

    def drawCOM(self):
        p.addUserDebugLine(
            self.COM_pos + np.array([0, 0, 2]),
            self.COM_pos + np.array([0, 0, -2]),
            [1, 0, 0],
            5, 0.1)  # TODO rendering to draw COM

    def getGroundContactPoint(self):
        footGroundContact = []

        if self.checkGroundContact('right'):
            ankleRollContact = p.getContactPoints(
                self.r, self.plane, self.jointIdx['rightAnkleRoll'], -1)
            anklePitchContact = p.getContactPoints(
                self.r, self.plane, self.jointIdx['rightAnklePitch'], -1)
            # use extend instead of append
            footGroundContact.extend(ankleRollContact)
            footGroundContact.extend(anklePitchContact)
        if self.checkGroundContact('left'):
            ankleRollContact = p.getContactPoints(
                self.r, self.plane, self.jointIdx['leftAnkleRoll'], -1)
            anklePitchContact = p.getContactPoints(
                self.r, self.plane, self.jointIdx['leftAnklePitch'], -1)
            footGroundContact.extend(ankleRollContact)
            footGroundContact.extend(anklePitchContact)
        # return a flag to indicate no contact point
        if len(footGroundContact) < 1:
            self.contact_points = np.array([[0.0, 0.0, 0.0]])
            return self.contact_points, False
        footGroundContactPoint = np.array(
            [data[5] for data in footGroundContact])

        points = footGroundContactPoint[:, 0:2]  # only use x and y coordinates
        # print(points)

        self.contact_points = points
        return self.contact_points, True

    def getSupportPolygon(self):
        footGroundContact = []

        if self.checkGroundContact('right'):
            ankleRollContact = p.getContactPoints(
                self.r, self.plane, self.jointIdx['rightAnkleRoll'], -1)
            anklePitchContact = p.getContactPoints(
                self.r, self.plane, self.jointIdx['rightAnklePitch'], -1)
            # use extend instead of append
            footGroundContact.extend(ankleRollContact)
            footGroundContact.extend(anklePitchContact)
        if self.checkGroundContact('left'):
            ankleRollContact = p.getContactPoints(
                self.r, self.plane, self.jointIdx['leftAnkleRoll'], -1)
            anklePitchContact = p.getContactPoints(
                self.r, self.plane, self.jointIdx['leftAnklePitch'], -1)
            footGroundContact.extend(ankleRollContact)
            footGroundContact.extend(anklePitchContact)

        footGroundContactPoint = np.array(
            [data[5] for data in footGroundContact])
        # return a flag to indicate no contact point
        if len(footGroundContactPoint) < 1:
            return np.array([[0.0, 0.0, 0.0]]), False

        from scipy.spatial import ConvexHull
        points = footGroundContactPoint[:, 0:2]  # only use x and y coordinates
        # print(points)

        self.contact_points = points

        if len(points) >= 4:
            hull = ConvexHull(points)  # only use x and y coordinates
            self.hull = points[hull.vertices, :]
        else:
            self.hull = points

        # print(self.hull)
        # self.drawSupportPolygon()
        return self.hull, True

    def drawSupportPolygon(self):
        hull = self.hull  # 2D
        if len(hull) <= 1:
            return

        support_polygon_center = np.zeros(np.shape(hull[0]))
        for i in range(len(hull)):
            if i >= len(hull) - 1:  # end point
                start = np.array([hull[i][0], hull[i][1], 0.0])
                end = np.array([hull[0][0], hull[0][1], 0.0])
            else:
                start = np.array([hull[i][0], hull[i][1], 0.0])
                end = np.array([hull[i + 1][0], hull[i + 1][1], 0.0])
            # TODO rendering to draw support polygon
            p.addUserDebugLine(start, end, [0, 0, 1], 10, 0.1)
            support_polygon_center += np.array(hull[i])

        support_polygon_center /= len(hull)
        support_polygon_center = np.array(
            [support_polygon_center[0], support_polygon_center[1], 0])

        p.addUserDebugLine(
            support_polygon_center + np.array([0, 0, 2]),
            support_polygon_center + np.array([0, 0, -2]),
            [0, 1, 0],
            5, 0.1)  # TODO rendering to draw COM
        return

    def jointReactionForce(self, side):
        assert side in ['left', 'right']
        ankle_joint_state = p.getJointState(
            self.r, self.jointIdx[side + 'AnklePitch'])
        torque = np.sqrt(
            ankle_joint_state[2][3] ** 2 +
            ankle_joint_state[2][4] ** 2 +
            ankle_joint_state[2][5] ** 2)  # total torque
        return torque

    def checkFall(self):
        base_pos, base_quat = p.getBasePositionAndOrientation(self.r)
        # chest_link_state = p.getLinkState(
        #   self.r, self.jointIdx['torsoRoll'],
        #   computeLinkVelocity=1)
        fall = False
        # if chest_link_state[0][2] < 0.5 or base_pos[2] < 0.5:
        # chest_z pelvis_z
        #            fall = True
        #        else:
        #            fall = False
        # TODO detect fall based on contact
        link = list(self.controllable_joints)  # create new list
        link.remove('leftAnklePitch')  # remove foot link from list
        link.remove('rightAnklePitch')
        link.remove('leftAnkleRoll')  # remove foot link from list
        link.remove('rightAnkleRoll')
        fall = False
        for key in link:
            if (len(p.getContactPoints(
                    self.r, self.plane, self.jointIdx[key], -1)) > 0):
                # print(key)
                fall = True
                break
        if base_pos[2] <= 0.3:  # TODO check fall criteria
            fall = True
        if self.COM_pos[2] <= 0.3:
            fall = True
        # COM_z = self.COM_pos_local[2] #not suitable for walking
        # COM_xy = np.linalg.norm([self.COM_pos_local[0],self.COM_pos_local[1]])
        # if COM_z<COM_xy:#out of friction cone
        #     fall = True
        return fall

    def setZeroOrderHoldNominalPose(self):
        for jointName in self.controllable_joints:
            p.setJointMotorControl2(self.r,
                                    self.jointIdx[jointName],
                                    p.POSITION_CONTROL,
                                    targetPosition=self.q_nom[jointName],
                                    force=self.u_max[jointName],
                                    )

    def calPDTorque(self, u):
        # calculate torque using self desinged PD controller
        final = np.isnan(u).any()

        # Set control
        torque_dict = dict()
        for i in range(self.nu):
            jointName = self.controlled_joints[i]
            self.action.update({jointName: u[i]})  # TODO
            if not final:
                if jointName in self.Kp:  # PD gains defined
                    torque = self.pd_controller[jointName].control(u[i], 0.0)
                    # torque = u[i]
                    torque = np.clip(
                        torque, -self.u_max[jointName],
                        self.u_max[jointName])
                    torque_dict.update({jointName: torque})

        return torque_dict

    def setPDPositionControl(self, torque_dict, u):
        final = np.isnan(u).any()

        # Set control
        for i in range(self.nu):
            jointName = self.controlled_joints[i]
            self.action.update({jointName: u[i]})  # TODO
            if not final:
                p.setJointMotorControl2(
                    self.r, self.jointIdx[jointName],
                    targetPosition=u[i],
                    targetVelocity=0,
                    maxVelocity=self.v_max[jointName],
                    force=abs(torque_dict[jointName]),
                    controlMode=p.POSITION_CONTROL,
                    # positionGain=self.Kp[jointName],
                    # velocityGain=self.Kd[jointName],
                )

    def setPDVelocityControl(self, torque_dict):  # set control
        for jointName, torque in torque_dict.items():
            p.setJointMotorControl2(
                self.r,
                self.jointIdx[jointName],
                targetVelocity=np.sign(torque) *
                self.v_max[jointName],
                force=np.abs(torque),
                controlMode=p.VELOCITY_CONTROL,
                )

    def setPDTorqueControl(self, torque_dict):
        for jointName, torque in torque_dict.items():
            p.setJointMotorControl2(
                self.r, self.jointIdx[jointName],
                targetVelocity=np.sign(torque) * self.v_max[jointName],
                force=0, controlMode=p.VELOCITY_CONTROL,)
            p.setJointMotorControl2(
                self.r, self.jointIdx[jointName],
                force=torque,
                controlMode=p.TORQUE_CONTROL,
            )

    def setDefaultControl(self, u):
        final = np.isnan(u).any()

        # Set control
        for i in range(self.nu):
            jointName = self.controlled_joints[i]
            self.action.update({jointName: u[i]})  # TODO
            if not final:
                if self.bullet_default_PD:
                    p.setJointMotorControl2(
                        self.r, self.jointIdx[jointName],
                        targetPosition=u[i],
                        targetVelocity=0, maxVelocity=self.v_max[jointName],
                        force=self.u_max[jointName],
                        controlMode=p.POSITION_CONTROL,)

                else:
                    if self.Kp.get(jointName) is None:  # PD gains not defined
                        p.setJointMotorControl2(
                            self.r, self.jointIdx[jointName],
                            targetPosition=u[i],
                            targetVelocity=0, maxVelocity=self.v_max
                            [jointName],
                            force=self.u_max[jointName],
                            controlMode=p.POSITION_CONTROL,)

        # Set zero order hold for uncontrolled joints
        for jointName in self.uncontrolled_joints:
            p.setJointMotorControl2(self.r,
                                    self.jointIdx[jointName],
                                    p.POSITION_CONTROL,
                                    targetPosition=self.q_nom[jointName],
                                    force=self.u_max[jointName],
                                    maxVelocity=self.v_max[jointName])
        return

    def setControl(self, u):
        final = np.isnan(u).any()

        # Set control
        for i in range(self.nu):
            jointName = self.controlled_joints[i]
            self.action.update({jointName: u[i]})  # TODO
            if not final:
                if self.bullet_default_PD:
                    p.setJointMotorControl2(
                        self.r, self.jointIdx[jointName],
                        targetPosition=u[i],
                        targetVelocity=0,
                        maxVelocity=self.v_max[jointName],
                        force=self.u_max[jointName],
                        controlMode=p.POSITION_CONTROL,
                        # positionGain=self.Kp[jointName],
                        # velocityGain=self.Kd[jointName],
                    )

                else:
                    if jointName in self.Kp:  # PD gains defined
                        torque = self.pd_controller[jointName].control(u[i], 0)
                        # torque = u[i]
                        torque = np.clip(
                            torque, -self.u_max[jointName],
                            self.u_max[jointName])
                        # print(torque)
                        p.setJointMotorControl2(
                            self.r, self.jointIdx[jointName],
                            targetVelocity=(np.sign(torque)
                                            * self.v_max[jointName]),
                            force=np.abs(torque),
                            # force=0,#disable motor
                            controlMode=p.VELOCITY_CONTROL,
                            # velocityGains=0.0001,
                        )
                        # p.setJointMotorControl2(
                        #     self.r, self.jointIdx[jointName],
                        #     force=torque,
                        #     controlMode=p.TORQUE_CONTROL,
                        # )

                    else:  # PD gains not defined
                        p.setJointMotorControl2(
                            self.r, self.jointIdx[jointName],
                            targetPosition=u[i],
                            targetVelocity=0,
                            maxVelocity=self.v_max[jointName],
                            force=self.u_max[jointName],
                            controlMode=p.POSITION_CONTROL,
                            # positionGain=self.Kp[jointName],
                            # velocityGain=self.Kd[jointName],
                        )
        # Set zero order hold for uncontrolled joints
        for jointName in self.uncontrolled_joints:
            p.setJointMotorControl2(self.r,
                                    self.jointIdx[jointName],
                                    p.POSITION_CONTROL,
                                    targetPosition=self.q_nom[jointName],
                                    force=self.u_max[jointName],
                                    maxVelocity=self.v_max[jointName])

    def applyForceOnPelvis(self, force, pos):
        pos = np.array(pos) + np.array([0, 0.0035, 0])
        p.applyExternalForce(self.r, -1, forceObj=force,
                             posObj=pos, flags=p.LINK_FRAME)

    def drawBaseFrame(self):
        # This may not be the Pelvis CoM position _exactly_ but should be fine,
        # otherwise can apply local transformation
        base_pos, base_quat = p.getBasePositionAndOrientation(self.r)
        # base_orn = p.getEulerFromQuaternion(base_quat)

        # Gravitational acceleration acts as a reference for pitch and roll but
        # not for yaw.

        # overwrite base_pos with the calculated base COM position
        # base_pos is tuple, tuple is immutable
        # change tuple to array
        base_pos = np.array(base_pos)
        base_pos[0] = self.linkCOMPos['pelvisBase'][0]
        base_pos[1] = self.linkCOMPos['pelvisBase'][1]
        base_pos[2] = self.linkCOMPos['pelvisBase'][2]
        # base_pos = self.linkCOMPos['pelvisBase']

        # Rz = self.rotZ(base_orn[2])
        # Rz_i = np.linalg.inv(Rz)
        R = self.transform(base_quat)
        # R_i = np.linalg.inv(R)

        x_axis = np.array([[0.8, 0, 0]])
        y_axis = np.array([[0, 0.8, 0]])
        z_axis = np.array([[0, 0, 0.8]])

        x_axis_base = np.transpose(R @ x_axis.transpose())
        y_axis_base = np.transpose(R @ y_axis.transpose())
        z_axis_base = np.transpose(R @ z_axis.transpose())

        p.addUserDebugLine(
            np.array(base_pos),
            np.array(base_pos) + x_axis_base[0],
            [1, 0, 0],
            5, 0.1)  # x axis
        p.addUserDebugLine(
            np.array(base_pos),
            np.array(base_pos) + y_axis_base[0],
            [0, 1, 0],
            5, 0.1)  # y axis
        p.addUserDebugLine(
            np.array(base_pos),
            np.array(base_pos) + z_axis_base[0],
            [0, 0, 1],
            5, 0.1)  # z axis

        return

    def drawBaseYawFrame(self):
        # This may not be the Pelvis CoM position _exactly_ but should be fine,
        # otherwise can apply local transformation
        base_pos, base_quat = p.getBasePositionAndOrientation(self.r)
        base_orn = p.getEulerFromQuaternion(base_quat)

        # Gravitational acceleration acts as a reference for pitch and roll but
        # not for yaw.

        # overwrite base_pos with the calculated base COM position
        # base_pos is tuple, tuple is immutable
        # change tuple to array
        base_pos = np.array(base_pos)
        base_pos[0] = self.linkCOMPos['pelvisBase'][0]
        base_pos[1] = self.linkCOMPos['pelvisBase'][1]
        base_pos[2] = self.linkCOMPos['pelvisBase'][2]
        # base_pos = self.linkCOMPos['pelvisBase']

        Rz = self.rotZ(base_orn[2])
        # Rz_i = np.linalg.inv(Rz)
        # R = self.transform(base_quat)
        # R_i = np.linalg.inv(R)

        x_axis = np.array([[1.5, 0, 0]])
        y_axis = np.array([[0, 1.5, 0]])
        z_axis = np.array([[0, 0, 1.5]])

        x_axis_base = np.transpose(Rz @ x_axis.transpose())
        y_axis_base = np.transpose(Rz @ y_axis.transpose())
        z_axis_base = np.transpose(Rz @ z_axis.transpose())

        origin = np.array([0.5, 0, 0.3])

        p.addUserDebugLine(
            np.array(origin),
            np.array(origin) + x_axis_base[0],
            [1, 0, 0],
            5, 0.1)  # x axis
        p.addUserDebugLine(
            np.array(origin),
            np.array(origin) + y_axis_base[0],
            [0, 1, 0],
            5, 0.1)  # y axis
        p.addUserDebugLine(
            np.array(origin),
            np.array(origin) + z_axis_base[0],
            [0, 0, 1],
            5, 0.1)  # z axis

        return

    def drawSkeletonYaw(self):
        # This may not be the Pelvis CoM position _exactly_ but should be fine,
        # otherwise can apply local transformation
        base_pos, base_quat = p.getBasePositionAndOrientation(self.r)
        base_orn = p.getEulerFromQuaternion(base_quat)

        # Gravitational acceleration acts as a reference for pitch and roll but
        # not for yaw.

        # overwrite base_pos with the calculated base COM position
        # base_pos is tuple, tuple is immutable
        # change tuple to array
        base_pos = np.array(base_pos)
        base_pos[0] = self.linkCOMPos['pelvisBase'][0]
        base_pos[1] = self.linkCOMPos['pelvisBase'][1]
        base_pos[2] = self.linkCOMPos['pelvisBase'][2]
        # base_pos = self.linkCOMPos['pelvisBase']

        Rz = self.rotZ(base_orn[2])
        Rz_i = np.linalg.inv(Rz)
        # R = self.transform(base_quat)
        # R_i = np.linalg.inv(R)

        # chest
        chest_link_state = p.getLinkState(
            self.r, self.jointIdx['torsoRoll'],
            computeLinkVelocity=1)
        chest_link_dis = np.array(chest_link_state[0]) - np.array(base_pos)
        chest_link_dis.resize(1, 3)
        # chest_link_dis_base = np.transpose(R_i @ chest_link_dis.transpose())
        chest_link_dis_yaw = np.transpose(Rz_i @ chest_link_dis.transpose())

        left_foot_link_state = p.getLinkState(
            self.r, self.jointIdx['leftAnkleRoll'],
            computeLinkVelocity=0)
        left_foot_link_dis = np.array(left_foot_link_state[0]) - base_pos
        left_foot_link_dis.resize(1, 3)
        # left_foot_link_dis_base = np.transpose(
        #     R_i @left_foot_link_dis.transpose())
        left_foot_link_dis_yaw = np.transpose(
            Rz_i @left_foot_link_dis.transpose())
        # left_foot_link_dis_yaw = left_foot_link_dis

        right_foot_link_state = p.getLinkState(
            self.r, self.jointIdx['rightAnkleRoll'],
            computeLinkVelocity=0)
        right_foot_link_dis = np.array(right_foot_link_state[0]) - base_pos
        right_foot_link_dis.resize(1, 3)
        # right_foot_link_dis_base = np.transpose(
        #    R_i @right_foot_link_dis.transpose())
        right_foot_link_dis_yaw = np.transpose(
            Rz_i @right_foot_link_dis.transpose())
        # right_foot_link_dis_yaw = right_foot_link_dis

        left_elbow_link_state = p.getLinkState(
            self.r, self.jointIdx['leftElbowPitch'],
            computeLinkVelocity=0)
        left_elbow_link_dis = np.array(left_elbow_link_state[0]) - base_pos
        left_elbow_link_dis.resize(1, 3)
        # left_elbow_link_dis_base = np.transpose(
        #     R_i @left_elbow_link_dis.transpose())
        left_elbow_link_dis_yaw = np.transpose(
            Rz_i @left_elbow_link_dis.transpose())

        right_elbow_link_state = p.getLinkState(
            self.r, self.jointIdx['rightElbowPitch'],
            computeLinkVelocity=0)
        right_elbow_link_dis = np.array(right_elbow_link_state[0]) - base_pos
        right_elbow_link_dis.resize(1, 3)
        # right_elbow_link_dis_base = np.transpose(
        #     R_i @right_elbow_link_dis.transpose())
        right_elbow_link_dis_yaw = np.transpose(
            Rz_i @right_elbow_link_dis.transpose())

        base_pos[0] = 1
        base_pos[1] = 0

        orientation = np.array([[0.5, 0, 0]])
        # orientation_base = np.transpose(R_i @ orientation.transpose())
        # np.transpose(Rz_i @ orientation.transpose())
        orientation_yaw = orientation

        p.addUserDebugLine(
            base_pos, base_pos + orientation_yaw[0],
            [0, 1, 0],
            3, 1)  # pelvis to chest
        p.addUserDebugLine(
            base_pos, base_pos + chest_link_dis_yaw[0],
            [1, 0, 0],
            3, 1)  # pelvis to chest
        p.addUserDebugLine(
            base_pos, base_pos + left_foot_link_dis_yaw[0],
            [1, 0, 0],
            3, 1)  # pelvis to left foot
        p.addUserDebugLine(
            base_pos, base_pos + right_foot_link_dis_yaw[0],
            [1, 0, 0],
            3, 1)  # pelvis to right foot
        p.addUserDebugLine(
            base_pos + chest_link_dis_yaw[0],
            base_pos + left_elbow_link_dis_yaw[0],
            [1, 0, 0],
            3, 1)  # pelvis to left foot
        p.addUserDebugLine(
            base_pos + chest_link_dis_yaw[0],
            base_pos + right_elbow_link_dis_yaw[0],
            [1, 0, 0],
            3, 1)  # pelvis to right foot
        return

    def drawSkeletonBase(self):
        # This may not be the Pelvis CoM position _exactly_ but should be fine,
        # otherwise can apply local transformation
        base_pos, base_quat = p.getBasePositionAndOrientation(self.r)
        # base_orn = p.getEulerFromQuaternion(base_quat)

        # Gravitational acceleration acts as a reference for pitch and roll but
        # not for yaw.

        # overwrite base_pos with the calculated base COM position
        # base_pos is tuple, tuple is immutable
        # change tuple to array
        base_pos = np.array(base_pos)
        base_pos[0] = self.linkCOMPos['pelvisBase'][0]
        base_pos[1] = self.linkCOMPos['pelvisBase'][1]
        base_pos[2] = self.linkCOMPos['pelvisBase'][2]
        # base_pos = self.linkCOMPos['pelvisBase']

        # Rz = self.rotZ(base_orn[2])
        # Rz_i = np.linalg.inv(Rz)
        R = self.transform(base_quat)
        R_i = np.linalg.inv(R)

        # chest
        chest_link_state = p.getLinkState(
            self.r, self.jointIdx['torsoRoll'],
            computeLinkVelocity=1)
        chest_link_dis = np.array(chest_link_state[0]) - np.array(base_pos)
        chest_link_dis.resize(1, 3)
        chest_link_dis_base = np.transpose(R_i @ chest_link_dis.transpose())
        # chest_link_dis_yaw = np.transpose(Rz_i @ chest_link_dis.transpose())

        left_foot_link_state = p.getLinkState(
            self.r, self.jointIdx['leftAnkleRoll'],
            computeLinkVelocity=0)
        left_foot_link_dis = np.array(left_foot_link_state[0]) - base_pos
        left_foot_link_dis.resize(1, 3)
        left_foot_link_dis_base = np.transpose(
            R_i @left_foot_link_dis.transpose())
        # left_foot_link_dis_yaw = np.transpose(
        #     Rz_i @left_foot_link_dis.transpose())
        # left_foot_link_dis_yaw = left_foot_link_dis

        right_foot_link_state = p.getLinkState(
            self.r, self.jointIdx['rightAnkleRoll'],
            computeLinkVelocity=0)
        right_foot_link_dis = np.array(right_foot_link_state[0]) - base_pos
        right_foot_link_dis.resize(1, 3)
        right_foot_link_dis_base = np.transpose(
            R_i @right_foot_link_dis.transpose())
        # right_foot_link_dis_yaw = np.transpose(
        #     Rz_i @right_foot_link_dis.transpose())
        # right_foot_link_dis_yaw = right_foot_link_dis

        left_elbow_link_state = p.getLinkState(
            self.r, self.jointIdx['leftElbowPitch'],
            computeLinkVelocity=0)
        left_elbow_link_dis = np.array(left_elbow_link_state[0]) - base_pos
        left_elbow_link_dis.resize(1, 3)
        left_elbow_link_dis_base = np.transpose(
            R_i @left_elbow_link_dis.transpose())
        # left_elbow_link_dis_yaw = np.transpose(
        #     Rz_i @left_elbow_link_dis.transpose())

        right_elbow_link_state = p.getLinkState(
            self.r, self.jointIdx['rightElbowPitch'],
            computeLinkVelocity=0)
        right_elbow_link_dis = np.array(right_elbow_link_state[0]) - base_pos
        right_elbow_link_dis.resize(1, 3)
        right_elbow_link_dis_base = np.transpose(
            R_i @right_elbow_link_dis.transpose())
        # right_elbow_link_dis_yaw = np.transpose(
        #     Rz_i @right_elbow_link_dis.transpose())

        base_pos[0] = 1
        base_pos[1] = 0
        base_pos[2] = 1.104

        orientation = np.array([[0.5, 0, 0]])
        # np.transpose(R_i @ orientation.transpose())
        orientation_base = orientation
        # np.transpose(Rz_i @ orientation.transpose())
        # orientation_yaw = orientation

        p.addUserDebugLine(
            base_pos, base_pos + orientation_base[0],
            [0, 1, 0],
            3, 1)  # pelvis to chest
        p.addUserDebugLine(
            base_pos, base_pos + chest_link_dis_base[0],
            [1, 0, 0],
            3, 1)  # pelvis to chest
        p.addUserDebugLine(
            base_pos, base_pos + left_foot_link_dis_base[0],
            [1, 0, 0],
            3, 1)  # pelvis to left foot
        p.addUserDebugLine(
            base_pos, base_pos + right_foot_link_dis_base[0],
            [1, 0, 0],
            3, 1)  # pelvis to right foot
        p.addUserDebugLine(
            base_pos + chest_link_dis_base[0],
            base_pos + left_elbow_link_dis_base[0],
            [1, 0, 0],
            3, 1)  # pelvis to left foot
        p.addUserDebugLine(
            base_pos + chest_link_dis_base[0],
            base_pos + right_elbow_link_dis_base[0],
            [1, 0, 0],
            3, 1)  # pelvis to right foot
        return

    def testKinematic(self):
        base_pos, base_quat = p.getBasePositionAndOrientation(self.r)
        # base_orn = p.getEulerFromQuaternion(base_quat)

        R = self.transform(base_quat)
        T0 = np.eye(4)
        T0[0:3, 0:3] = R
        T0[0:3, 3] = np.array(base_pos)

        print(T0)

        # base_pos = self.linkCOMPos['pelvisBase']

        # Rz = self.rotZ(base_orn[2])
        # Rz_i = np.linalg.inv(Rz)

        T_dict = dict()
        joint_pos_dict = dict()
        COM_dict = dict()
        COM_ref_dict = dict()
        T_dict.update({-1: np.array(T0)})
        joint_pos_dict.update({-1: np.array(base_pos)})
        COM_dict.update({-1: np.array(base_pos)})
        COM_ref_dict.update({-1: np.array(base_pos)})

        for name in self.controllable_joints:
            jointNo = self.jointIdx[name]
            joint_name = name
            # for jointNo in range(p.getNumJoints(self.r)):
            # info = p.getJointInfo(self.r, jointNo)
            # joint_name = info[1].decode("utf-8")
            #
            # self.jointIdx.update({joint_name: info[0]})
            # self.jointNameIdx.update({info[0]: joint_name})

            link_state = p.getLinkState(self.r, jointNo, computeLinkVelocity=0)
            inertiaFramePos = link_state[2]
            inertiaFrameOrn = link_state[3]

            joint_info = p.getJointInfo(self.r, jointNo)
            jointAxis = joint_info[13]
            parentFramePos = joint_info[14]
            parentFrameOrn = joint_info[15]
            parentIdx = joint_info[16]

            joint_state = p.getJointState(self.r, jointNo)
            jointPosition = joint_state[0]

            jointAngle = np.array(jointAxis) * jointPosition
            R_frame = self.transform(parentFrameOrn)

            R_joint = self.rotZ(
                jointAngle[2]) @self.rotY(
                jointAngle[1]) @self.rotX(
                jointAngle[0])
            R = R_frame @ R_joint

            T_link = np.eye(4)
            T_link[0:3, 0:3] = R
            T_link[0:3, 3] = np.array(parentFramePos)
            temp = T_dict[parentIdx] @ T_link

            jointpos = temp[0:3, 3]

            R_inertia = self.transform(inertiaFrameOrn)
            T_inertia = np.eye(4)
            T_inertia[0:3, 0:3] = R_inertia
            T_inertia[0:3, 3] = np.array(inertiaFramePos)

            T = T_dict[parentIdx] @ T_link @ T_inertia

            COM = T[0:3, 3]
            print(joint_name, np.array(link_state[0]) - COM)

            joint_pos_dict.update({jointNo: jointpos})
            T_dict.update({jointNo: T})
            COM_dict.update({jointNo: COM})
            COM_ref_dict.update({jointNo: np.array(link_state[0])})

            start = COM_ref_dict[parentIdx]
            end = np.array(link_state[0])
            p.addUserDebugLine(start, end, [1, 0, 0], 1, 1)
            start = COM_dict[parentIdx]
            end = COM
            p.addUserDebugLine(start, end, [0, 0, 1], 1, 1)
        return
