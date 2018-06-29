import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import os
import pybullet as p
from scipy.spatial import ConvexHull

from valkyrie.envs.filter import ButterworthFilter
from valkyrie.envs.pd_controller import PDController
from valkyrie.envs.cop_calculator import COPCalculator


CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

KP_DEFAULTS = {
    "leftAnklePitch": 3000,  # -0.71
    "leftAnkleRoll": 300,  # -0.71
    "leftElbowPitch": 200,
    "leftHipPitch": 2000,  # -0.49
    "leftHipRoll": 1000,  # -0.49
    "leftHipYaw": 500,
    "leftKneePitch": 2000,  # 1.205
    "leftShoulderPitch": 700,
    "leftShoulderRoll": 1500,
    "leftShoulderYaw": 200,
    "rightAnklePitch": 3000,  # -0.71
    "rightAnkleRoll": 300,  # -0.71
    "rightElbowPitch": 200,
    "rightHipPitch": 2000,  # -0.49
    "rightHipRoll": 1000,  # -0.49
    "rightHipYaw": 500,
    "rightKneePitch": 2000,  # 1.205
    "rightShoulderPitch": 700,
    "rightShoulderRoll": 1500,
    "rightShoulderYaw": 200,
    "torsoPitch": 4500,
    "torsoRoll": 4500,
    "torsoYaw": 4500,
}

KD_DEFAULTS = {
    "leftAnklePitch": 3,  # -0.71
    "leftAnkleRoll": 3,  # -0.71
    "leftElbowPitch": 5,
    "leftHipPitch": 30,  # -0.49
    "leftHipRoll": 30,  # -0.49
    "leftHipYaw": 20,
    "leftKneePitch": 30,  # 1.205
    "leftShoulderPitch": 10,
    "leftShoulderRoll": 30,
    "leftShoulderYaw": 2,
    "rightAnklePitch": 3,  # -0.71
    "rightAnkleRoll": 3,  # -0.71
    "rightElbowPitch": 5,
    "rightHipPitch": 30,  # -0.49
    "rightHipRoll": 30,  # -0.49
    "rightHipYaw": 20,
    "rightKneePitch": 30,  # 1.205
    "rightShoulderPitch": 10,
    "rightShoulderRoll": 30,
    "rightShoulderYaw": 2,
    "torsoPitch": 30,
    "torsoRoll": 30,
    "torsoYaw": 30,
}


class ValkyrieEnvBase(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(
            self,
            controlled_joints=[
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
                "torsoPitch"],
            max_time=16,  # in seconds
            initial_gap_time=0.01,  # in seconds
            render=True,
            pd_freq=500.0,
            physics_freq=1000.0,
            Kp=KP_DEFAULTS,
            Kd=KD_DEFAULTS,
            use_bullet_default_pd=True,
            logFileName=CURRENT_DIR):
        self.seed()
        self.g = 9.806651
        ##########
        # define robot_id joints and default positions
        self.controllable_joints = np.array([
            "leftAnklePitch",
            "leftAnkleRoll",
            "leftElbowPitch",
            "leftHipPitch",
            "leftHipRoll",
            "leftHipYaw",
            "leftKneePitch",
            "leftShoulderPitch",
            "leftShoulderRoll",
            "leftShoulderYaw",
            "lowerNeckPitch",
            "neckYaw",
            "rightAnklePitch",
            "rightAnkleRoll",
            "rightElbowPitch",
            "rightHipPitch",
            "rightHipRoll",
            "rightHipYaw",
            "rightKneePitch",
            "rightShoulderPitch",
            "rightShoulderRoll",
            "rightShoulderYaw",
            "torsoPitch",
            "torsoRoll",
            "torsoYaw",
            "upperNeckPitch"])
        self.controlled_joints = np.array(controlled_joints)
        self.uncontrolled_joints = np.array(
            [joint for joint in self.controllable_joints
             if joint not in self.controlled_joints])
        self.default_joint_config = {
            "leftAnklePitch": 0.0,  # -0.71
            "leftAnkleRoll": 0.0,
            "leftElbowPitch": -0.785398163397,
            "leftHipPitch": 0.0,  # -0.49
            "leftHipRoll": 0.0,
            "leftHipYaw": 0.0,
            "leftKneePitch": 0.0,  # 1.205
            "leftShoulderPitch": 0.300196631343,
            "leftShoulderRoll": -1.25,
            "leftShoulderYaw": 0.0,
            "lowerNeckPitch": 0.0,
            "neckYaw": 0.0,
            "rightAnklePitch": 0.0,  # -0.71
            "rightAnkleRoll": 0.0,
            "rightElbowPitch": 0.785398163397,
            "rightHipPitch": 0.0,  # -0.49
            "rightHipRoll": 0.0,
            "rightHipYaw": 0.0,
            "rightKneePitch": 0.0,  # 1.205
            "rightShoulderPitch": 0.300196631343,
            "rightShoulderRoll": 1.25,
            "rightShoulderYaw": 0.0,
            "torsoPitch": 0.0,
            "torsoRoll": 0.0,
            "torsoYaw": 0.0,
            "upperNeckPitch": 0.0}
        self.default_joint_positions = np.array(
            [self.default_joint_config[j] for j in self.controllable_joints])
        self.default_base_pos = np.array([0, 0, 1.175])  # straight
        self.default_base_orn = np.array([0, 0, 0, 1])  # x, y, z, w

        ##########
        # Initialise parameters for simulation and PD controller
        self.use_bullet_default_pd = use_bullet_default_pd
        self.pd_freq = pd_freq
        self.physics_freq = physics_freq
        self.Kp = Kp  # proportional gain
        self.Kd = Kd  # derivative gain
        self.max_time = max_time
        self.max_steps = self.max_time * self.pd_freq  # reset timestep for PD
        self.initial_gap_time = initial_gap_time
        self.initial_gap_steps = self.initial_gap_time * self.pd_freq
        self._action_repeats = int(self.physics_freq / self.pd_freq)
        self._dt_physics = 1.0 / self.physics_freq  # PD control loop timestep
        self._dt_filter = 1.0 / self.pd_freq
        # various things we need to track / update over time
        self.action = dict()
        self.pd_controller = dict()
        self.pd_torque_adjusted = dict()
        self.pd_torque_filtered = dict()
        self.pd_torque_unfiltered = dict()
        self.last_torque_applied = dict()
        self.last_torque_target = dict()
        for joint in self.controlled_joints:
            self.action[joint] = 0.0
            self.last_torque_applied[joint] = 0.0
            self.last_torque_target[joint] = 0.0
        # Centre of mass. COM local coordinateisare wrt centre of mass of foot.
        # robot operates solely on the sagittal plane, the orientation of global
        # frame and local frame is aligned
        self.COM_pos = np.zeros(3)  # global COM
        self.COM_vel = np.zeros(3)
        self.COM_pos_local = np.zeros(3)  # local COM
        self.COM_pos_local_filtered = np.zeros(3)
        self.COM_pos_local_surrogate = np.zeros(3)
        self.support_polygon_centre = np.zeros(3)
        self.support_polygon_centre_surrogate = np.zeros(3)
        # Pelvis acceleration
        self.pelvis_acc_gap_step = 10
        self.pelvis_acc = np.zeros(3)
        self.pelvis_acc_base = np.zeros(3)
        self.pelvis_vel_history_array = np.zeros((self.pelvis_acc_gap_step, 3))
        self.pelvis_vel_history = np.zeros(3)
        self.pelvis_vel_history_base = np.zeros(3)

        ##########
        # set torque limits
        self.u_max = {
            "leftAnklePitch": 205,
            "leftAnkleRoll": 205,
            "leftElbowPitch": 65,
            "leftForearmYaw": 26,
            "leftHipPitch": 350,
            "leftHipRoll": 350,
            "leftHipYaw": 190,
            "leftKneePitch": 350,
            "leftShoulderPitch": 190,
            "leftShoulderRoll": 190,
            "leftShoulderYaw": 65,
            "leftWristPitch": 14,
            "leftWristRoll": 14,
            "lowerNeckPitch": 50,
            "neckYaw": 50,
            "rightAnklePitch": 205,
            "rightAnkleRoll": 205,
            "rightElbowPitch": 65,
            "rightForearmYaw": 26,
            "rightHipPitch": 350,
            "rightHipRoll": 350,
            "rightHipYaw": 190,
            "rightKneePitch": 350,
            "rightShoulderPitch": 190,
            "rightShoulderRoll": 190,
            "rightShoulderYaw": 65,
            "rightWristPitch": 14,
            "rightWristRoll": 14,
            "torsoPitch": 150,
            "torsoRoll": 150,
            "torsoYaw": 190,
            "upperNeckPitch": 50}
        # set velocity limits
        self.v_max = {
            "leftAnklePitch": 11,
            "leftAnkleRoll": 11,
            "leftElbowPitch": 11.5,
            "leftHipPitch": 6.11,
            "leftHipRoll": 7,
            "leftHipYaw": 5.89,
            "leftKneePitch": 6.11,
            "leftShoulderPitch": 5.89,
            "leftShoulderRoll": 5.89,
            "leftShoulderYaw": 11.5,
            "lowerNeckPitch": 5,
            "neckYaw": 5,
            "rightAnklePitch": 11,
            "rightAnkleRoll": 11,
            "rightElbowPitch": 11.5,
            "rightHipPitch": 6.11,
            "rightHipRoll": 7,
            "rightHipYaw": 5.89,
            "rightKneePitch": 6.11,
            "rightShoulderPitch": 5.89,
            "rightShoulderRoll": 5.89,
            "rightShoulderYaw": 11.5,
            "torsoPitch": 9,
            "torsoRoll": 9,
            "torsoYaw": 5.89,
            "upperNeckPitch": 5}

        ##########
        # Define observation and action spaces
        self.num_states = 51
        MAXIMUM_FLOAT = np.finfo(np.float32).max
        observation_high = np.array([MAXIMUM_FLOAT * self.num_states])
        self.observation_space = spaces.Box(-observation_high, observation_high,
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.controlled_joints))

        ##########
        # Initialise pybullet simulation
        self.steps_taken = 0
        self._render = render
        if self._render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        self._setup_simulation(self.default_base_pos, self.default_base_orn)

    def disconnect(self):
        """Disconnect from physics simulator."""
        p.disconnect()
        self.close()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(
            self,
            Kp=KP_DEFAULTS,
            Kd=KD_DEFAULTS,
            base_pos=None,
            base_orn=None,
            fixed_base=False):
        self.Kp = Kp  # proportional gain
        self.Kd = Kd  # derivative gain
        self._setup_simulation(base_pos, base_orn, fixed_base)
        self.steps_taken = 0
        self._observation = self.get_extended_observation()
        return self._observation

    def _setup_simulation(self, base_pos, base_orn, fixed_base=False):
        """Configure pybullet physics simulator."""
        if base_pos is None:
            base_pos = self.default_base_pos
        if base_orn is None:
            base_orn = self.default_base_orn
        self._setup_filter()
        # Physics engine parameter default solver iteration = 50
        p.setPhysicsEngineParameter(numSolverIterations=25, erp=0.2)
        p.resetSimulation()
        p.setRealTimeSimulation(0)
        p.setGravity(0, 0, -self.g)
        p.setTimeStep(self._dt_physics)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        plane_urdf = os.path.join(CURRENT_DIR, "assets", "plane", "plane.urdf")
        self.plane_id = p.loadURDF(plane_urdf, basePosition=[0, 0, 0],
                                   useFixedBase=True)
        robot_urdf = os.path.join(
            CURRENT_DIR,
            "assets",
            "valkyrie_bullet_mass_sims"
            "_modified_foot_collision_box"
            "_modified_self_collision"
            ".urdf")
        self.robot_id = p.loadURDF(
            fileName=robot_urdf,
            basePosition=base_pos,
            baseOrientation=base_orn,
            flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION,
            useFixedBase=fixed_base)
        self._setup_camera()
        self._setup_joint_mapping()
        self._set_dynamics()
        self.reset_joint_states(base_pos, base_orn)
        self.set_zero_order_hold_nominal_pose()
        self.calculate_link_masses()

        # TODO test joint reaction force torque
        p.enableJointForceTorqueSensor(
            self.robot_id, self.joint_idx['leftAnkleRoll'], True)
        p.enableJointForceTorqueSensor(
            self.robot_id, self.joint_idx['rightAnkleRoll'], True)

        for _ in range(int(self.initial_gap_steps)):  # PD loop time steps
            for _ in range(self._action_repeats):
                p.stepSimulation()
            # update information
            self.calculate_link_COM_position()
            self.calculate_link_COM_velocity()
            self.calculate_COM_position()
            self.calculate_COM_velocity()
            self.calculate_pelvis_acc()
            self.calculate_COP()
            self.initialise_filtering()
            self.perform_filtering()
        self.calculate_ground_contact_points()
        # self.get_support_polygon()
        for joint in self.controlled_joints:
            joint_state = p.getJointState(self.robot_id, self.joint_idx[joint])
            self.last_torque_applied[joint] = joint_state[3]

    def step(self, action, pelvis_push_force=[0, 0, 0]):
        """Take action, get observation, reward, done (bool), info (empty)."""
        torque_dict = self.calculate_PD_torque(action)
        # self.set_control(action)
        # higher frequency for physics simulation
        for i in range(self._action_repeats):
            # displacement from COM to the centre of the pelvis
            p.applyExternalForce(
                self.robot_id,
                -1,  # base
                forceObj=pelvis_push_force,
                posObj=[0, 0.0035, 0],
                flags=p.LINK_FRAME)
            for joint, torque in torque_dict.items():
                self.pd_torque_filtered[joint] = self.pd_controller[joint].u_kal
                self.pd_torque_unfiltered[joint] = \
                    self.pd_controller[joint].u_raw
                self.pd_torque_adjusted[joint] = self.pd_controller[joint].u_adj
            # self.set_control(action)
            self.set_PD_velocity_control(torque_dict)
            p.stepSimulation()  # one simulation step

        # update information
        self.steps_taken += 1
        self.calculate_COM_position()
        self.calculate_COM_velocity()
        self.calculate_link_COM_position()
        self.calculate_link_COM_velocity()
        self.calculate_COP()
        self.calculate_ground_contact_points()
        self.calculate_pelvis_acc()

        self.perform_filtering()
        self.last_torque_target.update(torque_dict)
        for joint in self.controlled_joints:  # TODO
            joint_state = p.getJointState(self.robot_id, self.joint_idx[joint])
            self.last_torque_applied[joint] = joint_state[3]

        self._observation = self.get_extended_observation()  # filtered
        reward, _ = self.reward()  # balancing
        done = self.is_fallen()
        return self._observation, reward, done, {}

    def reward(self):
        """Return reward for current state and terms used in the reward.

        Returns
        -------
        reward : float
        reward_term : dict
            {name: value} of all elements summed to make the reward
        """
        raise NotImplementedError()

    def _set_dynamics(
            self,
            lateralFriction=1.0,
            spinningFriction=0.03,
            rollingFriction=0.03,
            restitution=0.0,
            linearDamping=0.04,
            angularDamping=0.04):
        """Set joint damping, and plane and foot friction and restitution."""
        # set damping
        for joint_id in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, joint_id)
            p.changeDynamics(self.robot_id, info[0])
        # set plane_id and ankle friction and restitution
        for joint in ('rightAnklePitch', 'rightAnkleRoll', 'leftAnklePitch',
                      'leftAnkleRoll'):
            p.changeDynamics(self.robot_id,
                             self.joint_idx[joint],
                             restitution=restitution,
                             lateralFriction=lateralFriction,
                             spinningFriction=spinningFriction,
                             rollingFriction=rollingFriction)
        p.changeDynamics(self.plane_id,
                         -1,  # -1 for base
                         restitution=restitution,
                         lateralFriction=lateralFriction,
                         spinningFriction=spinningFriction,
                         rollingFriction=rollingFriction)

    def _setup_filter(self):
        # TODO test calculation and filtering for COP
        self.COP_filter = COPCalculator(10, 10, self._dt_filter, 1)

        # filtering states
        # TODO test filtering for state and reward
        self.state_filter = {}
        for i in range(self.num_states):
            # TODO experiment with different cutoff frequencies
            self.state_filter[i] = ButterworthFilter(
                sample_period=self._dt_filter,
                cutoff=10,
                filter_order=1)
        self.COM_pos_local_filter = {}
        for i in range(3):
            self.COM_pos_local_filter[i] = ButterworthFilter(
                sample_period=self._dt_filter,
                cutoff=10,
                filter_order=1)
        for joint in self.controlled_joints:
            # TODO Add self defined PD controller
            if joint in self.Kp:
                self.pd_controller.update({joint: PDController(
                    Kp=self.Kp[joint],
                    Kd=self.Kd[joint],
                    u_max=self.u_max[joint],
                    v_max=self.v_max[joint],
                    name=joint,
                    is_filter=[True, True, False],
                    sample_period=self._dt_filter,
                    cutoff=[10, 10, 10],
                    filter_order=1)})  # 250
            else:  # PD gains not defined
                continue

    def _setup_camera(
            self,
            distance=3,
            yaw=45,
            pitch=0,
            target_position=[0, 0, 0.9]):
        """Turn on camera at given position.

        Wraps pybullet.resetDebugVisualizerCamera

        To be side-on and view sagittal balancing
               yaw=0, target_position=[0, 0, 0.9]
        To be head-on and view lateral balancing
               yaw=90, target_position=[0.5, 0, 0.9]

        Parameters
        ----------
        distance : int (default 3)
        yaw : int (default 45)
        pitch : int (default 0)
        target_position : list (default 0, 0, 0.9)
        """
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        p.resetDebugVisualizerCamera(
            cameraDistance=distance,
            cameraYaw=yaw,
            cameraPitch=pitch,
            cameraTargetPosition=target_position)

    def _setup_joint_mapping(self):
        """Store joint name-to-index mapping."""
        self.joint_idx = dict()
        self.joint_idx2name = dict()
        for joint_number in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, joint_number)
            joint_index = info[0]
            joint_name = info[1].decode("utf-8")
            self.joint_idx[joint_name] = joint_index
            self.joint_idx2name[joint_index] = joint_name

    def _termination(self):
        """True if robot_id has fallen or max_steps reached."""
        return (self.steps_taken > self.max_steps) or (self.is_fallen())

    def reset_joint_states(self, base_pos=None, base_orn=None):
        """Reset joints to default configuration, and base velocity to zero."""
        if base_pos is None:
            base_pos = self.default_base_pos
        if base_orn is None:
            base_orn = self.default_base_orn
        p.resetBasePositionAndOrientation(self.robot_id, base_pos, base_orn)
        p.resetBaseVelocity(self.robot_id, [0, 0, 0], [0, 0, 0])
        for joint in self.default_joint_config:
            p.resetJointState(
                self.robot_id,
                self.joint_idx[joint],
                targetValue=self.default_joint_config[joint],
                targetVelocity=0)

    def initialise_filtering(self):
        """Initialise filters for states, PD controller, and COM position."""
        observation = self.get_observation()
        for i in range(self.num_states):
            self.state_filter[i].initialise_xy(observation[i])
        # TODO filtering for PD controller
        for j in self.controlled_joints:
            position, velocity, _, torque = p.getJointState(self.robot_id,
                                                            self.joint_idx[j])
            if j in self.Kp:  # if PD gains is defined for joint j
                self.pd_controller[j].reset(position, velocity, torque)
        # Initialise filtering for COM position
        for i in range(3):
            self.COM_pos_local_filter[i].initialise_xy(self.COM_pos_local[i])
            self.COM_pos_local_filtered[i] = self.COM_pos_local_filter[i].y[0]

    def perform_filtering(self):  # TODO test filtering
        """Apply filtering of states, PD controller, and COM position."""
        observation = self.get_observation()
        for i in range(self.num_states):
            self.state_filter[i](observation[i])
        # Binay state filter
        # self.left_contact_filter(observation[39])
        # self.right_contact_filter(observation[59])

        # TODO filtering for PD controller
        for j in self.controlled_joints:
            position, velocity, _, _ = p.getJointState(self.robot_id,
                                                       self.joint_idx[j])
            if j in self.Kp:  # if PD gains is defined for joint j
                self.pd_controller[j].update_measurements(position, velocity)
        # perform filtering of COM position
        for i in range(3):
            self.COM_pos_local_filter[i](self.COM_pos_local[i])
            self.COM_pos_local_filtered[i] = self.COM_pos_local_filter[i].y[0]

    def get_observation(self):
        x_observation = np.zeros((self.num_states,))

        # This may not be the Pelvis CoM position _exactly_ but should be fine,
        # otherwise can apply local transformation
        _, base_quat = p.getBasePositionAndOrientation(self.robot_id)
        base_orn = p.getEulerFromQuaternion(base_quat)
        base_pos_vel, base_orn_vel = p.getBaseVelocity(self.robot_id)

        # Gravitational acceleration acts as a reference for pitch and roll but
        # not for yaw.

        # use calculated base COM position
        base_pos = self.link_COM_position['pelvisBase']

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
            self.robot_id, self.joint_idx['torsoRoll'],
            computeLinkVelocity=1)
        chest_link_dis = np.array(chest_link_state[0]) - np.array(base_pos)
        chest_link_dis.resize(1, 3)
        # chest_link_dis_base = np.transpose(R_i @ chest_link_dis.transpose())
        chest_link_dis_yaw = np.transpose(Rz_i @ chest_link_dis.transpose())
        # chest_link_dis_yaw = chest_link_dis
        # chest_com_position_x - pelvis_com_position_x
        # chest_com_position_y - pelvis_com_position_y
        # chest_com_position_z - pelvis_com_position_z
        x_observation[8] = chest_link_dis_yaw[0][0]
        x_observation[9] = chest_link_dis_yaw[0][1]
        x_observation[10] = chest_link_dis_yaw[0][2]

        torso_pitch_joint_state = p.getJointState(
            self.robot_id, self.joint_idx['torsoPitch'])
        x_observation[11] = torso_pitch_joint_state[0]
        x_observation[12] = (torso_pitch_joint_state[1]
                             / self.v_max['torsoPitch'])

        right_hip_roll_joint_state = p.getJointState(
            self.robot_id, self.joint_idx['rightHipRoll'])
        x_observation[13] = right_hip_roll_joint_state[0]  # position
        # velocity
        x_observation[14] = (right_hip_roll_joint_state[1]
                             / self.v_max['rightHipRoll'])
        right_hip_pitch_joint_state = p.getJointState(
            self.robot_id, self.joint_idx['rightHipPitch'])
        x_observation[15] = right_hip_pitch_joint_state[0]  # position
        # velocity
        x_observation[16] = (right_hip_pitch_joint_state[1]
                             / self.v_max['rightHipPitch'])
        right_knee_pitch_joint_state = p.getJointState(
            self.robot_id, self.joint_idx['rightKneePitch'])
        x_observation[17] = right_knee_pitch_joint_state[0]  # position
        # velocity
        x_observation[18] = (right_knee_pitch_joint_state[1]
                             / self.v_max['rightKneePitch'])
        right_ankle_pitch_joint_state = p.getJointState(
            self.robot_id, self.joint_idx['rightAnklePitch'])
        x_observation[19] = right_ankle_pitch_joint_state[0]
        x_observation[20] = (right_ankle_pitch_joint_state[1]
                             / self.v_max['rightAnklePitch'])
        right_ankle_roll_joint_state = p.getJointState(
            self.robot_id, self.joint_idx['rightAnkleRoll'])
        x_observation[21] = right_ankle_roll_joint_state[0]
        x_observation[22] = (right_ankle_roll_joint_state[1]
                             / self.v_max['rightAnkleRoll'])

        right_foot_link_state = p.getLinkState(
            self.robot_id, self.joint_idx['rightAnkleRoll'],
            computeLinkVelocity=0)
        right_foot_link_dis = np.array(right_foot_link_state[0]) - base_pos
        right_foot_link_dis.resize(1, 3)
        # right_foot_link_dis_base = np.transpose(
        #     R_i @right_foot_link_dis.transpose())
        right_foot_link_dis_yaw = np.transpose(
            Rz_i @right_foot_link_dis.transpose())
        # right_foot_link_dis_yaw = right_foot_link_dis
        # foot_com_position_x - pelvis_com_position_x
        # foot_com_position_y - pelvis_com_position_y
        # foot_com_position_z - pelvis_com_position_z
        x_observation[23] = right_foot_link_dis_yaw[0][0]
        x_observation[24] = right_foot_link_dis_yaw[0][1]
        x_observation[25] = right_foot_link_dis_yaw[0][2]

        left_hip_roll_joint_state = p.getJointState(
            self.robot_id, self.joint_idx['leftHipRoll'])
        x_observation[26] = left_hip_roll_joint_state[0]  # position
        # velocity
        x_observation[27] = (left_hip_roll_joint_state[1]
                             / self.v_max['leftHipRoll'])
        left_hip_pitch_joint_state = p.getJointState(
            self.robot_id, self.joint_idx['leftHipPitch'])
        x_observation[28] = left_hip_pitch_joint_state[0]  # position
        # velocity
        x_observation[29] = (left_hip_pitch_joint_state[1]
                             / self.v_max['leftHipPitch'])
        left_knee_pitch_joint_state = p.getJointState(
            self.robot_id, self.joint_idx['leftKneePitch'])
        x_observation[30] = left_knee_pitch_joint_state[0]  # position
        # velocity
        x_observation[31] = (left_knee_pitch_joint_state[1]
                             / self.v_max['leftKneePitch'])
        left_ankle_pitch_joint_state = p.getJointState(
            self.robot_id, self.joint_idx['leftAnklePitch'])
        x_observation[32] = left_ankle_pitch_joint_state[0]
        x_observation[33] = (left_ankle_pitch_joint_state[1]
                             / self.v_max['leftAnklePitch'])
        left_ankle_roll_joint_state = p.getJointState(
            self.robot_id, self.joint_idx['leftAnkleRoll'])
        x_observation[34] = left_ankle_roll_joint_state[0]
        x_observation[35] = (left_ankle_roll_joint_state[1]
                             / self.v_max['leftAnkleRoll'])

        left_foot_link_state = p.getLinkState(
            self.robot_id, self.joint_idx['leftAnkleRoll'],
            computeLinkVelocity=0)
        left_foot_link_dis = np.array(left_foot_link_state[0]) - base_pos
        left_foot_link_dis.resize(1, 3)
        # left_foot_link_dis_base = np.transpose(
        #     R_i @left_foot_link_dis.transpose())
        left_foot_link_dis_yaw = np.transpose(
            Rz_i @left_foot_link_dis.transpose())
        # left_foot_link_dis_yaw = left_foot_link_dis
        # foot_com_position_x - pelvis_com_position_x
        # foot_com_position_y - pelvis_com_position_y
        # foot_com_position_z - pelvis_com_position_z
        x_observation[36] = left_foot_link_dis_yaw[0][0]
        x_observation[37] = left_foot_link_dis_yaw[0][1]
        x_observation[38] = left_foot_link_dis_yaw[0][2]

        COM_dis = np.array(self.COM_pos-np.array(base_pos))
        COM_dis.resize(1, 3)
        # COM_dis_base = np.transpose(R_i @ COM_dis.transpose())
        COM_dis_yaw = np.transpose(Rz_i @ COM_dis.transpose())
        # COM_dis_yaw = COM_dis
        x_observation[39] = COM_dis_yaw[0][0]
        x_observation[40] = COM_dis_yaw[0][1]
        x_observation[41] = COM_dis_yaw[0][2]

        cp = self.get_capture_point()
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

    def get_extended_observation(self):
        """Same as get_filtered_observation() method."""
        return self.get_filtered_observation()

    def get_filtered_observation(self):
        observation = self.get_observation()
        observation_filtered = np.zeros(np.size(observation))
        for i in range(self.num_states):
            observation_filtered[i] = self.state_filter[i].y[0]
        # TODO binary values should not be filtered.
        # observation_filtered[48] = observation[48]
        # observation_filtered[49] = observation[49]
        # observation_filtered[39]=self.left_contact_filter.y
        # observation_filtered[59]=self.right_contact_filter.y
        return observation_filtered

    def calculate_COM_position(self):
        """Get position of Centre of Mass."""
        # update global COM position
        summ = np.zeros(3)
        for link, mass in self.link_masses.items():
            summ += np.array(self.link_COM_position[link]) * mass
        summ /= self.total_mass
        self.COM_pos = summ

        # update local COM position w.r.t centre of support polygon
        right_foot_info = p.getLinkState(
            self.robot_id,
            self.joint_idx['rightAnkleRoll'],
            computeLinkVelocity=0)
        left_foot_info = p.getLinkState(
            self.robot_id,
            self.joint_idx['leftAnkleRoll'],
            computeLinkVelocity=0)
        # Transformation from the link frame position to centre of bottom of
        # foot w.r.t link frame
        T = np.array([[0.045], [0], [-0.088]])  # (3, 1) column vector
        right_quat = right_foot_info[1]
        left_quat = left_foot_info[1]
        right_T1 = self.transform(right_quat) @ T
        left_T1 = self.transform(left_quat) @ T
        right_foot_bottom_centre = right_foot_info[4] + right_T1.T
        left_foot_bottom_centre = left_foot_info[4] + left_T1.T
        # support polygon changes if there is foot contact
        if self.is_ground_contact('right') and self.is_ground_contact('left'):
                self.support_polygon_centre = (
                    right_foot_bottom_centre + left_foot_bottom_centre) / 2.0
        elif self.is_ground_contact('right'):
                self.support_polygon_centre = right_foot_bottom_centre
        elif self.is_ground_contact('left'):
            self.support_polygon_centre = left_foot_bottom_centre
        # else both feet not in contact: maintain current support_polygon value.
        self.COM_pos_local = np.ravel(self.COM_pos-self.support_polygon_centre)

        self.support_polygon_centre_surrogate = (
            right_foot_bottom_centre + left_foot_bottom_centre) / 2.0
        # set z value to the lowest of the two feet i.e. closest to the floor
        self.support_polygon_centre_surrogate[0][2] = min(
            right_foot_bottom_centre[0][2], left_foot_bottom_centre[0][2])
        return self.COM_pos

    def calculate_link_COM_velocity(self):
        """Get centre of mass velocity for all links and base."""
        self.link_COM_velocity = {}
        base_pos_vel, base_orn_vel = p.getBaseVelocity(self.robot_id)
        self.link_COM_velocity["pelvisBase"] = base_pos_vel
        for joint, idx in self.joint_idx.items():
            info = p.getLinkState(self.robot_id, idx, computeLinkVelocity=1)
            self.link_COM_velocity[joint] = info[6]
        return self.link_COM_velocity

    def calculate_link_COM_position(self):
        """Compute centre of mass position for all links and base."""
        self.link_COM_position = {}
        base_pos, base_quat = p.getBasePositionAndOrientation(self.robot_id)
        # TODO check if base position is the COM of the pelvis
        self.link_COM_position["pelvisBase"] = np.array(base_pos)
        for joint, idx in self.joint_idx.items():
            info = p.getLinkState(self.robot_id, idx)
            self.link_COM_position[joint] = info[0]
        return self.link_COM_position

    def calculate_link_masses(self):
        """Compute link mass and total mass information."""
        info = p.getDynamicsInfo(self.robot_id, -1)  # for base link
        self.link_masses = dict()
        self.link_masses["pelvisBase"] = info[0]
        self.total_mass = info[0]
        for joint, idx in self.joint_idx.items():
            info = p.getDynamicsInfo(self.robot_id, idx)
            self.link_masses[joint] = info[0]
            self.total_mass += info[0]
        return self.link_masses

    def calculate_ground_contact_points(self):
        """Compute foot contact points with ground.

        Returns
        -------
        contact_points : numpy.array of length 3
        """
        foot_ground_contact_info = []
        for side in ('right', 'left'):
            if self.is_ground_contact(side):
                ankle_roll_contact = p.getContactPoints(
                    self.robot_id,
                    self.plane_id,
                    self.joint_idx[side+'AnkleRoll'],
                    -1)
                ankle_pitch_contact = p.getContactPoints(
                    self.robot_id,
                    self.plane_id,
                    self.joint_idx[side+'AnklePitch'],
                    -1)
                # use extend not append because getContactPoints returns a list
                foot_ground_contact_info.extend(ankle_roll_contact)
                foot_ground_contact_info.extend(ankle_pitch_contact)
        # get just x, y cartesian coordinates of contact position on robot_id
        self.contact_points = np.array(
            [info[5][0:2] for info in foot_ground_contact_info])
        return self.contact_points

    def get_capture_point(self):
        """Physical properties derived from capture point theory."""
        # input x is the target displacement w.r.t centre of foot
        h = self.COM_pos_local[2]  # height of COM w.r.t bottom of foot
        # h = self.COM_pos[2] - self.support_polygon_centre[0][2]
        z = max(0.05, h)  # Make sure denominator is not zero
        COM_vel = self.COM_vel
        cp = COM_vel * np.sqrt(z / self.g)
        return np.array(cp)

    def target_COM_velocity(self, target, axis='x'):
        """Return target velocity for target displacement wrt centre of foot."""
        if axis == 'x':
            com_position = self.COM_pos_local[0]  # COM pos wrt to bottom foot
        elif axis == 'y':
            com_position = self.COM_pos_local[1]
        else:
            raise ValueError("axis must be 'x' or 'y'. Got '{}'".format(axis))
        dist = target - com_position
        h = self.COM_pos_local[2]  # height of COM w.r.t bottom of foot
        z = max(0.01, h)  # Make sure denominator is not zero
        tau = np.sqrt(z / self.g)
        velocity = dist / tau
        return velocity

    def transform(self, quaternions):
        """Transform quaternion into rotation matrix.

        quaternions : list of length 4
        """
        qx, qy, qz, qw = quaternions

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

    def is_ground_contact(self, side=None):
        """True if robot_id is in contact with ground.

        Parameters
        ----------
        side : str (optional)
            Either 'left' or 'right', else checks both sides.
       """
        if side:
            ground_contact_points = p.getContactPoints(
                self.robot_id,
                self.plane_id,
                self.joint_idx[side+'AnkleRoll'],
                -1)
            return len(ground_contact_points) > 0
        else:
            is_left_contact = self.is_ground_contact('left')
            is_right_contact = self.is_ground_contact('right')
            return is_left_contact or is_right_contact

    def is_fallen(self):
        """Return True if robot has fallen, else False."""
        base_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        # TODO check is_fallen criteria
        if base_pos[2] <= 0.3 or self.COM_pos[2] <= 0.3:
            return True
        for joint in self.controllable_joints:
            if 'Ankle' not in joint:  # check if any part except feet on floor
                contact_points = p.getContactPoints(
                    self.robot_id,
                    self.plane_id,
                    self.joint_idx[joint],
                    -1)  # -1 for base
                if len(contact_points) > 0:
                    return True
        return False

    def calculate_COP(self):
        """Calculate centre of pressure."""
        # TODO ground contact detection using contact point
        foot_ground_contact = []
        ankle_roll_contact = p.getContactPoints(
            self.robot_id,
            self.plane_id,
            self.joint_idx['leftAnkleRoll'],
            -1)
        ankle_pitch_contact = p.getContactPoints(
            self.robot_id,
            self.plane_id,
            self.joint_idx['leftAnklePitch'],
            -1)
        foot_ground_contact.extend(ankle_roll_contact)
        foot_ground_contact.extend(ankle_pitch_contact)
        left_contact_info = foot_ground_contact

        foot_ground_contact = []
        ankle_roll_contact = p.getContactPoints(
            self.robot_id,
            self.plane_id,
            self.joint_idx['rightAnkleRoll'],
            -1)
        ankle_pitch_contact = p.getContactPoints(
            self.robot_id,
            self.plane_id,
            self.joint_idx['rightAnklePitch'],
            -1)
        foot_ground_contact.extend(ankle_roll_contact)
        foot_ground_contact.extend(ankle_pitch_contact)
        right_contact_info = foot_ground_contact

        self.COP_filter(
            right_contact_info=right_contact_info,
            left_contact_info=left_contact_info)
        self.COP_info = self.COP_filter.get_COP()
        self.COP_info_filtered = self.COP_filter.get_filtered_COP()
        return self.COP_info_filtered

    def calculate_pelvis_acc(self):
        """Calculate pelvis acceleration. (NOTE: don't understand this yet)"""
        # TODO add pelvis acceleration
        base_pos, base_quat = p.getBasePositionAndOrientation(self.robot_id)
        R = self.transform(base_quat)
        R_i = np.linalg.inv(R)
        pelvis_vel, _ = p.getBaseVelocity(self.robot_id)
        stop = len(self.pelvis_vel_history_array) - 2
        self.pelvis_vel_history_array[0:stop, :, None] = \
            self.pelvis_vel_history_array[1:stop+1, :, None]  # shift element
        for i in range(len(self.pelvis_vel_history_array)-1):
            self.pelvis_vel_history_array[i] = \
                self.pelvis_vel_history_array[i+1]

        self.pelvis_vel_history_array[
            len(self.pelvis_vel_history_array) - 1, :] = np.array(pelvis_vel)
        x = self.pelvis_vel_history_array[:, 0]
        y = self.pelvis_vel_history_array[:, 1]

        z = self.pelvis_vel_history_array[:, 2]
        # time step
        t = np.arange(len(self.pelvis_vel_history_array)) * self._dt_physics
        x_acc = np.polyfit(t, x, 1)  # use slope of velocity as acceleration
        y_acc = np.polyfit(t, y, 1)
        z_acc = np.polyfit(t, z, 1)
        self.pelvis_acc = np.array([x_acc[0], y_acc[0], z_acc[0]])
        self.pelvis_acc_base = np.transpose(R_i @ self.pelvis_acc.transpose())

        self.pelvis_vel_history = np.array(pelvis_vel)
        self.pelvis_vel_history_base = np.transpose(
            R_i @self.pelvis_vel_history.transpose())

    def calculate_COM_velocity(self):
        """Calculate velocity of Centre of Mass."""
        summ = np.zeros((1, 3))
        for link, mass in self.link_masses.items():
            summ += np.array(self.link_COM_velocity[link]) * mass
        summ /= self.total_mass
        self.COM_vel[0:3] = np.array(summ)
        return self.COM_vel

    def calculate_PD_torque(self, u):
        """Calculate torque using self desinged PD controller.

        Parameters
        ----------
        u : list of same length as self.controlled_joints
            target position for controlled joint 0, 1, ..., nu

        Returns
        -------
        torques : dict
            {joint: torque} as computed by self.pd_controller
        """
        # Set control. pd_controller call will fail if any element of u is NaN
        torques = dict()
        for i, joint in enumerate(self.controlled_joints):
            if joint in self.Kp:  # PD gains defined for joint
                torque = self.pd_controller[joint].control(u[i], 0.0)
                torque = np.clip(torque, -self.u_max[joint], self.u_max[joint])
                torques[joint] = torque
        return torques

    def joint_reaction_force(self, side):
        """Return total reaction force (torque) on given side.

        Parameters
        ----------
        side : str
            Either 'left' or 'right'
        """
        ankle_joint_state = p.getJointState(self.robot_id,
                                            self.joint_idx[side+'AnklePitch'])
        _, _, _, Mx, My, Mz = ankle_joint_state[2]
        total_torque = np.sqrt(np.sum(np.square([Mx, My, Mz])))
        return total_torque

    def set_zero_order_hold_nominal_pose(self):
        """Set robot_id into default standing position.

        Default position is given by `default_joint_config` attribute.
        """
        for joint in self.controllable_joints:
            p.setJointMotorControl2(
                self.robot_id,
                self.joint_idx[joint],
                p.POSITION_CONTROL,
                targetPosition=self.default_joint_config[joint],
                force=self.u_max[joint])

    def set_PD_velocity_control(self, torque_dict):
        """Set the maximum motor force limit for joints.

        We are using this as a form of _torque_ control, where the target
        velocity is _always_ set to be the maximum velocity allowed, and we
        input the maximum torque that we want to apply to reach that velocity.

        When a step is taken, force (up to the limit we input) is applied to
        reach the target velocity, where
            target velocity = self.v_max * sign(torque).

        Parameters
        ----------
        torque_dict : dict of {str: float}
            joint names and the maximum torque to allow upon taking a step
        """
        for joint, torque in torque_dict.items():
            p.setJointMotorControl2(
                self.robot_id,
                self.joint_idx[joint],
                # set desired velocity of joint
                targetVelocity=np.sign(torque)*self.v_max[joint],
                # set maximum motor force that can be used
                force=np.abs(torque),
                controlMode=p.VELOCITY_CONTROL)

    def rotX(self, theta):
        """Roll rotation matrix."""
        R = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(theta), -np.sin(theta)],
            [0.0, np.sin(theta), np.cos(theta)]])
        return R

    def rotY(self, theta):
        """Pitch rotation matrix."""
        R = np.array([
            [np.cos(theta), 0.0, np.sin(theta)],
            [0.0, 1.0, 0.0],
            [-np.sin(theta), 0.0, np.cos(theta)]])
        return R

    def rotZ(self, theta):
        """Yaw rotation matrix."""
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0]])
        return R

    def render(self, mode='human', close=False):
        p.addUserDebugLine(
            self.COM_pos + np.array([0, 0, 2]),
            self.COM_pos + np.array([0, 0, -2]),
            [1, 0, 0],
            5,
            0.1)  # TODO rendering to draw COM
        p.addUserDebugLine(
            self.support_polygon_centre[0]+np.array([0, 0, 2]),
            self.support_polygon_centre[0]+np.array([0, 0, -2]),
            [0, 1, 0],
            5,
            0.1)  # TODO rendering to draw support polygon
        p.addUserDebugLine(
            self.support_polygon_centre[0]+np.array([2, 0, 0]),
            self.support_polygon_centre[0]+np.array([-2, 0, 0]),
            [0, 1, 0],
            5,
            0.1)  # TODO rendering to draw support polygon

    def start_logging_video(self):
        """Begin saving MP4 video of simulation state."""
        self.logId = p.startStateLogging(
            loggingType=p.STATE_LOGGING_VIDEO_MP4,
            fileName=self._logFileName + '/video.mp4')
        p.startStateLogging(self.logId)

    def stop_logging_video(self):
        """Stop recording MP4 video of simulation state."""
        p.stopStateLogging(self.logId)

    def toggle_rendering(self):
        """Turn visualisation on/off."""
        if self._render:  # It's on, so turn it off
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            self._render = False
        else:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            self._render = True

    def diagnostics(self):
        """Return dictionary with name and value of all measure quantities."""
        readings = dict()
        readings['rightGroundContact'] = self.is_ground_contact('right')
        readings['leftGroundContact'] = self.is_ground_contact('left')
        readings['rightGroundContactReactionForce'] = \
            self.is_ground_contact_reacton_force('right')
        readings['leftGroundContactReactionForce'] = \
            self.is_ground_contact_reacton_force('left')
        readings['rightGroundReactionForce'] = \
            self.joint_reaction_force('right')
        readings['leftGroundReactionForce'] = self.joint_reaction_force('left')

        for joint in ('torsoPitch',
                      'rightHipPitch',
                      'leftHipPitch',
                      'rightKneePitch',
                      'leftKneePitch',
                      'rightAnklePitch',
                      'leftAnklePitch'):
            joint_state = p.getJointState(self.robot_id, self.joint_idx[joint])
            (readings[joint+'Angle'], readings[joint+'Velocity'], _,
             readings[joint+'_torque']) = joint_state

        left_foot_link_state = p.getLinkState(self.robot_id,
                                              self.joint_idx['leftAnkleRoll'],
                                              computeLinkVelocity=0)  # 0 for no
        readings['leftFootPitch'] = left_foot_link_state[1][1]
        right_foot_link_state = p.getLinkState(self.robot_id,
                                               self.joint_idx['rightAnkleRoll'],
                                               computeLinkVelocity=0)
        readings['rightFootPitch'] = right_foot_link_state[1][1]

        # right_COP,right_contact_force,_ = self.caclulate_foot_COP('right')
        # left_COP,left_contact_force,_ = self.caclulate_foot_COP('left')
        (COP, contact_force, _, right_COP, right_contact_force, _, left_COP,
         left_contact_force, _) = self.COP_info
        readings['rightCOP'] = right_COP
        readings['leftCOP'] = left_COP
        readings['right_contact_force'] = right_contact_force
        readings['left_contact_force'] = left_contact_force
        readings['contact_force'] = contact_force
        readings['COP'] = COP

        (COP, contact_force, _, right_COP, right_contact_force, _, left_COP,
         left_contact_force, _) = self.COP_info_filtered
        readings['rightCOP_filtered'] = right_COP
        readings['leftCOP_filtered'] = left_COP
        readings['right_contact_force_filtered'] = right_contact_force
        readings['left_contact_force_filtered'] = left_contact_force
        readings['contact_force_filtered'] = contact_force
        readings['COP_filtered'] = COP

        readings['COM_vel'] = np.array(self.COM_vel)
        readings['COM_pos'] = np.array(self.COM_pos)
        readings['COM_pos_local'] = np.array(self.COM_pos_local)
        readings['COM_pos_local_filtered'] = np.array(
            self.COM_pos_local_filtered)
        readings['support_polygon_centre'] = self.support_polygon_centre[0]
        readings['support_polygon_centre_surrogate'] = \
            self.support_polygon_centre_surrogate[0]

        # record joint torque calculated by self defined PD controller
        for joint in self.controlled_joints:
            if joint in self.Kp:  # if PD gains are defined
                readings[joint+'PDPosition'] = \
                    self.pd_controller[joint].filtered_position
                readings[joint+'PDVelocity'] = \
                    self.pd_controller[joint].filtered_velocity
                readings[joint+'PDPositionAdjusted'] = \
                    self.pd_controller[joint].adjusted_position
                readings[joint+'PDVelocityAdjusted'] = \
                    self.pd_controller[joint].adjusted_velocity
                readings[joint+'PD_torque'] = self.pd_controller[joint].u_kal
                readings[joint+'PD_torque_Kp'] = self.pd_controller[joint].u_e
                readings[joint+'PD_torque_Kd'] = self.pd_controller[joint].u_de
                readings[joint+'PDPositionKalman'] = \
                    self.pd_controller[joint].kalman_filtered_position
                readings[joint+'PDVelocityKalman'] = \
                    self.pd_controller[joint].kalman_filtered_velocity
                readings[joint+'PDFiltered_torque'] = \
                    self.pd_controller[joint].u_kal
                readings[joint+'PDUnfiltered_torque'] = \
                    self.pd_controller[joint].u_raw
                readings[joint+'PDAdjusted_torque'] = \
                    self.pd_controller[joint].u_adj

        for joint in ('torsoPitch',
                      'rightHipPitch',
                      'leftHipPitch',
                      'rightKneePitch',
                      'leftKneePitch',
                      'rightAnklePitch',
                      'leftAnklePitch'):
            readings[joint+'AngleTarget'] = self.action[joint]

        # return filterred joint angles
        observation = self.get_observation()
        readings['torsoPitchAngle10HzFiltered'] = observation[11]
        readings['leftHipPitchAngle10HzFiltered'] = observation[19]
        readings['leftKneePitchAngle10HzFiltered'] = observation[21]
        readings['leftAnklePitchAngle10HzFiltered'] = observation[23]
        readings['rightHipPitchAngle10HzFiltered'] = observation[32]
        readings['rightKneePitchAngle10HzFiltered'] = observation[34]
        readings['rightAnklePitchAngle10HzFiltered'] = observation[36]

        readings['pelvis_acc_global'] = self.pelvis_acc
        readings['pelvis_acc_base'] = self.pelvis_acc_base
        readings['pelvis_vel_global'] = self.pelvis_vel_history
        readings['pelvis_vel_base'] = self.pelvis_vel_history_base

        # record individual terms in reward
        _, reward_term = self.reward()
        readings.update(reward_term)
        return readings

    def _debug(self):
        # Transformation from link frame pos to geometry centre wrt link frame
        T = np.array([[0.066],
                      [0],
                      [-0.056]])
        # Trans from link frame pos to centre of bottom of foot wrt link frame
        T1 = np.array([[0.066],
                       [0],
                       [-0.088]])
        for side in ('right', 'left'):
            (link_world_pos, _, local_inertia_pos, _, link_frame_pos,
             link_frame_orn) = p.getLinkState(
                 self.robot_id,
                 self.joint_idx[side+'AnkleRoll'],
                 computeLinkVelocity=0)
            T_ = (self.transform(link_frame_orn)) @ T
            T1_ = (self.transform(link_frame_orn)) @ T1

            foot_centre = link_frame_pos + T_.T
            foot_bottom_centre = link_frame_pos + T1_.T
            print("----------")
            print(side+" foot")
            print('foot centre')
            print(foot_centre)
            print(foot_bottom_centre)
            print('linkWorldPosition')
            print(link_world_pos)
            print('localInertialFramePosition')
            print(local_inertia_pos)
            print('worldLinkFramePosition')
            print(link_frame_pos)


class ValkyrieEnvBasic(ValkyrieEnvBase):

    def reward(self):
        """Return reward for current state and terms used in the reward.

        Returns
        -------
        reward : float
        reward_term : dict
            {name: value} of all elements summed to make the reward
        """
        base_pos, base_quat = p.getBasePositionAndOrientation(self.robot_id)
        base_orn = p.getEulerFromQuaternion(base_quat)
        chest_link_state = p.getLinkState(self.robot_id,
                                          self.joint_idx['torsoRoll'],
                                          computeLinkVelocity=1)
        torso_roll_err, torso_pitch_err, _, _ = chest_link_state[1]
        pelvis_roll_err, pelvis_pitch_err, _ = base_orn
        # Nominal COM x and z position [0.0, 1.104]
        # Nominal COM x and z position w.r.t COM of foot [0.0, 1.064] #TODO test

        # filtering for reward
        x_pos_err = 0.0 - self.COM_pos_local[0]
        y_pos_err = 0.0 - self.COM_pos_local[1]
        z_pos_err = 1.175 * 1.02 - self.COM_pos_local[2]

        xy_pos_target = np.array([0.0, 0.0])
        xy_pos = self.COM_pos_local[[0, 1]]
        xy_pos_err = np.linalg.norm(xy_pos - xy_pos_target)

        x_vel_err = self.target_COM_velocity(0.0, 'x') - self.COM_vel[0]
        y_vel_err = self.target_COM_velocity(0.0, 'y') - self.COM_vel[1]
        z_vel_err = 0.0 - self.COM_vel[2]

        xy_vel_target = np.array([self.target_COM_velocity(0.0, 'x'),
                                  self.target_COM_velocity(0.0, 'y')])
        xy_vel = self.COM_vel[[0, 1]]
        xy_vel_err = np.linalg.norm(xy_vel - xy_vel_target)

        # force distribution
        (COP, contact_force, _, right_COP, right_contact_force, _,
         left_COP, left_contact_force, _) = self.COP_info_filtered
        force_target = self.total_mass * self.g / 2.0
        # Z contact force
        right_foot_force_err = force_target - right_contact_force[2]
        left_foot_force_err = force_target - left_contact_force[2]

        # foot roll
        right_foot_link_state = p.getLinkState(self.robot_id,
                                               self.joint_idx['rightAnkleRoll'],
                                               computeLinkVelocity=0)
        right_foot_orn = p.getEulerFromQuaternion(right_foot_link_state[1])
        right_foot_roll_err = right_foot_orn[2]
        left_foot_link_state = p.getLinkState(self.robot_id,
                                              self.joint_idx['leftAnkleRoll'],
                                              computeLinkVelocity=0)
        left_foot_orn = p.getEulerFromQuaternion(left_foot_link_state[1])
        left_foot_roll_err = left_foot_orn[2]

        x_pos_reward = np.exp(-19.51 * x_pos_err ** 2)
        y_pos_reward = np.exp(-19.51 * y_pos_err ** 2)
        z_pos_reward = np.exp(-113.84 * z_pos_err ** 2)  # -79.73
        xy_pos_reward = np.exp(-19.51 * xy_pos_err ** 2)

        x_vel_reward = np.exp(-0.57 * (x_vel_err) ** 2)
        y_vel_reward = np.exp(-0.57 * (y_vel_err) ** 2)
        z_vel_reward = np.exp(-3.69 * (z_vel_err) ** 2)  # -1.85
        xy_vel_reward = np.exp(-0.57 * (xy_vel_err) ** 2)

        torso_pitch_reward = np.exp(-4.68 * (torso_pitch_err) ** 2)
        torso_roll_reward = np.exp(-4.68 * (torso_roll_err) ** 2)
        pelvis_pitch_reward = np.exp(-4.68 * (pelvis_pitch_err) ** 2)
        pelvis_roll_reward = np.exp(-4.68 * (pelvis_roll_err) ** 2)

        right_foot_force_reward = np.exp(-2e-5 * (right_foot_force_err) ** 2)
        right_foot_roll_reward = np.exp(-4.68 * (right_foot_roll_err) ** 2)
        left_foot_force_reward = np.exp(-2e-5 * (left_foot_force_err) ** 2)
        left_foot_roll_reward = np.exp(-4.68 * (left_foot_roll_err) ** 2)

        foot_contact_term = 0
        if not self.is_ground_contact():  # both feet lost contact
            foot_contact_term -= 1  # TODO increase penalty for losing contact

        fall_term = 0
        if self.is_fallen():
            fall_term -= 10

        reward = 10 * (2.0 * xy_pos_reward
                       + 3.0 * z_pos_reward
                       + 2.0 * xy_vel_reward
                       + 1.0 * z_vel_reward
                       + 1.0 * torso_pitch_reward
                       + 1.0 * torso_roll_reward
                       + 1.0 * pelvis_pitch_reward
                       + 1.0 * pelvis_roll_reward
                       + 1.0 * right_foot_force_reward
                       + 1.0 * left_foot_force_reward
                       # + 1.0 * right_foot_roll_reward
                       # + 1.0 * left_foot_roll_reward
                       )/(2.0+3.0+2.0+1.0+1.0+1.0+1.0+1.0+1.0+1.0)  # sum coefs
        reward = reward + fall_term + foot_contact_term

        ##########
        # penalize reward when target position hard to achieve: position - actn
        position_follow_penalty = 0
        for joint in self.controlled_joints:  # TODO
            joint_state = p.getJointState(self.robot_id, self.joint_idx[joint])
            position_follow_penalty -= abs(joint_state[0] - self.action[joint])

        # penalize reward when joint is moving too fast: (vel / max_vel)^2
        velocity_penalty = 0
        for joint in self.controlled_joints:  # TODO
            joint_state = p.getJointState(self.robot_id, self.joint_idx[joint])
            velocity_penalty -= (joint_state[1] / self.v_max[joint])**2

        # penalize reward when torque: torque / max_torque
        torque_penalty = 0
        for joint in self.controlled_joints:  # TODO
            joint_state = p.getJointState(self.robot_id, self.joint_idx[joint])
            torque_penalty -= abs(joint_state[3] / self.u_max[joint])

        # penalize power rate of joint motor: vel/max_vel * torque/max_torque
        power_penalty = 0
        for joint in self.controlled_joints:  # TODO
            joint_state = p.getJointState(self.robot_id, self.joint_idx[joint])
            power_penalty -= (abs(joint_state[3] / self.u_max[joint])
                              * abs(joint_state[1] / self.v_max[joint]))

        # penalize change in torque: torque_change / max_torque
        torque_change_penalty = 0
        for joint in self.controlled_joints:
            joint_state = p.getJointState(self.robot_id, self.joint_idx[joint])
            torque_change = self.last_torque_applied[joint] - joint_state[3]
            torque_change_penalty -= abs(torque_change / self.u_max[joint])

        # reward += 2 * velocity_penalty
        # reward += 30 * velocity_penalty/len(self.controlled_joints)
        # reward += 30 * power_penalty / len(self.controlled_joints)
        # reward += 5 * power_penalty

        reward_term = {
            "x_pos_reward": x_pos_reward,
            "y_pos_reward": y_pos_reward,
            "z_pos_reward": z_pos_reward,
            "x_vel_reward": x_vel_reward,
            "y_vel_reward": y_vel_reward,
            "z_vel_reward": z_vel_reward,
            "torso_pitch_reward": torso_pitch_reward,
            "pelvis_pitch_reward": pelvis_pitch_reward,
            "torso_roll_reward": torso_roll_reward,
            "pelvis_roll_reward": pelvis_roll_reward,
            "power_penalty": power_penalty,
            "torque_change_penalty": torque_change_penalty,
            "velocity_penalty": velocity_penalty,
            "torque_penalty": torque_penalty,
            "position_follow_penalty": position_follow_penalty,
            "xy_pos_reward": xy_pos_reward,
            "xy_vel_reward": xy_vel_reward,
            "left_foot_force_reward": left_foot_force_reward,
            "right_foot_force_reward": right_foot_force_reward,
            "left_foot_roll_reward": left_foot_roll_reward,
            "right_foot_roll_reward": right_foot_roll_reward,
            "foot_contact_term": foot_contact_term,
            "fall_term": fall_term
        }
        return reward, reward_term


class ValkyrieEnvExtended(ValkyrieEnvBase):
    """Class which methods which are nver called by basic environment"""

    def set_PD_position_control(self, torque_dict, u):
        """Apply torque to joints to move towards target positions given by u.

        Also updates `action` attribute.
        """
        final = np.isnan(u).any()
        for i, joint in enumerate(self.controlled_joints):
            if not final:
                p.setJointMotorControl2(
                    self.robot_id,
                    self.joint_idx[joint],
                    targetPosition=u[i],
                    targetVelocity=0,
                    maxVelocity=self.v_max[joint],
                    force=abs(torque_dict[joint]),
                    controlMode=p.POSITION_CONTROL,
                    # positionGain=self.Kp[joint],
                    # velocityGain=self.Kd[joint],
                )

    def set_PD_torque_control(self, torque_dict):
        """Set exact motor force that will be applied to joints when step taken.

        We do not usually use this because torque control in pybullet does not
        allow us to specify a velocity limit.

        Parameters
        ----------
        torque_dict : dict of {str: float}
            joint names and the torque to allow upon taking a step
        """
        for joint, torque in torque_dict.items():
            p.setJointMotorControl2(
                self.robot_id,
                self.joint_idx[joint],
                # set desired velocity of joint
                targetVelocity=np.sign(torque)*self.v_max[joint],
                force=0,
                controlMode=p.VELOCITY_CONTROL)
            p.setJointMotorControl2(
                self.robot_id,
                self.joint_idx[joint],
                # set force/torque that is actually applied
                force=torque,
                controlMode=p.TORQUE_CONTROL)

    def set_control(self, u):
        """Set desired control for each joint, to be run when a step taken.

        During the `step` the physics engine will simulate the joint motors to
        reach the given target value, within the maximum motor forces and other
        constraints.

        Parameters
        ----------
        u : list of same length as self.controlled_joints
            target position for controlled joint 0, 1, ...
        """
        for i, joint in enumerate(self.controlled_joints):
            self.action.update({joint: u[i]})  # TODO
            if self.use_bullet_default_pd:
                p.setJointMotorControl2(
                    self.robot_id,
                    self.joint_idx[joint],
                    targetPosition=u[i],
                    targetVelocity=0,
                    maxVelocity=self.v_max[joint],
                    force=self.u_max[joint],
                    positionGain=self.Kp[joint],
                    velocityGain=self.Kd[joint],
                    controlMode=p.POSITION_CONTROL)

            else:  # using own PD controller
                if joint in self.Kp:  # PD gains defined
                    torque = self.pd_controller[joint].control(u[i], 0)
                    torque = np.clip(
                        torque,
                        -self.u_max[joint],
                        self.u_max[joint])
                    p.setJointMotorControl2(
                        self.robot_id,
                        self.joint_idx[joint],
                        targetVelocity=(
                            np.sign(torque) * self.v_max[joint]),
                        force=np.abs(torque),
                        velocityGains=0.0001,
                        controlMode=p.VELOCITY_CONTROL)
                else:  # PD gains not defined
                    p.setJointMotorControl2(
                        self.robot_id,
                        self.joint_idx[joint],
                        targetPosition=u[i],
                        targetVelocity=0,
                        maxVelocity=self.v_max[joint],
                        force=self.u_max[joint],
                        controlMode=p.POSITION_CONTROL)

        # Set zero order hold for uncontrolled joints
        for joint in self.uncontrolled_joints:
            p.setJointMotorControl2(
                self.robot_id,
                self.joint_idx[joint],
                targetPosition=self.default_joint_config[joint],
                force=self.u_max[joint],
                maxVelocity=self.v_max[joint],
                controlMode=p.POSITION_CONTROL)

    def apply_force_on_pelvis(self, force, pos):
        """Apply force to pelvis.

        Parameters
        ----------
        force : list of 3 floats
            force vector to be applied [x, y, z] (in link coordinates)
        pos : list of three floats
            position on pelvis where the force is applied
        """
        pos = np.array(pos) + np.array([0, 0.0035, 0])
        p.applyExternalForce(self.robot_id,
                             -1,  # base
                             forceObj=force,
                             posObj=pos,
                             flags=p.LINK_FRAME)

    def is_ground_contact_reacton_force(self, side):
        """Return ground reaction force on given side.
        (NOTE: don't yet understand calculation used)
        Parameters
        ----------
        side : str
            Either 'left' or 'right'
        """
        # TODO ground contact detection using ground reaction force
        torque = self.joint_reaction_force(side)
        # TODO adjust sigmoid function to represent actual contact as close
        # as possible
        return 1 / (1 + np.exp(-0.3 * (torque - 4)))

    def draw_base_frame(self):
        """Draw approximate pelvis centre of mass."""
        # This may not be the Pelvis CoM position _exactly_ but should be fine,
        # otherwise can apply local transformation
        _, base_quat = p.getBasePositionAndOrientation(self.robot_id)
        base_pos = self.link_COM_position['pelvisBase']
        R = self.transform(base_quat)

        x_axis = np.array([[0.8, 0, 0]])
        y_axis = np.array([[0, 0.8, 0]])
        z_axis = np.array([[0, 0, 0.8]])

        x_axis_base = np.transpose(R @ x_axis.transpose())
        y_axis_base = np.transpose(R @ y_axis.transpose())
        z_axis_base = np.transpose(R @ z_axis.transpose())

        p.addUserDebugLine(  # x axis
            np.array(base_pos),
            np.array(base_pos) + x_axis_base[0],
            [1, 0, 0],
            5,
            0.1)
        p.addUserDebugLine(   # y axis
            np.array(base_pos),
            np.array(base_pos) + y_axis_base[0],
            [0, 1, 0],
            5,
            0.1)
        p.addUserDebugLine(  # z axis
            np.array(base_pos),
            np.array(base_pos) + z_axis_base[0],
            [0, 0, 1],
            5,
            0.1)

    def draw_base_yaw_frame(self):
        """Draw approximate yaw frame."""
        # This may not be the Pelvis CoM position _exactly_ but should be fine,
        # otherwise can apply local transformation
        _, base_quat = p.getBasePositionAndOrientation(self.robot_id)
        base_orn = p.getEulerFromQuaternion(base_quat)
        Rz = self.rotZ(base_orn[2])

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
            5,
            0.1)
        p.addUserDebugLine(
            np.array(origin),
            np.array(origin) + y_axis_base[0],
            [0, 1, 0],
            5,
            0.1)
        p.addUserDebugLine(
            np.array(origin),
            np.array(origin) + z_axis_base[0],
            [0, 0, 1],
            5,
            0.1)

    def draw_skeleton_yaw(self):
        # This may not be the Pelvis CoM position _exactly_ but should be fine,
        # otherwise can apply local transformation
        base_pos, base_quat = p.getBasePositionAndOrientation(self.robot_id)
        base_orn = p.getEulerFromQuaternion(base_quat)

        # Gravitational acceleration acts as a reference for pitch and roll but
        # not for yaw.

        # overwrite base_pos with the calculated base COM position
        # base_pos is tuple, tuple is immutable
        # change tuple to array
        base_pos = np.array(base_pos)
        base_pos[0] = self.link_COM_position['pelvisBase'][0]
        base_pos[1] = self.link_COM_position['pelvisBase'][1]
        base_pos[2] = self.link_COM_position['pelvisBase'][2]
        # base_pos = self.link_COM_position['pelvisBase']

        Rz = self.rotZ(base_orn[2])
        Rz_i = np.linalg.inv(Rz)
        # R = self.transform(base_quat)
        # R_i = np.linalg.inv(R)

        # chest
        chest_link_state = p.getLinkState(
            self.robot_id, self.joint_idx['torsoRoll'],
            computeLinkVelocity=1)
        chest_link_dis = np.array(chest_link_state[0]) - np.array(base_pos)
        chest_link_dis.resize(1, 3)
        # chest_link_dis_base = np.transpose(R_i @ chest_link_dis.transpose())
        chest_link_dis_yaw = np.transpose(Rz_i @ chest_link_dis.transpose())

        left_foot_link_state = p.getLinkState(
            self.robot_id, self.joint_idx['leftAnkleRoll'],
            computeLinkVelocity=0)
        left_foot_link_dis = np.array(left_foot_link_state[0]) - base_pos
        left_foot_link_dis.resize(1, 3)
        # left_foot_link_dis_base = np.transpose(
        #     R_i @left_foot_link_dis.transpose())
        left_foot_link_dis_yaw = np.transpose(
            Rz_i @left_foot_link_dis.transpose())
        # left_foot_link_dis_yaw = left_foot_link_dis

        right_foot_link_state = p.getLinkState(
            self.robot_id, self.joint_idx['rightAnkleRoll'],
            computeLinkVelocity=0)
        right_foot_link_dis = np.array(right_foot_link_state[0]) - base_pos
        right_foot_link_dis.resize(1, 3)
        # right_foot_link_dis_base = np.transpose(
        #    R_i @right_foot_link_dis.transpose())
        right_foot_link_dis_yaw = np.transpose(
            Rz_i @right_foot_link_dis.transpose())
        # right_foot_link_dis_yaw = right_foot_link_dis

        left_elbow_link_state = p.getLinkState(
            self.robot_id, self.joint_idx['leftElbowPitch'],
            computeLinkVelocity=0)
        left_elbow_link_dis = np.array(left_elbow_link_state[0]) - base_pos
        left_elbow_link_dis.resize(1, 3)
        # left_elbow_link_dis_base = np.transpose(
        #     R_i @left_elbow_link_dis.transpose())
        left_elbow_link_dis_yaw = np.transpose(
            Rz_i @left_elbow_link_dis.transpose())

        right_elbow_link_state = p.getLinkState(
            self.robot_id, self.joint_idx['rightElbowPitch'],
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

    def draw_skeleton_base(self):
        # This may not be the Pelvis CoM position _exactly_ but should be fine,
        # otherwise can apply local transformation
        base_pos, base_quat = p.getBasePositionAndOrientation(self.robot_id)

        # Gravitational acceleration acts as a reference for pitch and roll but
        # not for yaw.

        # overwrite base_pos with the calculated base COM position
        # base_pos is tuple, tuple is immutable, change tuple to array
        base_pos = np.array(base_pos)
        base_pos[0] = self.link_COM_position['pelvisBase'][0]
        base_pos[1] = self.link_COM_position['pelvisBase'][1]
        base_pos[2] = self.link_COM_position['pelvisBase'][2]

        # Rz = self.rotZ(base_orn[2])
        # Rz_i = np.linalg.inv(Rz)
        R = self.transform(base_quat)
        R_i = np.linalg.inv(R)

        # chest
        chest_link_state = p.getLinkState(self.robot_id,
                                          self.joint_idx['torsoRoll'],
                                          computeLinkVelocity=1)
        chest_link_dis = np.array(chest_link_state[0]) - np.array(base_pos)
        chest_link_dis.resize(1, 3)
        chest_link_dis_base = np.transpose(R_i @ chest_link_dis.transpose())
        # chest_link_dis_yaw = np.transpose(Rz_i @ chest_link_dis.transpose())

        left_foot_link_state = p.getLinkState(self.robot_id,
                                              self.joint_idx['leftAnkleRoll'],
                                              computeLinkVelocity=0)
        left_foot_link_dis = np.array(left_foot_link_state[0]) - base_pos
        left_foot_link_dis.resize(1, 3)
        left_foot_link_dis_base = np.transpose(
            R_i @left_foot_link_dis.transpose())
        # left_foot_link_dis_yaw = np.transpose(
        #     Rz_i @left_foot_link_dis.transpose())
        # left_foot_link_dis_yaw = left_foot_link_dis

        right_foot_link_state = p.getLinkState(self.robot_id,
                                               self.joint_idx['rightAnkleRoll'],
                                               computeLinkVelocity=0)
        right_foot_link_dis = np.array(right_foot_link_state[0]) - base_pos
        right_foot_link_dis.resize(1, 3)
        right_foot_link_dis_base = np.transpose(
            R_i @right_foot_link_dis.transpose())
        # right_foot_link_dis_yaw = np.transpose(
        #     Rz_i @right_foot_link_dis.transpose())
        # right_foot_link_dis_yaw = right_foot_link_dis

        left_elbow_link_state = p.getLinkState(self.robot_id,
                                               self.joint_idx['leftElbowPitch'],
                                               computeLinkVelocity=0)
        left_elbow_link_dis = np.array(left_elbow_link_state[0]) - base_pos
        left_elbow_link_dis.resize(1, 3)
        left_elbow_link_dis_base = np.transpose(
            R_i @left_elbow_link_dis.transpose())
        # left_elbow_link_dis_yaw = np.transpose(
        #     Rz_i @left_elbow_link_dis.transpose())

        right_elbow_link_state = p.getLinkState(
            self.robot_id, self.joint_idx['rightElbowPitch'],
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

        p.addUserDebugLine(base_pos,
                           base_pos + orientation_base[0],
                           [0, 1, 0],
                           3,
                           1)  # pelvis to chest
        p.addUserDebugLine(base_pos,
                           base_pos + chest_link_dis_base[0],
                           [1, 0, 0],
                           3,
                           1)  # pelvis to chest
        p.addUserDebugLine(base_pos,
                           base_pos + left_foot_link_dis_base[0],
                           [1, 0, 0],
                           3,
                           1)  # pelvis to left foot
        p.addUserDebugLine(base_pos,
                           base_pos + right_foot_link_dis_base[0],
                           [1, 0, 0],
                           3,
                           1)  # pelvis to right foot
        p.addUserDebugLine(base_pos + chest_link_dis_base[0],
                           base_pos + left_elbow_link_dis_base[0],
                           [1, 0, 0],
                           3,
                           1)  # pelvis to left foot
        p.addUserDebugLine(base_pos + chest_link_dis_base[0],
                           base_pos + right_elbow_link_dis_base[0],
                           [1, 0, 0],
                           3,
                           1)  # pelvis to right foot

    def draw_COM(self):
        p.addUserDebugLine(self.COM_pos + np.array([0, 0, 2]),
                           self.COM_pos + np.array([0, 0, -2]),
                           [1, 0, 0],
                           5,
                           0.1)  # TODO rendering to draw COM

    def draw_support_polygon(self):
        hull = self.hull  # 2D
        if len(hull) <= 1:
            return
        support_polygon_centre = np.zeros(np.shape(hull[0]))
        for i in range(len(hull)):
            if i >= len(hull) - 1:  # end point
                start = np.array([hull[i][0], hull[i][1], 0.0])
                end = np.array([hull[0][0], hull[0][1], 0.0])
            else:
                start = np.array([hull[i][0], hull[i][1], 0.0])
                end = np.array([hull[i + 1][0], hull[i + 1][1], 0.0])
            # TODO rendering to draw support polygon
            p.addUserDebugLine(start, end, [0, 0, 1], 10, 0.1)
            support_polygon_centre += np.array(hull[i])

        support_polygon_centre /= len(hull)
        support_polygon_centre = np.array(
            [support_polygon_centre[0], support_polygon_centre[1], 0])

        p.addUserDebugLine(
            support_polygon_centre + np.array([0, 0, 2]),
            support_polygon_centre + np.array([0, 0, -2]),
            [0, 1, 0],
            5, 0.1)  # TODO rendering to draw COM

    def get_support_polygon(self):
        foot_ground_contact = []
        for side in ('right', 'left'):
            if self.is_ground_contact(side):
                ankle_roll_contact = p.getContactPoints(
                    self.robot_id,
                    self.plane_id,
                    self.joint_idx[side+'AnkleRoll'],
                    -1)
                ankle_pitch_contact = p.getContactPoints(
                    self.robot_id,
                    self.plane_id,
                    self.joint_idx[side+'AnklePitch'],
                    -1)
                foot_ground_contact.extend(ankle_roll_contact)
                foot_ground_contact.extend(ankle_pitch_contact)
        foot_ground_contact_point = np.array(
            [data[5] for data in foot_ground_contact])

        if len(foot_ground_contact_point) < 1:
            return np.array([[0.0, 0.0, 0.0]])

        # only use x and y coordinates
        self.contact_points = foot_ground_contact_point[:, 0:2]
        if len(self.contact_points) >= 4:
            hull = ConvexHull(self.contact_points)
            self.hull = self.contact_points[hull.vertices, :]
        else:
            self.hull = self.contact_points
        return self.hull

    def calculate_rejectable_force(self, period):
        """
        Parameters
        ----------
        period : int
            Impulse lasting period
        """
        foot_length = 0.26  # TODO check length of foot
        # TODO check whether COM is at the centre of foot
        V = np.array([self.target_COM_velocity(-foot_length / 2, 'x'),
                      self.target_COM_velocity(foot_length / 2, 'x')])
        F = V * self.total_mass / period
        return F

    def calculate_rejectable_force_xy(self, period):
        """
        Parameters
        ----------
        period : int
            Impulse lasting period
        """
        base_pos, base_quat = p.getBasePositionAndOrientation(self.robot_id)
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
        Fx_min = self.total_mass/tau*x_min/period
        Fx_max = self.total_mass/tau*x_max/period
        Fy_min = self.total_mass/tau*y_min/period
        Fy_max = self.total_mass/tau*y_max/period

        return Fx_min, Fx_max, Fy_min, Fy_max

    def test_kinematic(self):
        """NOTE: DOES NOT WORK! and I don't understadn what it's meant to do."""
        base_pos, base_quat = p.getBasePositionAndOrientation(self.robot_id)
        # base_orn = p.getEulerFromQuaternion(base_quat)
        R = self.transform(base_quat)
        T0 = np.eye(4)
        T0[0:3, 0:3] = R
        T0[0:3, 3] = np.array(base_pos)
        print(T0)

        # base_pos = self.link_COM_position['pelvisBase']
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
            joint_index = self.joint_idx[name]
            joint_name = name
            # for joint_index in range(p.getNumJoints(self.robot_id)):
            # info = p.getJointInfo(self.robot_id, joint_index)
            # joint_name = info[1].decode("utf-8")
            # self.joint_idx.update({joint_name: info[0]})
            # self.jointIdx.update({info[0]: joint_name})

            link_state = p.getLinkState(self.robot_id, joint_index,
                                        computeLinkVelocity=0)
            inertiaFramePos = link_state[2]
            inertiaFrameOrn = link_state[3]

            joint_info = p.getJointInfo(self.robot_id, joint_index)
            jointAxis = joint_info[13]
            parentFramePos = joint_info[14]
            parentFrameOrn = joint_info[15]
            parentIdx = joint_info[16]

            joint_state = p.getJointState(self.robot_id, joint_index)
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

            joint_pos_dict.update({joint_index: jointpos})
            T_dict.update({joint_index: T})
            COM_dict.update({joint_index: COM})
            COM_ref_dict.update({joint_index: np.array(link_state[0])})

            start = COM_ref_dict[parentIdx]
            end = np.array(link_state[0])
            p.addUserDebugLine(start, end, [1, 0, 0], 1, 1)
            start = COM_dict[parentIdx]
            end = COM
            p.addUserDebugLine(start, end, [0, 0, 1], 1, 1)
