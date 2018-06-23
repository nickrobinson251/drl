import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import os
import pybullet as p
from scipy.spatial import ConvexHull

from valkyrie.envs.filter import FilterClass
from valkyrie.envs.pd_controller import PDController
from valkyrie.envs.sensor_signal_process import calculate_COP


CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

KP_DEFAULTS = {
    "torsoYaw": 4500,
    "torsoPitch": 4500,
    "torsoRoll": 4500,
    "rightHipYaw": 500,
    "rightHipRoll": 1000,  # -0.49
    "rightHipPitch": 2000,  # -0.49
    "rightKneePitch": 2000,  # 1.205
    "rightAnklePitch": 3000,  # -0.71
    "rightAnkleRoll": 300,  # -0.71
    "leftHipYaw": 500,
    "leftHipRoll": 1000,  # -0.49
    "leftHipPitch": 2000,  # -0.49
    "leftKneePitch": 2000,  # 1.205
    "leftAnklePitch": 3000,  # -0.71
    "leftAnkleRoll": 300,  # -0.71
    "rightShoulderPitch": 700,
    "rightShoulderRoll": 1500,
    "rightShoulderYaw": 200,
    "rightElbowPitch": 200,
    "leftShoulderPitch": 700,
    "leftShoulderRoll": 1500,
    "leftShoulderYaw": 200,
    "leftElbowPitch": 200,
}

KD_DEFAULTS = {
    "torsoYaw": 30,
    "torsoPitch": 30,
    "torsoRoll": 30,
    "rightHipYaw": 20,
    "rightHipRoll": 30,  # -0.49
    "rightHipPitch": 30,  # -0.49
    "rightKneePitch": 30,  # 1.205
    "rightAnklePitch": 3,  # -0.71
    "rightAnkleRoll": 3,  # -0.71
    "leftHipYaw": 20,
    "leftHipRoll": 30,  # -0.49
    "leftHipPitch": 30,  # -0.49
    "leftKneePitch": 30,  # 1.205
    "leftAnklePitch": 3,  # -0.71
    "leftAnkleRoll": 3,  # -0.71
    "rightShoulderPitch": 10,
    "rightShoulderRoll": 30,
    "rightShoulderYaw": 2,
    "rightElbowPitch": 5,
    "leftShoulderPitch": 10,
    "leftShoulderRoll": 30,
    "leftShoulderYaw": 2,
    "leftElbowPitch": 5
}


class ValkyrieEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(
            self,
            max_time=16,  # in seconds
            initial_gap_time=0.01,  # in seconds
            isEnableSelfCollision=True,
            renders=True,
            pd_freq=500.0,
            physics_freq=1000.0,
            Kp=KP_DEFAULTS,
            Kd=KD_DEFAULTS,
            use_bullet_default_pd=True,
            logFileName=CURRENT_DIR,
            controlled_joints_list=None):
        self.seed()

        ##########
        # robot
        self.controllable_joints = np.array([
            "neckYaw",
            "upperNeckPitch",
            "lowerNeckPitch",
            "torsoYaw",
            "torsoPitch",
            "torsoRoll",
            "rightShoulderYaw",
            "rightShoulderPitch",
            "rightShoulderRoll",
            "leftShoulderYaw",
            "leftShoulderPitch",
            "leftShoulderRoll",
            "rightElbowPitch",
            "leftElbowPitch",
            "rightHipYaw",
            "rightHipPitch",
            "rightHipRoll",
            "leftHipYaw",
            "leftHipPitch",
            "leftHipRoll",
            "rightKneePitch",
            "leftKneePitch",
            "rightAnklePitch",
            "rightAnkleRoll",
            "leftAnklePitch",
            "leftAnkleRoll"])

        if controlled_joints_list:
            self.controlled_joints = np.array(controlled_joints_list)
        else:
            # TODO add control for hip roll joints "rightHipRoll", "leftHipRoll"
            self.controlled_joints = np.array([
                "torsoPitch",
                "rightHipPitch",
                "leftHipPitch",
                "rightKneePitch",
                "leftKneePitch",
                "rightAnklePitch",
                "leftAnklePitch",
                "rightAnkleRoll",
                "leftAnkleRoll"])
        self.uncontrolled_joints = np.array(
            [joint for joint in self.controllable_joints
             if joint not in self.controlled_joints])
        self.nu = len(self.controlled_joints)
        self.nq = len(self.controllable_joints)

        # nominal joint configuration
        self.default_joint_config = {
            "neckYaw": 0.0,
            "upperNeckPitch": 0.0,
            "lowerNeckPitch": 0.0,
            "torsoYaw": 0.0,
            "torsoPitch": 0.0,
            "torsoRoll": 0.0,
            "rightShoulderYaw": 0.0,
            "rightShoulderPitch": 0.300196631343,
            "rightShoulderRoll": 1.25,
            "leftShoulderYaw": 0.0,
            "leftShoulderPitch": 0.300196631343,
            "leftShoulderRoll": -1.25,
            "rightElbowPitch": 0.785398163397,
            "leftElbowPitch": -0.785398163397,
            "rightHipYaw": 0.0,
            "rightHipPitch": 0.0,  # -0.49
            "rightHipRoll": 0.0,
            "leftHipYaw": 0.0,
            "leftHipPitch": 0.0,  # -0.49
            "leftHipRoll": 0.0,
            "rightKneePitch": 0.0,  # 1.205
            "leftKneePitch": 0.0,  # 1.205
            "rightAnklePitch": 0.0,  # -0.71
            "rightAnkleRoll": 0.0,
            "leftAnklePitch": 0.0,  # -0.71
            "leftAnkleRoll": 0.0}
        self.default_joint_positions = np.array(
            [self.default_joint_config[j] for j in self.controllable_joints])
        self.default_base_pos = np.array([0, 0, 1.175])  # straight #1.025 bend
        self.default_base_orn = np.array([0, 0, 0, 1])  # x, y, z, w

        ##########
        # PD controller
        self.use_bullet_default_pd = use_bullet_default_pd
        self.pd_freq = pd_freq
        self.physics_freq = physics_freq
        self.Kp = Kp  # proportional gain
        self.Kd = Kd  # derivative gain
        # various things we need to track / update over time
        self.pd_controller = dict()  # TODO Add self defined PD controller
        self.pd_torque_filtered = dict()
        self.pd_torque_unfiltered = dict()
        self.pd_torque_adjusted = dict()
        self.hist_torque = dict()
        self.hist_torque_target = dict()
        for joint in self.controlled_joints:
            self.hist_torque[joint] = 0.0
            self.hist_torque_target[joint] = 0.0

        ##########
        # torque limits
        self.u_max = {
            "torsoYaw": 190,
            "torsoPitch": 150,
            "torsoRoll": 150,
            "rightShoulderPitch": 190,
            "rightShoulderRoll": 190,
            "rightShoulderYaw": 65,
            "rightElbowPitch": 65,
            "rightForearmYaw": 26,
            "rightWristRoll": 14,
            "rightWristPitch": 14,
            "leftShoulderPitch": 190,
            "leftShoulderRoll": 190,
            "leftShoulderYaw": 65,
            "leftElbowPitch": 65,
            "leftForearmYaw": 26,
            "leftWristRoll": 14,
            "leftWristPitch": 14,
            "rightHipYaw": 190,
            "rightHipRoll": 350,
            "rightHipPitch": 350,
            "rightKneePitch": 350,
            "rightAnklePitch": 205,
            "rightAnkleRoll": 205,
            "leftHipYaw": 190,
            "leftHipRoll": 350,
            "leftHipPitch": 350,
            "leftKneePitch": 350,
            "leftAnklePitch": 205,
            "leftAnkleRoll": 205,
            "lowerNeckPitch": 50,
            "upperNeckPitch": 50,
            "neckYaw": 50}

        # velocity limits
        self.v_max = {
            "torsoYaw": 5.89,
            "torsoPitch": 9,
            "torsoRoll": 9,
            "rightShoulderPitch": 5.89,
            "rightShoulderRoll": 5.89,
            "rightShoulderYaw": 11.5,
            "rightElbowPitch": 11.5,
            "leftShoulderPitch": 5.89,
            "leftShoulderRoll": 5.89,
            "leftShoulderYaw": 11.5,
            "leftElbowPitch": 11.5,
            "rightHipYaw": 5.89,
            "rightHipRoll": 7,
            "rightHipPitch": 6.11,
            "rightKneePitch": 6.11,
            "rightAnklePitch": 11,
            "rightAnkleRoll": 11,
            "leftHipYaw": 5.89,
            "leftHipRoll": 7,
            "leftHipPitch": 6.11,
            "leftKneePitch": 6.11,
            "leftAnklePitch": 11,
            "leftAnkleRoll": 11,
            "lowerNeckPitch": 5,
            "upperNeckPitch": 5,
            "neckYaw": 5}

        ##########
        # Centre of mass.
        # COM global coordinate
        self.COM_pos = np.array([0.0, 0.0, 0.0])
        self.COM_vel = np.array([0.0, 0.0, 0.0])
        # COM local coordinate  w.r.t centre of mass of foot link.
        # robot operates solely on the sagittal plane, the orientation of global
        # frame and local frame is aligned
        self.COM_pos_local = np.array([0.0, 0.0, 0.0])
        self.COM_pos_local_filter = np.array([0.0, 0.0, 0.0])
        self.COM_pos_local_surrogate = np.array([0.0, 0.0, 0.0])
        self.support_polygon_centre = np.array([[0.0, 0.0, 0.0]])
        self.support_polygon_centre_surrogate = np.array([[0.0, 0.0, 0.0]])

        # TODO pelvis acceleration
        self.pelvis_acc_gap_step = 10  # 30
        self.pelvis_acc = np.array([0.0, 0.0, 0.0])
        self.pelvis_acc_base = np.array([0.0, 0.0, 0.0])
        self.pelvis_vel_history_array = np.zeros((self.pelvis_acc_gap_step, 3))
        self.pelvis_vel_history = np.array([0.0, 0.0, 0.0])
        self.pelvis_vel_history_base = np.array([0.0, 0.0, 0.0])

        ##########
        # Simulation
        # reset timestep in PD control loop
        self.initial_gap_steps = initial_gap_time * self.pd_freq
        self.max_steps = max_time * self.pd_freq  # PD control loop timestep
        self._actionRepeat = int(self.physics_freq / self.pd_freq)
        self._dt_physics = (1. / self.physics_freq)
        self._dt = self._dt_physics  # PD control loop timestep
        self._dt_pd = (1. / self.pd_freq)
        self._dt_filter = self._dt_pd  # filter time step

        ##########
        # Setup Simulation
        self._renders = renders
        if self._renders:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        self.num_states = 51  # 60  # 2 * 26 + 2 * 6
        self._setup_simulation(self.default_base_pos, self.default_base_orn)
        self.steps_taken = 0

        ##########
        # Define spaces
        maximum_float = np.finfo(np.float32).max
        observation_high = np.array([maximum_float * self.num_states])
        self.observation_space = spaces.Box(-observation_high, observation_high,
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(self.nu)

    def disconnect(self):
        """Disconnect from physics simulator."""
        p.disconnect()

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
        self.g = 9.806651
        # Physics engine parameter default solver iteration = 50
        p.setPhysicsEngineParameter(numSolverIterations=25, erp=0.2)
        p.resetSimulation()
        p.setRealTimeSimulation(0)
        p.setGravity(0, 0, -self.g)
        p.setTimeStep(self._dt)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        plane_urdf = os.path.join(CURRENT_DIR, "assets", "plane", "plane.urdf")
        self.plane = p.loadURDF(plane_urdf, basePosition=[0, 0, 0],
                                useFixedBase=True)
        robot_urdf = os.path.join(
            CURRENT_DIR,
            "assets",
            "valkyrie_bullet_mass_sims"
            "_modified_foot_collision_box"
            "_modified_self_collision"
            ".urdf")
        self.robot = p.loadURDF(
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
            self.robot, self.joint_idx['leftAnkleRoll'], True)
        p.enableJointForceTorqueSensor(
            self.robot, self.joint_idx['rightAnkleRoll'], True)

        for _ in range(int(self.initial_gap_steps)):  # PD loop time steps
            for _ in range(int(self._actionRepeat)):
                p.stepSimulation()
            # update information
            self.calculate_link_COM_position()
            self.calculate_link_COM_velocity()
            self.calculate_COM_position()
            self.calculate_COM_velocity()
            self.calculate_pelvis_acc()  # TODO calculate pelvis acceleration
            self.calculate_COP()
            # initialize filter value
            self.initialize_filtering()
            self.perform_filtering()

        self.calculate_ground_contact_points()
        self.get_support_polygon()  # hull of support polygon
        # record history of joint torque output for PD control
        self.action = dict()
        self.hist_torque = dict()
        for joint in self.controlled_joints:
            joint_state = p.getJointState(self.robot, self.joint_idx[joint])
            self.hist_torque[joint] = joint_state[3]
            self.action[joint] = 0.0  # initialize PD control input

    def step(self, action, pelvis_push_force=[0, 0, 0]):
        """Take action, get observation, reward, done (flag), info (empty)."""
        torque_dict = self.calculate_PD_torque(action)
        applied_torque_dict = torque_dict
        prev_torque_dict = self.hist_torque_target
        self.set_control(action)
        # higher frequency for physics simulation
        for i in range(int(self._actionRepeat)):
            # displacement from COM to the centre of the pelvis
            p.applyExternalForce(self.robot,
                                 -1,  # base
                                 forceObj=pelvis_push_force,
                                 posObj=[0, 0.0035, 0],
                                 flags=p.LINK_FRAME)
            for joint, torque in torque_dict.items():
                applied_torque_dict[joint] = np.clip(
                    2*torque - prev_torque_dict[joint],
                    -self.u_max[joint], self.u_max[joint])
                self.pd_torque_filtered[joint] = self.pd_controller[joint].u
                self.pd_torque_unfiltered[joint] = \
                    self.pd_controller[joint].u_raw
                self.pd_torque_adjusted[joint] = applied_torque_dict[joint]
                # applied_torque_dict[joint] = self.pd_controller[joint].u_adj
                # add momentum turn to reduce delay

            self.set_control(action)
            self.set_PD_velocity_control(torque_dict)
            # self.set_PD_velocity_control(applied_torque_dict)
            # self.set_PD_position_control(applied_torque_dict, action)
            # self.set_PD_torque_control(torque_dict)
            p.stepSimulation()  # one simulation step

        prev_torque_dict.update(torque_dict)
        self.hist_torque_target.update(prev_torque_dict)
        # update COM information
        self.calculate_link_COM_position()
        self.calculate_link_COM_velocity()
        self.calculate_COM_position()
        self.calculate_COM_velocity()
        # TODO calculate pelvis acceleration
        self.calculate_pelvis_acc()
        self.calculate_COP()
        self.calculate_ground_contact_points()
        # perform filtering
        self.perform_filtering()
        self._observation = self.get_extended_observation()  # filtered
        self.steps_taken += 1
        reward, _ = self.reward()  # balancing
        done = self._termination()

        for joint in self.controlled_joints:  # TODO
            joint_state = p.getJointState(self.robot, self.joint_idx[joint])
            self.hist_torque.update({joint: joint_state[3]})
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
        for jointNo in range(p.getNumJoints(self.robot)):
            info = p.getJointInfo(self.robot, jointNo)
            p.changeDynamics(self.robot, info[0],)

        # set plane and ankle friction and restitution
        for joint in ('rightAnklePitch', 'rightAnkleRoll', 'leftAnklePitch',
                      'leftAnkleRoll'):
            p.changeDynamics(self.robot,
                             self.joint_idx[joint],
                             restitution=restitution,
                             lateralFriction=lateralFriction,
                             spinningFriction=spinningFriction,
                             rollingFriction=rollingFriction)
        p.changeDynamics(self.plane, -1,  # -1 for base
                         restitution=restitution,
                         lateralFriction=lateralFriction,
                         spinningFriction=spinningFriction,
                         rollingFriction=rollingFriction)

    def _setup_filter(self):
        # filtering for COP
        # TODO test calculation and filtering for COP
        self.COP_filter_method = calculate_COP(10, 10, self._dt_filter, 1)

        # filtering states
        # TODO test filtering for state and reward
        self.state_filter_method = {}
        for i in range(self.num_states):
            # TODO experiment with different cutoff frequencies
            self.state_filter_method[i] = FilterClass()
            self.state_filter_method[i].butterworth(T=self._dt_filter,
                                                    cutoff=10,
                                                    N=1)
        self.COM_pos_local_filter_method = {}
        for i in range(3):
            self.COM_pos_local_filter_method[i] = FilterClass()
            self.COM_pos_local_filter_method[i].butterworth(T=self._dt_filter,
                                                            cutoff=10,
                                                            N=1)
        for joint in self.controlled_joints:
            # TODO Add self defined PD controller
            if joint in self.Kp:
                self.pd_controller.update({joint: PDController(
                    gains=[self.Kp[joint], self.Kd[joint]],
                    u_max=self.u_max[joint],
                    v_max=self.v_max[joint],
                    name=joint,
                    is_filter=[True, True, False],
                    T=self._dt_filter,
                    cutoff=[10, 10, 10],
                    N=1)})  # 250
            else:  # PD gains not defined
                continue

    def _setup_camera(
            self,
            cameraDistance=3,
            cameraYaw=45,
            cameraPitch=0,
            cameraTargetPosition=[0, 0, 0.9]):
        """Turn on camera at given position.

        Wraps pybullet.resetDebugVisualizerCamera

        To be side-on and view sagittal balancing
               cameraYaw=0, cameraTargetPosition=[0, 0, 0.9]
        To be head-on and view lateral balancing
               cameraYaw=90, cameraTargetPosition=[0.5, 0, 0.9]

        Parameters
        ----------
        cameraDistance : int (default 3)
        cameraYaw : int (default 45)
        cameraPitch : int (default 0)
        cameraTargetPositions : list (default 0, 0, 0.9)
        """
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        p.resetDebugVisualizerCamera(cameraDistance=cameraDistance,
                                     cameraYaw=cameraYaw,
                                     cameraPitch=cameraPitch,
                                     cameraTargetPosition=cameraTargetPosition)

    def _setup_joint_mapping(self):
        """Store joint name-to-index mapping."""
        self.joint_idx = dict()
        self.joint_idx2name = dict()
        for joint_number in range(p.getNumJoints(self.robot)):
            info = p.getJointInfo(self.robot, joint_number)
            joint_index = info[0]
            joint_name = info[1].decode("utf-8")
            self.joint_idx[joint_name] = joint_index
            self.joint_idx2name[joint_index] = joint_name

    def render(self, mode='human', close=False):
        p.addUserDebugLine(self.COM_pos + np.array([0, 0, 2]),
                           self.COM_pos + np.array([0, 0, -2]),
                           [1, 0, 0],
                           5,
                           0.1)  # TODO rendering to draw COM
        p.addUserDebugLine(self.support_polygon_centre[0]+np.array([0, 0, 2]),
                           self.support_polygon_centre[0]+np.array([0, 0, -2]),
                           [0, 1, 0],
                           5,
                           0.1)  # TODO rendering to draw support polygon
        p.addUserDebugLine(self.support_polygon_centre[0]+np.array([2, 0, 0]),
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

    def _termination(self):
        """Same as self.is_fallen. True if robot has fallen, else False.."""
        # return (self.steps_taken > self.max_steps) or (self.is_fallen())
        return self.is_fallen()

    def reset_joint_states(self, base_pos=None, base_orn=None):
        """Reset joints to default configuration, and base velocity to zero."""
        if base_pos is None:
            base_pos = self.default_base_pos
        if base_orn is None:
            base_orn = self.default_base_orn
        p.resetBasePositionAndOrientation(self.robot, base_pos, base_orn)
        p.resetBaseVelocity(self.robot, [0, 0, 0], [0, 0, 0])
        for jointName in self.default_joint_config:
            p.resetJointState(self.robot,
                              self.joint_idx[jointName],
                              targetValue=self.default_joint_config[jointName],
                              targetVelocity=0)

    def initialize_filtering(self):
        """Initialise filters for states, PD controller, and COM position."""
        observation = self.get_observation()
        for i in range(self.num_states):
            self.state_filter_method[i].initializeFilter(observation[i])
        # TODO filtering for PD controller
        for j in self.controlled_joints:
            position, velocity, _, torque = p.getJointState(self.robot,
                                                            self.joint_idx[j])
            if j in self.Kp:  # if PD gains is defined for joint j
                self.pd_controller[j].reset(position, velocity, torque)
        # Initialise filtering for COM position
        for i in range(3):
            self.COM_pos_local_filter_method[i].initializeFilter(
                self.COM_pos_local[i])
            self.COM_pos_local_filter[i] = \
                self.COM_pos_local_filter_method[i].y[0]

    def perform_filtering(self):  # TODO test filtering
        """Apply filtering of states, PD controller, and COM position."""
        observation = self.get_observation()
        for i in range(self.num_states):
            self.state_filter_method[i].applyFilter(observation[i])
        # Binay state filter
        # self.left_contact_filter_method.applyFilter(observation[39])
        # self.right_contact_filter_method.applyFilter(observation[59])

        # TODO filtering for PD controller
        for j in self.controlled_joints:
            position, velocity, _, _ = p.getJointState(self.robot,
                                                       self.joint_idx[j])
            if j in self.Kp:  # if PD gains is defined for joint j
                self.pd_controller[j].updateMeasurements(position, velocity)
        # perform filtering of COM position
        for i in range(3):
            self.COM_pos_local_filter_method[i].applyFilter(
                self.COM_pos_local[i])
            self.COM_pos_local_filter[i] = \
                self.COM_pos_local_filter_method[i].y[0]

    def get_observation(self):
        x_observation = np.zeros((self.num_states,))

        # This may not be the Pelvis CoM position _exactly_ but should be fine,
        # otherwise can apply local transformation
        _, base_quat = p.getBasePositionAndOrientation(self.robot)
        base_orn = p.getEulerFromQuaternion(base_quat)
        base_pos_vel, base_orn_vel = p.getBaseVelocity(self.robot)

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
            self.robot, self.joint_idx['torsoRoll'],
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
            self.robot, self.joint_idx['torsoPitch'])
        x_observation[11] = torso_pitch_joint_state[0]
        x_observation[12] = (torso_pitch_joint_state[1]
                             / self.v_max['torsoPitch'])

        right_hip_roll_joint_state = p.getJointState(
            self.robot, self.joint_idx['rightHipRoll'])
        x_observation[13] = right_hip_roll_joint_state[0]  # position
        # velocity
        x_observation[14] = (right_hip_roll_joint_state[1]
                             / self.v_max['rightHipRoll'])
        right_hip_pitch_joint_state = p.getJointState(
            self.robot, self.joint_idx['rightHipPitch'])
        x_observation[15] = right_hip_pitch_joint_state[0]  # position
        # velocity
        x_observation[16] = (right_hip_pitch_joint_state[1]
                             / self.v_max['rightHipPitch'])
        right_knee_pitch_joint_state = p.getJointState(
            self.robot, self.joint_idx['rightKneePitch'])
        x_observation[17] = right_knee_pitch_joint_state[0]  # position
        # velocity
        x_observation[18] = (right_knee_pitch_joint_state[1]
                             / self.v_max['rightKneePitch'])
        right_ankle_pitch_joint_state = p.getJointState(
            self.robot, self.joint_idx['rightAnklePitch'])
        x_observation[19] = right_ankle_pitch_joint_state[0]
        x_observation[20] = (right_ankle_pitch_joint_state[1]
                             / self.v_max['rightAnklePitch'])
        right_ankle_roll_joint_state = p.getJointState(
            self.robot, self.joint_idx['rightAnkleRoll'])
        x_observation[21] = right_ankle_roll_joint_state[0]
        x_observation[22] = (right_ankle_roll_joint_state[1]
                             / self.v_max['rightAnkleRoll'])

        right_foot_link_state = p.getLinkState(
            self.robot, self.joint_idx['rightAnkleRoll'],
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
            self.robot, self.joint_idx['leftHipRoll'])
        x_observation[26] = left_hip_roll_joint_state[0]  # position
        # velocity
        x_observation[27] = (left_hip_roll_joint_state[1]
                             / self.v_max['leftHipRoll'])
        left_hip_pitch_joint_state = p.getJointState(
            self.robot, self.joint_idx['leftHipPitch'])
        x_observation[28] = left_hip_pitch_joint_state[0]  # position
        # velocity
        x_observation[29] = (left_hip_pitch_joint_state[1]
                             / self.v_max['leftHipPitch'])
        left_knee_pitch_joint_state = p.getJointState(
            self.robot, self.joint_idx['leftKneePitch'])
        x_observation[30] = left_knee_pitch_joint_state[0]  # position
        # velocity
        x_observation[31] = (left_knee_pitch_joint_state[1]
                             / self.v_max['leftKneePitch'])
        left_ankle_pitch_joint_state = p.getJointState(
            self.robot, self.joint_idx['leftAnklePitch'])
        x_observation[32] = left_ankle_pitch_joint_state[0]
        x_observation[33] = (left_ankle_pitch_joint_state[1]
                             / self.v_max['leftAnklePitch'])
        left_ankle_roll_joint_state = p.getJointState(
            self.robot, self.joint_idx['leftAnkleRoll'])
        x_observation[34] = left_ankle_roll_joint_state[0]
        x_observation[35] = (left_ankle_roll_joint_state[1]
                             / self.v_max['leftAnkleRoll'])

        left_foot_link_state = p.getLinkState(
            self.robot, self.joint_idx['leftAnkleRoll'],
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
            observation_filtered[i] = self.state_filter_method[i].y[0]
        # TODO binary values should not be filtered.
        # observation_filtered[48] = observation[48]
        # observation_filtered[49] = observation[49]
        # observation_filtered[39]=self.left_contact_filter_method.y
        # observation_filtered[59]=self.right_contact_filter_method.y
        return observation_filtered

    def get_reading(self):
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
            joint_state = p.getJointState(self.robot, self.joint_idx[joint])
            (readings[joint+'Angle'], readings[joint+'Velocity'], _,
             readings[joint+'_torque']) = joint_state

        left_foot_link_state = p.getLinkState(self.robot,
                                              self.joint_idx['leftAnkleRoll'],
                                              computeLinkVelocity=0)  # 0 for no
        readings['leftFootPitch'] = left_foot_link_state[1][1]
        right_foot_link_state = p.getLinkState(self.robot,
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
        readings['COM_pos_local_filter'] = np.array(self.COM_pos_local_filter)
        readings['support_polygon_centre'] = self.support_polygon_centre[0]
        readings['support_polygon_centre_surrogate'] = \
            self.support_polygon_centre_surrogate[0]

        # record joint torque calculated by self defined PD controller
        for joint in self.controlled_joints:
            if joint in self.Kp:  # if PD gains are defined
                position = self.pd_controller[joint].filtered_position
                velocity = self.pd_controller[joint].filtered_velocity
                # torque = self.pd_controller[joint].control(self.action[joint],
                # 0)
                torque = self.pd_controller[joint].u
                P_torque = self.pd_controller[joint].u_e
                D_torque = self.pd_controller[joint].u_de
                readings[joint+'PDPosition'] = position
                readings[joint+'PDVelocity'] = velocity
                readings[joint+'PDPositionAdjusted'] = \
                    self.pd_controller[joint].adjusted_position
                readings[joint+'PDVelocityAdjusted'] = \
                    self.pd_controller[joint].adjusted_velocity
                readings[joint+'PD_torque'] = torque
                readings[joint+'PD_torque_Kp'] = P_torque
                readings[joint+'PD_torque_Kd'] = D_torque
                readings[joint+'PDPositionKalman'] = \
                    self.pd_controller[joint].kalman_filtered_position
                readings[joint+'PDVelocityKalman'] = \
                    self.pd_controller[joint].kalman_filtered_velocity
                readings[joint+'PD_torqueWithAdjustedFeedback'] = \
                    self.pd_controller[joint].u_adj
                readings[joint+'PDFiltered_torque'] = \
                    self.pd_controller[joint].u
                readings[joint+'PDUnfiltered_torque'] = \
                    self.pd_controller[joint].u_raw
                if len(self.pd_torque_adjusted) > 0:  # non-empty if step taken
                    readings[joint+'PDAdjusted_torque'] = \
                        self.pd_torque_adjusted[joint]

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

        # readings.update({'torsoPitchAngle50HzFiltered':
        # self.pd_controller['torsoPitch'].measured_position})
        # readings.update({'leftHipPitchAngle50HzFiltered':
        # self.pd_controller['leftHipPitch'].measured_position})
        # readings.update({'leftKneePitchAngle50HzFiltered':
        # self.pd_controller['leftKneePitch'].measured_position})
        # readings.update({'leftAnklePitchAngle50HzFiltered':
        # self.pd_controller['leftAnklePitch'].measured_position})
        # readings.update({'rightHipPitchAngle50HzFiltered':
        # self.pd_controller['rightHipPitch'].measured_position})
        # readings.update({'rightKneePitchAngle50HzFiltered':
        # self.pd_controller['rightKneePitch'].measured_position})
        # readings.update({'rightAnklePitchAngle50HzFiltered':
        # self.pd_controller['rightAnklePitch'].measured_position})

        readings['pelvis_acc_global'] = self.pelvis_acc
        readings['pelvis_acc_base'] = self.pelvis_acc_base
        readings['pelvis_vel_global'] = self.pelvis_vel_history
        readings['pelvis_vel_base'] = self.pelvis_vel_history_base

        # record individual terms in reward
        _, reward_term = self.reward()
        readings.update(reward_term)
        return readings

    def calculate_COM_position(self):
        """Get position of Centre of Mass."""
        summ = np.zeros(3)
        for link, mass in self.link_masses.items():
            summ += np.array(self.link_COM_position[link]) * mass
        summ /= self.total_mass
        self.COM_pos = summ  # update global COM position

        # update local COM position w.r.t centre of support polygon
        right_foot_info = p.getLinkState(self.robot,
                                         self.joint_idx['rightAnkleRoll'],
                                         computeLinkVelocity=0)
        left_foot_info = p.getLinkState(self.robot,
                                        self.joint_idx['leftAnkleRoll'],
                                        computeLinkVelocity=0)

        # T = np.array([[0.066],[0],[-0.056]])#Transformation from the link
        # frame position to geometry centre w.r.t link frame
        # Transformation from the link frame position to centre of bottom of
        # foot w.r.t link frame
        T = np.array([[0.045], [0], [-0.088]])
        right_quat = right_foot_info[1]
        left_quat = left_foot_info[1]
        right_T1 = self.transform(right_quat) @ T
        left_T1 = self.transform(left_quat) @ T
        right_foot_bottom_centre = right_foot_info[4] + right_T1.T
        left_foot_bottom_centre = left_foot_info[4] + left_T1.T

        # support polygon changes if their is foot contact
        if self.is_ground_contact('right'):
            if self.is_ground_contact('left'):
                self.support_polygon_centre = (
                    right_foot_bottom_centre + left_foot_bottom_centre) / 2.0
            else:
                self.support_polygon_centre = right_foot_bottom_centre
        elif self.is_ground_contact('left'):
            self.support_polygon_centre = left_foot_bottom_centre
        # else both feet not in contact. Maintain current support_polygon value.

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
        base_pos_vel, base_orn_vel = p.getBaseVelocity(self.robot)
        self.link_COM_velocity["pelvisBase"] = base_pos_vel
        for joint, idx in self.joint_idx.items():
            info = p.getLinkState(self.robot, idx, computeLinkVelocity=1)
            self.link_COM_velocity[joint] = info[6]
        return self.link_COM_velocity

    # def target_COM_velocity(self, x, axis='x'):
    #     # input x is the target displacement w.r.t centre of foot
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

    def calculate_link_COM_position(self):
        """Compute centre of mass position for all links and base."""
        self.link_COM_position = {}
        base_pos, base_quat = p.getBasePositionAndOrientation(self.robot)
        # base_orn = p.getEulerFromQuaternion(base_quat)
        # Transformation from the pelvis base position to pelvis COM w.r.t
        # local pelvis frame
        # T = np.array([[-0.00532], [-0.003512], [-0.0036]])
        # T1 = (self.transform(base_quat)) @ T  # using quaternion
        # T1 = (self.rotZ(base_orn[2]))@(self.rotY(base_orn[1]))
        # @(self.rotX(base_orn[0]))@T  # using Euler angles
        # base_com = base_pos + T1.T  # [[]]
        # base_com = base_com[0]  # []
        # self.link_COM_position.update({"pelvisBase": base_com})
        # TODO check if base position is the COM of the pelvis base position is
        # the COM of the pelvis
        self.link_COM_position["pelvisBase"] = np.array(base_pos)
        for joint, idx in self.joint_idx.items():
            info = p.getLinkState(self.robot, idx, computeLinkVelocity=0)
            self.link_COM_position[joint] = info[0]
        return self.link_COM_position

    def calculate_link_masses(self):
        """Compute link mass and total mass information."""
        info = p.getDynamicsInfo(self.robot, -1)  # for base link
        self.link_masses = dict()
        self.link_masses["pelvisBase"] = info[0]
        self.total_mass = info[0]
        for joint, idx in self.joint_idx.items():
            info = p.getDynamicsInfo(self.robot, idx)
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
                    self.robot,
                    self.plane,
                    self.joint_idx[side+'AnkleRoll'],
                    -1)
                ankle_pitch_contact = p.getContactPoints(
                    self.robot,
                    self.plane,
                    self.joint_idx[side+'AnklePitch'],
                    -1)
                # use extend not append because getContactPoints returns a list
                foot_ground_contact_info.extend(ankle_roll_contact)
                foot_ground_contact_info.extend(ankle_pitch_contact)
        # get just x, y cartesian coordinates of contact position on robot
        self.contact_points = np.array(
            [info[5][0:2] for info in foot_ground_contact_info])
        return self.contact_points

    def get_support_polygon(self):
        foot_ground_contact = []
        for side in ('right', 'left'):
            if self.is_ground_contact(side):
                ankle_roll_contact = p.getContactPoints(
                    self.robot,
                    self.plane,
                    self.joint_idx[side+'AnkleRoll'],
                    -1)
                ankle_pitch_contact = p.getContactPoints(
                    self.robot,
                    self.plane,
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

    def toggle_rendering(self):
        """Turn visualisation on/off."""
        if self._renders:  # It's on, so turn it off
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            self._renders = False
        else:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            self._renders = True

    def debug(self):  # for debugging
        # Transformation from the link frame position to geometry centre w.r.t
        # link frame
        T = np.array([[0.066], [0], [-0.056]])
        # Transformation from the link frame position to centre of bottom of
        # foot w.r.t link frame
        T1 = np.array([[0.066], [0], [-0.088]])
        for side in ('right', 'left'):
            (link_world_pos, _, local_inertia_pos, _, link_frame_pos,
             link_frame_orn) = p.getLinkState(self.robot,
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
        base_pos, base_quat = p.getBasePositionAndOrientation(self.robot)
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
        """True if robot is in contact with ground.

        Parameters
        ----------
        side : str (optional)
            Either 'left' or 'right', else checks both sides.
       """
        # TODO ground contact detection using contact point
        if side:
            ground_contact_points = p.getContactPoints(
                self.robot, self.plane, self.joint_idx[side+'AnkleRoll'], -1)
            return len(ground_contact_points) > 0
        else:
            is_left_contact = self.is_ground_contact('left')
            is_right_contact = self.is_ground_contact('right')
            return is_left_contact or is_right_contact

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
        # return 1/(1+np.exp(-(torque-9)))#sigmoid function
        return 1 / (1 + np.exp(-0.3 * (torque - 4)))

    def is_fallen(self):
        """Return True if robot has fallen, else False."""
        is_fallen = False
        base_pos, _ = p.getBasePositionAndOrientation(self.robot)
        # chest_link_state = p.getLinkState(self.robot,
        #                                   self.joint_idx['torsoRoll'],
        #                                   computeLinkVelocity=1)
        # if chest_link_state[0][2] < 0.5 or base_pos[2] < 0.5:
        #     is_fallen = True
        # exclude feet from contact points that count as a fall
        link = list(self.controllable_joints)
        link.remove('leftAnklePitch')
        link.remove('rightAnklePitch')
        link.remove('leftAnkleRoll')
        link.remove('rightAnkleRoll')
        for joint in link:
            contact_points = p.getContactPoints(self.robot,
                                                self.plane,
                                                self.joint_idx[joint],
                                                -1)  # -1 for base
        if len(contact_points) > 0:
            is_fallen = True
        if base_pos[2] <= 0.3:  # TODO check is_fallen criteria
            is_fallen = True
        if self.COM_pos[2] <= 0.3:
            is_fallen = True
        # COM_z = self.COM_pos_local[2]  # not suitable for walking
        # COM_xy = np.linalg.norm([self.COM_pos_local[0],self.COM_pos_local[1]])
        # if COM_z<COM_xy:#out of friction cone
        #     is_fallen = True
        return is_fallen

    def caclulate_foot_COP(self, side):
        """Calculate Centre of Pressure for foot of chosen side.

        Parameters
        ----------
        side : str
            Either 'left' or 'right'
        """
        foot_ground_contact = []
        # TODO ground contact detection using contact point
        ankle_roll_contact = p.getContactPoints(
            self.robot, self.plane, self.joint_idx[side + 'AnkleRoll'], -1)
        ankle_pitch_contact = p.getContactPoints(
            self.robot, self.plane, self.joint_idx[side + 'AnklePitch'], -1)

        foot_ground_contact.extend(ankle_roll_contact)
        foot_ground_contact.extend(ankle_pitch_contact)
        if not self.is_ground_contact(side):  # no contact
            return [
                0.0, 0.0, 0.0], [
                0.0, 0.0, 0.0], False  # COP does not exists
        # print(len(ankle_roll_contact))
        pCOP = np.array([0, 0, 0])  # position of Center of pressure
        # force among the x,y,z axis of world frame
        contactForce = np.array([0, 0, 0])
        # print(len(foot_ground_contact))
        for i in range(len(foot_ground_contact)):
            # contact normal of foot pointing towards plane
            contactNormal = np.array(foot_ground_contact[i][7])
            # contact normal of plane pointing towards foot
            contactNormal = -contactNormal
            contactNormalForce = np.array(foot_ground_contact[i][9])
            contactPosition = np.array(
                foot_ground_contact[i][5])  # position on plane
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
        # contactForce = contactForce / len(foot_ground_contact)
        return pCOP, contactForce, True

    def caclulate_foot_COP2(self, side):
        """Calculate Centre of Pressure for foot of chosen side.
        (NOTE: don't understand difference to caclulate_foot_COP method)

        Parameters
        ----------
        side : str
            Either 'left' or 'right'
        """
        joint_state = p.getJointState(self.robot,
                                      self.joint_idx[side+'AnkleRoll'])
        joint_reaction_force = joint_state[2]  # [Fx, Fy, Fz, Mx, My, Mz]
        Fx, Fy, Fz, Mx, My, Mz = -np.array(joint_reaction_force)
        foot_info = p.getLinkState(
            self.robot, self.joint_idx[side + 'AnkleRoll'],
            computeLinkVelocity=0)
        # link_frame_pos = foot_info[4]
        link_frame_pos = foot_info[0]
        # Transformation from link frame position to geometry centre w.r.t
        # link frame
        # T = np.array([[0.066],[0],[-0.056]])
        # Transformation from link frame position to centre of bottom of foot
        # w.r.t link frame
        # T = np.array([[0.044], [0], [ -0.088]])
        # right_quat = right_foot_info[1]
        # left_quat = left_foot_info[1]
        # right_T1 = (self.transform(right_quat)) @ T
        # left_T1 = (self.transform(left_quat)) @ T
        # print(right_T1)
        # print(left_T1)
        # right_foot_bottom_centre = right_foot_info[4] + right_T1.T
        # left_foot_bottom_centre = left_foot_info[4] + left_T1.T

        # TODO ground contact detection using contact point
        if not self.is_ground_contact(side):  # no contact
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
        # print(len(foot_ground_contact))
        return pCOP, contactForce, True

    def calculate_COP(self):
        """Calculate centre of pressure. (NOTE: don't understand this yet)"""
        # TODO ground contact detection using contact point
        foot_ground_contact = []
        ankle_roll_contact = p.getContactPoints(self.robot,
                                                self.plane,
                                                self.joint_idx['leftAnkleRoll'],
                                                -1)
        ankle_pitch_contact = p.getContactPoints(
            self.robot, self.plane, self.joint_idx['leftAnklePitch'], -1)
        foot_ground_contact.extend(ankle_roll_contact)
        foot_ground_contact.extend(ankle_pitch_contact)
        left_contact_info = foot_ground_contact

        foot_ground_contact = []
        ankle_roll_contact = p.getContactPoints(
            self.robot, self.plane, self.joint_idx['rightAnkleRoll'], -1)
        ankle_pitch_contact = p.getContactPoints(
            self.robot,
            self.plane,
            self.joint_idx['rightAnklePitch'],
            -1)
        foot_ground_contact.extend(ankle_roll_contact)
        foot_ground_contact.extend(ankle_pitch_contact)
        right_contact_info = foot_ground_contact

        self.COP_info = self.COP_filter_method(right_contact_info,
                                               left_contact_info)  # right first
        self.COP_info_filtered = self.COP_filter_method.get_filtered_COP()
        return self.COP_info_filtered

    def calculate_pelvis_acc(self):
        """Calculate pelvis acceleration. (NOTE: don't understand this yet)"""
        # TODO add pelvis acceleration
        base_pos, base_quat = p.getBasePositionAndOrientation(self.robot)
        # base_orn = p.getEulerFromQuaternion(base_quat)
        R = self.transform(base_quat)
        R_i = np.linalg.inv(R)
        pelvis_vel, _ = p.getBaseVelocity(self.robot)
        stop = len(self.pelvis_vel_history_array) - 2
        self.pelvis_vel_history_array[0:stop, :, None] = \
            self.pelvis_vel_history_array[1:stop+1, :, None]  # shift element
        for i in range(len(self.pelvis_vel_history_array)-1):
            self.pelvis_vel_history_array[i] = \
                self.pelvis_vel_history_array[i+1]

        self.pelvis_vel_history_array[len(
            self.pelvis_vel_history_array) - 1, :] = np.array(pelvis_vel)
        x = self.pelvis_vel_history_array[:, 0]
        # print(x)
        y = self.pelvis_vel_history_array[:, 1]

        z = self.pelvis_vel_history_array[:, 2]

        t = np.arange(len(self.pelvis_vel_history_array)
                      ) * self._dt  # time step
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
        return summ

    def calculate_PD_torque(self, u):
        """Calculate torque using self desinged PD controller.

        NOTE: this method updates self.action

        Parameters
        ----------
        u : list of length self.nu
            target position for controlled joint 0, 1, ..., nu

        Returns
        -------
        torques : dict
            {joint: torque} as computed by self.pd_controller
        """
        # Set control. pd_controller call will fail if any element of u is NaN
        torques = dict()
        for i in range(self.nu):
            jointName = self.controlled_joints[i]
            self.action.update({jointName: u[i]})  # TODO
            if jointName in self.Kp:  # PD gains defined for joint
                torque = self.pd_controller[jointName].control(u[i], 0.0)
                # torque = u[i]
                torque = np.clip(
                    torque, -self.u_max[jointName],
                    self.u_max[jointName])
                torques[jointName] = torque
        return torques

    def joint_reaction_force(self, side):
        """Return reaction force on given side.

        Parameters
        ----------
        side : str
            Either 'left' or 'right'
        """
        ankle_joint_state = p.getJointState(self.robot,
                                            self.joint_idx[side+'AnklePitch'])
        _, _, _, Mx, My, Mz = ankle_joint_state[2]
        total_torque = np.sqrt(np.sum(np.square([Mx, My, Mz])))
        return total_torque

    def set_zero_order_hold_nominal_pose(self):
        """Set robot into default standing position.

        Default position is given by `default_joint_config` attribute.
        """
        for jointName in self.controllable_joints:
            p.setJointMotorControl2(
                self.robot,
                self.joint_idx[jointName],
                p.POSITION_CONTROL,
                targetPosition=self.default_joint_config[jointName],
                force=self.u_max[jointName])

    def set_PD_position_control(self, torque_dict, u):
        """Apply torque to joints to move towards target positions given by u.

        Also updates `action` attribute.
        """
        final = np.isnan(u).any()
        # Set control
        for i in range(self.nu):
            jointName = self.controlled_joints[i]
            self.action.update({jointName: u[i]})  # TODO
            if not final:
                p.setJointMotorControl2(
                    self.robot, self.joint_idx[jointName],
                    targetPosition=u[i],
                    targetVelocity=0,
                    maxVelocity=self.v_max[jointName],
                    force=abs(torque_dict[jointName]),
                    controlMode=p.POSITION_CONTROL,
                    # positionGain=self.Kp[jointName],
                    # velocityGain=self.Kd[jointName],
                )

    def set_PD_velocity_control(self, torque_dict):
        """Set the maximum motor force limit for joints.

        When a step is taken, force is applied to reach target velocity,
        where target velocity = self.v_max * sign(max_torque).

        Parameters
        ----------
        torque_dict : dict of {joint_name: max_torque}
        """
        for joint, torque in torque_dict.items():
            p.setJointMotorControl2(
                self.robot,
                self.joint_idx[joint],
                # set desired velocity of joint
                targetVelocity=np.sign(torque)*self.v_max[joint],
                # set maximum motor force that can be used
                force=np.abs(torque),
                controlMode=p.VELOCITY_CONTROL)

    def set_PD_torque_control(self, torque_dict):
        """Set exact motor force that will be applied to joints when step taken.

        Parameters
        ----------
        torque_dict : dict of {joint_name: torque}
        """
        for joint, torque in torque_dict.items():
            p.setJointMotorControl2(
                self.robot,
                self.joint_idx[joint],
                # set desired velocity of joint
                targetVelocity=np.sign(torque)*self.v_max[joint],
                force=0,
                controlMode=p.VELOCITY_CONTROL)
            p.setJointMotorControl2(
                self.robot,
                self.joint_idx[joint],
                # set force/torque that is actually applied
                force=torque,
                controlMode=p.TORQUE_CONTROL)

    def set_control(self, u):
        """Set desired control for each joint, to be run when a step taken.

        During the `step` the physics engine will simulate the joint motors to
        reach the given target value,  within the maximum motor forces and other
        constraints.

        Parameters
        ----------
        u : list of length self.nu
            target position for controlled joint 0, 1, ..., nu
        """
        # final = np.isnan(u).any()
        # Set control
        for i in range(self.nu):
            jointName = self.controlled_joints[i]
            self.action.update({jointName: u[i]})  # TODO
            if self.use_bullet_default_pd:
                p.setJointMotorControl2(self.robot,
                                        self.joint_idx[jointName],
                                        targetPosition=u[i],
                                        targetVelocity=0,
                                        maxVelocity=self.v_max[jointName],
                                        force=self.u_max[jointName],
                                        positionGain=self.Kp[jointName],
                                        velocityGain=self.Kd[jointName],
                                        controlMode=p.POSITION_CONTROL)

            else:  # using own PD controller
                if jointName in self.Kp:  # PD gains defined
                    torque = self.pd_controller[jointName].control(u[i], 0)
                    torque = np.clip(torque,
                                     -self.u_max[jointName],
                                     self.u_max[jointName])
                    p.setJointMotorControl2(self.robot,
                                            self.joint_idx[jointName],
                                            targetVelocity=(
                                                np.sign(torque)
                                                * self.v_max[jointName]),
                                            force=np.abs(torque),
                                            # force=0,  # disable motor
                                            velocityGains=0.0001,
                                            controlMode=p.VELOCITY_CONTROL)
                else:  # PD gains not defined
                    p.setJointMotorControl2(self.robot,
                                            self.joint_idx[jointName],
                                            targetPosition=u[i],
                                            targetVelocity=0,
                                            maxVelocity=self.v_max[jointName],
                                            force=self.u_max[jointName],
                                            controlMode=p.POSITION_CONTROL)

        # Set zero order hold for uncontrolled joints
        for jointName in self.uncontrolled_joints:
            p.setJointMotorControl2(
                self.robot,
                self.joint_idx[jointName],
                targetPosition=self.default_joint_config[jointName],
                force=self.u_max[jointName],
                maxVelocity=self.v_max[jointName],
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
        p.applyExternalForce(self.robot,
                             -1,  # base
                             forceObj=force,
                             posObj=pos,
                             flags=p.LINK_FRAME)

    def draw_base_frame(self):
        """Draw approximate pelvis centre of mass."""
        # This may not be the Pelvis CoM position _exactly_ but should be fine,
        # otherwise can apply local transformation
        _, base_quat = p.getBasePositionAndOrientation(self.robot)
        base_pos = self.link_COM_position['pelvisBase']

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

        p.addUserDebugLine(np.array(base_pos),
                           np.array(base_pos) + x_axis_base[0],
                           [1, 0, 0],
                           5,
                           0.1)  # x axis
        p.addUserDebugLine(np.array(base_pos),
                           np.array(base_pos) + y_axis_base[0],
                           [0, 1, 0],
                           5,
                           0.1)  # y axis
        p.addUserDebugLine(np.array(base_pos),
                           np.array(base_pos) + z_axis_base[0],
                           [0, 0, 1],
                           5,
                           0.1)  # z axis

    def draw_base_yaw_frame(self):
        """Draw approximate yaw frame."""
        # This may not be the Pelvis CoM position _exactly_ but should be fine,
        # otherwise can apply local transformation
        _, base_quat = p.getBasePositionAndOrientation(self.robot)
        base_orn = p.getEulerFromQuaternion(base_quat)

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

        p.addUserDebugLine(np.array(origin),
                           np.array(origin) + x_axis_base[0],
                           [1, 0, 0],
                           5,
                           0.1)  # x axis
        p.addUserDebugLine(np.array(origin),
                           np.array(origin) + y_axis_base[0],
                           [0, 1, 0],
                           5,
                           0.1)  # y axis
        p.addUserDebugLine(np.array(origin),
                           np.array(origin) + z_axis_base[0],
                           [0, 0, 1],
                           5,
                           0.1)  # z axis

        return

    def draw_skeleton_yaw(self):
        # This may not be the Pelvis CoM position _exactly_ but should be fine,
        # otherwise can apply local transformation
        base_pos, base_quat = p.getBasePositionAndOrientation(self.robot)
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
            self.robot, self.joint_idx['torsoRoll'],
            computeLinkVelocity=1)
        chest_link_dis = np.array(chest_link_state[0]) - np.array(base_pos)
        chest_link_dis.resize(1, 3)
        # chest_link_dis_base = np.transpose(R_i @ chest_link_dis.transpose())
        chest_link_dis_yaw = np.transpose(Rz_i @ chest_link_dis.transpose())

        left_foot_link_state = p.getLinkState(
            self.robot, self.joint_idx['leftAnkleRoll'],
            computeLinkVelocity=0)
        left_foot_link_dis = np.array(left_foot_link_state[0]) - base_pos
        left_foot_link_dis.resize(1, 3)
        # left_foot_link_dis_base = np.transpose(
        #     R_i @left_foot_link_dis.transpose())
        left_foot_link_dis_yaw = np.transpose(
            Rz_i @left_foot_link_dis.transpose())
        # left_foot_link_dis_yaw = left_foot_link_dis

        right_foot_link_state = p.getLinkState(
            self.robot, self.joint_idx['rightAnkleRoll'],
            computeLinkVelocity=0)
        right_foot_link_dis = np.array(right_foot_link_state[0]) - base_pos
        right_foot_link_dis.resize(1, 3)
        # right_foot_link_dis_base = np.transpose(
        #    R_i @right_foot_link_dis.transpose())
        right_foot_link_dis_yaw = np.transpose(
            Rz_i @right_foot_link_dis.transpose())
        # right_foot_link_dis_yaw = right_foot_link_dis

        left_elbow_link_state = p.getLinkState(
            self.robot, self.joint_idx['leftElbowPitch'],
            computeLinkVelocity=0)
        left_elbow_link_dis = np.array(left_elbow_link_state[0]) - base_pos
        left_elbow_link_dis.resize(1, 3)
        # left_elbow_link_dis_base = np.transpose(
        #     R_i @left_elbow_link_dis.transpose())
        left_elbow_link_dis_yaw = np.transpose(
            Rz_i @left_elbow_link_dis.transpose())

        right_elbow_link_state = p.getLinkState(
            self.robot, self.joint_idx['rightElbowPitch'],
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

    def draw_skeleton_base(self):
        # This may not be the Pelvis CoM position _exactly_ but should be fine,
        # otherwise can apply local transformation
        base_pos, base_quat = p.getBasePositionAndOrientation(self.robot)

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
        chest_link_state = p.getLinkState(self.robot,
                                          self.joint_idx['torsoRoll'],
                                          computeLinkVelocity=1)
        chest_link_dis = np.array(chest_link_state[0]) - np.array(base_pos)
        chest_link_dis.resize(1, 3)
        chest_link_dis_base = np.transpose(R_i @ chest_link_dis.transpose())
        # chest_link_dis_yaw = np.transpose(Rz_i @ chest_link_dis.transpose())

        left_foot_link_state = p.getLinkState(self.robot,
                                              self.joint_idx['leftAnkleRoll'],
                                              computeLinkVelocity=0)
        left_foot_link_dis = np.array(left_foot_link_state[0]) - base_pos
        left_foot_link_dis.resize(1, 3)
        left_foot_link_dis_base = np.transpose(
            R_i @left_foot_link_dis.transpose())
        # left_foot_link_dis_yaw = np.transpose(
        #     Rz_i @left_foot_link_dis.transpose())
        # left_foot_link_dis_yaw = left_foot_link_dis

        right_foot_link_state = p.getLinkState(self.robot,
                                               self.joint_idx['rightAnkleRoll'],
                                               computeLinkVelocity=0)
        right_foot_link_dis = np.array(right_foot_link_state[0]) - base_pos
        right_foot_link_dis.resize(1, 3)
        right_foot_link_dis_base = np.transpose(
            R_i @right_foot_link_dis.transpose())
        # right_foot_link_dis_yaw = np.transpose(
        #     Rz_i @right_foot_link_dis.transpose())
        # right_foot_link_dis_yaw = right_foot_link_dis

        left_elbow_link_state = p.getLinkState(self.robot,
                                               self.joint_idx['leftElbowPitch'],
                                               computeLinkVelocity=0)
        left_elbow_link_dis = np.array(left_elbow_link_state[0]) - base_pos
        left_elbow_link_dis.resize(1, 3)
        left_elbow_link_dis_base = np.transpose(
            R_i @left_elbow_link_dis.transpose())
        # left_elbow_link_dis_yaw = np.transpose(
        #     Rz_i @left_elbow_link_dis.transpose())

        right_elbow_link_state = p.getLinkState(
            self.robot, self.joint_idx['rightElbowPitch'],
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
        return

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
        return

    def draw_COM(self):
        p.addUserDebugLine(self.COM_pos + np.array([0, 0, 2]),
                           self.COM_pos + np.array([0, 0, -2]),
                           [1, 0, 0],
                           5,
                           0.1)  # TODO rendering to draw COM

    def test_kinematic(self):
        """NOTE: DOES NOT WORK! and I don't understadn what it's meant to do."""
        base_pos, base_quat = p.getBasePositionAndOrientation(self.robot)
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
            jointNo = self.joint_idx[name]
            joint_name = name
            # for jointNo in range(p.getNumJoints(self.robot)):
            # info = p.getJointInfo(self.robot, jointNo)
            # joint_name = info[1].decode("utf-8")
            # self.joint_idx.update({joint_name: info[0]})
            # self.jointNameIdx.update({info[0]: joint_name})

            link_state = p.getLinkState(self.robot, jointNo,
                                        computeLinkVelocity=0)
            inertiaFramePos = link_state[2]
            inertiaFrameOrn = link_state[3]

            joint_info = p.getJointInfo(self.robot, jointNo)
            jointAxis = joint_info[13]
            parentFramePos = joint_info[14]
            parentFrameOrn = joint_info[15]
            parentIdx = joint_info[16]

            joint_state = p.getJointState(self.robot, jointNo)
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


class ValkyrieEnvBasic(ValkyrieEnv):

    def reward(self):
        """Return reward for current state and terms used in the reward.

        Returns
        -------
        reward : float
        reward_term : dict
            {name: value} of all elements summed to make the reward
        """
        base_pos, base_quat = p.getBasePositionAndOrientation(self.robot)
        base_orn = p.getEulerFromQuaternion(base_quat)
        chest_link_state = p.getLinkState(self.robot,
                                          self.joint_idx['torsoRoll'],
                                          computeLinkVelocity=1)
        torso_roll_err, torso_pitch_err, _, _ = chest_link_state[1]
        pelvis_roll_err, pelvis_pitch_err, _ = base_orn
        # Nominal COM x and z position [0.0, 1.104]
        # Nominal COM x and z position w.r.t COM of foot [0.0, 1.064] #TODO test

        # filtering for reward
        x_pos_err = 0.0 - self.COM_pos_local[0]
        y_pos_err = 0.0 - self.COM_pos_local[1]
        z_pos_err = 1.104 * 1.02 - self.COM_pos_local[2]

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
        right_foot_link_state = p.getLinkState(self.robot,
                                               self.joint_idx['rightAnkleRoll'],
                                               computeLinkVelocity=0)
        right_foot_orn = p.getEulerFromQuaternion(right_foot_link_state[1])
        right_foot_roll_err = right_foot_orn[2]
        left_foot_link_state = p.getLinkState(self.robot,
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
            joint_state = p.getJointState(self.robot, self.joint_idx[joint])
            position_follow_penalty -= abs(joint_state[0] - self.action[joint])

        # penalize reward when joint is moving too fast: (vel / max_vel)^2
        velocity_penalty = 0
        for joint in self.controlled_joints:  # TODO
            joint_state = p.getJointState(self.robot, self.joint_idx[joint])
            velocity_penalty -= (joint_state[1] / self.v_max[joint])**2

        # penalize reward when torque: torque / max_torque
        torque_penalty = 0
        for joint in self.controlled_joints:  # TODO
            joint_state = p.getJointState(self.robot, self.joint_idx[joint])
            torque_penalty -= abs(joint_state[3] / self.u_max[joint])

        # penalize power rate of joint motor: vel/max_vel * torque/max_torque
        power_penalty = 0
        for joint in self.controlled_joints:  # TODO
            joint_state = p.getJointState(self.robot, self.joint_idx[joint])
            power_penalty -= (abs(joint_state[3] / self.u_max[joint])
                              * abs(joint_state[1] / self.v_max[joint]))

        # penalize change in torque: torque_change / max_torque
        torque_change_penalty = 0
        for joint in self.controlled_joints:
            joint_state = p.getJointState(self.robot, self.joint_idx[joint])
            torque_change = self.hist_torque[joint] - joint_state[3]
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
