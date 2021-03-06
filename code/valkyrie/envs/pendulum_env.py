from gym import Env, spaces
from gym.utils import seeding
import numpy as np
import pybullet as p
import pybullet_data

from softqlearning.misc import Serializable


class InvertedPendulumEnv(Env, Serializable):
    """Inverted pendulum gym environment using pybullet"""

    metadata = {'render_modes': ['human']}

    def __init__(self):
        self.seed()
        self.deltas = [
            -1, -0.3, -0.1, -0.03, -0.01, 0, 0.01, 0.03, 0.1, 0.3, 1]
        self.action_space = spaces.Discrete(len(self.deltas))
        # pitch, gyro, commanded speed
        self.observation_space = spaces.Box(np.array([-np.pi, -np.pi, -5]),
                                            np.array([np.pi, np.pi, 5]))

        self.physics_client = p.connect(p.GUI)
        # make sure we can access URDF and MCJF files
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.reset()

    def reset(self):
        """Reset environment and return observation."""
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(0.01)  # seconds
        self.plane_id = p.loadURDF("plane.urdf")
        self.cart_id, self.pendulum_id = p.loadMJCF(
            "mjcf/inverted_pendulum.xml")
        self.steps_taken = 0
        self.target_velocity = 0
        return self._observe()

    def step(self, action):
        """Take an action.

        Returns
        -------
        observation : numpy.ndarray
        reward : float
        done : bool
        info : dict
        """
        self._assign_throttle(int(action))
        p.stepSimulation()
        self._observation = self._observe()
        reward = self._compute_reward()
        done = self._compute_done()
        info = {}
        self.steps_taken += 1
        return self._observation, reward, done, info

    def _assign_throttle(self, action):
        velocity_delta = self.deltas[action]
        self.target_velocity += velocity_delta
        # update angular velocity of wheels
        p.setJointMotorControl2(self.pendulum_id,
                                jointIndex=0,
                                controlMode=p.VELOCITY_CONTROL,
                                targetVelocity=self.target_velocity)
        p.setJointMotorControl2(self.pendulum_id,
                                jointIndex=1,
                                controlMode=p.VELOCITY_CONTROL,
                                targetVelocity=-self.target_velocity)

    def _observe(self):
        euler_x, _, _ = self._get_euler_angles()
        pitch_velocity, angular_velocity = p.getBaseVelocity(self.pendulum_id)
        return np.array([euler_x, angular_velocity[0], self.target_velocity])

    def _compute_reward(self):
        euler_x, _, _ = self._get_euler_angles()
        return (((1 - abs(euler_x)) / 10) - (abs(self.target_velocity / 100)))

    def _get_euler_angles(self):
        _, orientation = p.getBasePositionAndOrientation(self.pendulum_id)
        euler_angles = p.getEulerFromQuaternion(orientation)
        return euler_angles

    def _compute_done(self):
        position, _ = p.getBasePositionAndOrientation(self.pendulum_id)
        com_below_15cm = position[2] < 0.15
        at_step_limit = self.steps_taken >= 1500
        return com_below_15cm or at_step_limit

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human', close=False):
        """Does nothing because rendering is done by pybullet."""
        pass
