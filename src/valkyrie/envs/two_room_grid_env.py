#!/usr/bin/env python3

import copy
import matplotlib.pyplot as plt
import numpy as np

from gym import spaces
from gym.utils import seeding

EMPTY = BLACK = 0
WALL = GRAY = 1
TARGET = GREEN = 3
AGENT = RED = 4
SUCCESS = PINK = 6
COLORS = {
    BLACK: [0.0, 0.0, 0.0],
    GRAY: [0.5, 0.5, 0.5],
    GREEN: [0.0, 1.0, 0.0],
    RED: [1.0, 0.0, 0.0],
    PINK: [1.0, 0.0, 1.0]
}

NOOP = 0
DOWN = 1
UP = 2
LEFT = 3
RIGHT = 4


class TwoRoomGridEnv():
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        self.seed()
        self.actions = [NOOP, DOWN, UP, LEFT, RIGHT]
        self.inv_actions = [0, 2, 1, 4, 3]
        self.action_space = spaces.Discrete(len(self.actions))
        self.transition = {
            NOOP: [0, 0],
            DOWN: [1, 0],
            UP: [-1, 0],
            LEFT: [0, -1],
            RIGHT: [0, 1]}
        self.img_shape = [256, 256, 3]  # visualize observation

        # initialise env with random initial and target locations
        self.initial_grid = np.array([
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 0],
        ], dtype=int)
        self.grid_shape = self.initial_grid.shape
        room_row_coords = np.arange(self.grid_shape[0])
        room_col_coords = np.array([0, 1, -2, -1])
        self.target_location = (self.np_random.choice(room_row_coords),
                                self.np_random.choice(room_col_coords))
        self.initial_location = (self.np_random.choice(room_row_coords),
                                 self.np_random.choice(room_col_coords))
        self.initial_grid[self.target_location] = TARGET
        self.initial_grid[self.initial_location] = AGENT
        self.grid = copy.deepcopy(self.initial_grid)
        self.location = copy.deepcopy(self.initial_location)
        self.observation_space = spaces.Box(  # observations get normalised
            low=np.array([-1.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32)

        self.episode_total_reward = 0.0
        self.restart_once_done = False
        self.viewer = None

    def seed(self, seed=None):
        """Fix seed for reproducibility."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def normalise_observation(self, location, action, reward):
        """Normalise current location, previous action but not reward.

        Observation is [current location, last action, last reward].
        Normalisation used to help performance of neural network.
        """
        location = 2.0 * ((self.grid_shape[0] * location[0] + location[1])
                          / self.grid.size) - 1.
        action_normalised = ((action - np.mean(self.actions))
                             / np.max(self.actions))
        return np.array([location, action_normalised, reward])

    def step(self, action):
        """Return next observation, reward, done, and info on success."""
        action = int(action)
        info = {'success': False}
        done = False

        penalty_step = -0.1
        penalty_wall = -0.5
        reward = penalty_step
        self.episode_total_reward += reward

        # no movement requested
        if action == NOOP:
            info['success'] = True
            obs = self.normalise_observation(self.location, action, reward)
            return obs, reward, done, info

        # try to move
        new_location = (
            self.location[0] + self.transition[action][0],
            self.location[1] + self.transition[action][1])

        # movement fails
        is_out_of_grid = (
            new_location[0] < 0 or
            new_location[1] < 0 or
            new_location[0] >= self.grid_shape[0] or
            new_location[1] >= self.grid_shape[1])
        if is_out_of_grid:
            obs = self.normalise_observation(self.location, action, reward)
            return obs, reward, done, info

        new_location_type = self.grid[new_location]
        if new_location_type == WALL:
            self.episode_total_reward += penalty_wall
            obs = self.normalise_observation(self.location, action, reward)
            return obs, reward, done, info

        # movement succeeds
        elif new_location_type == EMPTY:
            self.grid[new_location] = AGENT
        elif new_location_type == TARGET:
            self.grid[new_location] = SUCCESS
        self.grid[self.location] = EMPTY
        self.location = new_location
        info['success'] = True

        if self.location == self.target_location:
            done = True
            reward += 1.0
            if self.restart_once_done:
                self.reset()

        obs = self.normalise_observation(self.location, action, reward)
        return obs, reward, done, info

    def reset(self):
        """Reset the environment and reward, and return an observation."""
        self.location = copy.deepcopy(self.initial_location)
        self.grid = copy.deepcopy(self.initial_grid)
        action = 0
        reward = 0.0
        self.episode_total_reward = reward
        return self.normalise_observation(self.location, action, reward)

    def close(self):
        if self.viewer:
            self.viewer.close()

    def _grid_to_image(self, img_shape=None):
        """Return image from the gridmap."""
        if img_shape is None:
            img_shape = self.img_shape
        observation = np.zeros(img_shape)
        gs0 = int(observation.shape[0] / self.grid.shape[0])
        gs1 = int(observation.shape[1] / self.grid.shape[1])
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                for k in range(3):
                    this_value = COLORS[self.grid[i, j]][k]
                    observation[i * gs0: (i+1) * gs0,
                                j * gs1: (j+1) * gs1,
                                k] = this_value
        return (255*observation).astype(np.uint8)

    def render(self, mode='human', close=False):
        """Vvisualise the environment according to specification."""
        if close:
            plt.close(1)  # Final plot
            return

        img = self._grid_to_image()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            plt.figure()
            plt.imshow(img)
            return
