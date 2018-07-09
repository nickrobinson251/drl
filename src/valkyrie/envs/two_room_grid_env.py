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
    num_env = 0

    def __init__(self):
        self.seed()
        self.actions = [NOOP, UP, DOWN, LEFT, RIGHT]
        self.inv_actions = [0, 2, 1, 4, 3]
        self.action_space = spaces.Discrete(len(self.actions))
        self.action_pos_dict = {
            NOOP: [0, 0],
            UP: [-1, 0],
            DOWN: [1, 0],
            LEFT: [0, -1],
            RIGHT: [0, 1]}
        self.img_shape = [256, 256, 3]  # visualize observation

        # initialize system observation
        self.plan = 0
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
        self.target_location = (self.np_random.choice(range(self.grid_shape[0])),
                                self.np_random.choice([0, 1, -2, -1]))
        self.start_location = (self.np_random.choice(range(self.grid_shape[0])),
                                self.np_random.choice([0, 1, -2, -1]))
        self.initial_grid[self.target_location] = TARGET
        self.initial_grid[self.start_location] = AGENT
        self.current_grid = copy.deepcopy(self.initial_grid)
        self.observation_space = spaces.Box(low=np.array([-1.0, -1.0, -1.0]),
                                            high=np.array([1.0, 1.0, 1.0]),
                                            dtype=np.float32)

        # observation: start, target, current location
        self.location = copy.deepcopy(self.start_location)
        self.episode_total_reward = 0.0
        self.restart_once_done = False  # restart or not once done
        self.viewer = None

    def seed(self, seed=None):
        """Fix seed for reproducibility."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_observation(self, coordinates, action, reward):
        """Return array of current location in the grid given coordinates,
        the previous action normalised and the previous reward."""
        # Normalized for better perform of the NN
        location = 2.0 * ((self.grid_shape[0] * coordinates[0] + coordinates[1])
                          / (self.grid_shape[0] * self.grid_shape[1])) - 1.
        action_normalised = ((action - np.mean(self.actions))
                             / np.max(self.actions))
        return np.array([location, action_normalised, reward])

    def step(self, action):
        """Return next observation, reward, done, and info on success."""
        action = int(action)
        info = {'success': False}
        done = False

        penalty_step = 0.1
        penalty_wall = 0.5
        reward = -penalty_step

        if action == NOOP:
            info['success'] = True
            self.episode_total_reward += reward
            observation = self.get_observation(self.location, action, reward)
            return observation, reward, done, info

        # Make a step
        next_position = (
            self.location[0] + self.action_pos_dict[action][0],
            self.location[1] + self.action_pos_dict[action][1])

        is_out_of_map = (
            next_position[0] < 0 or
            next_position[1] < 0 or
            next_position[0] >= self.grid_shape[0] or
            next_position[1] >= self.grid_shape[1])
        if is_out_of_map:
            info['success'] = False
            self.episode_total_reward += reward
            observation = self.get_observation(self.location, action, reward)
            return observation, reward, done, info

        # successful behavior
        target_position = self.current_grid[next_position]

        if target_position == EMPTY:
            self.current_grid[next_position] = AGENT

        elif target_position == WALL:
            info['success'] = False
            self.episode_total_reward += (reward - penalty_wall)
            observation = self.get_observation(self.location, action, reward)
            return observation, reward, done, info

        elif target_position == TARGET:
            self.current_grid[next_position] = SUCCESS

        self.current_grid[self.location] = EMPTY
        self.location = copy.deepcopy(next_position)
        info['success'] = True

        if next_position == self.target_location:
            done = True
            reward += 1.0
            if self.restart_once_done:
                self.reset()

        self.episode_total_reward += reward
        observation = self.get_observation(self.location, action, reward)
        return observation, reward, done, info

    def reset(self):
        """Return the initial observation of the environment."""
        self.location = copy.deepcopy(self.start_location)
        self.current_grid = copy.deepcopy(self.initial_grid)
        self.episode_total_reward = 0.0
        return self.get_observation(self.location, 0.0, 0.0)

    def close(self):
        if self.viewer:
            self.viewer.close()

    def _gridmap_to_image(self, img_shape=None):
        """Return image from the gridmap."""
        if img_shape is None:
            img_shape = self.img_shape
        observation = np.zeros(img_shape)
        gs0 = int(observation.shape[0] / self.current_grid.shape[0])
        gs1 = int(observation.shape[1] / self.current_grid.shape[1])
        for i in range(self.current_grid.shape[0]):
            for j in range(self.current_grid.shape[1]):
                for k in range(3):
                    this_value = COLORS[self.current_grid[i, j]][k]
                    observation[i * gs0: (i+1) * gs0,
                                j * gs1: (j+1) * gs1,
                                k] = this_value
        return (255*observation).astype(np.uint8)

    def render(self, mode='human', close=False):
        """Vvisualise the environment according to specification."""
        if close:
            plt.close(1)  # Final plot
            return

        img = self._gridmap_to_image()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            plt.figure()
            plt.imshow(img)
            return
