#!/usr/bin/env python3

""" Robot Module

This module initializes the robot class

Author: Abhijeet Agnihotri
"""

import numpy as np


class Robot:

	def __init__(self, y_dim, x_dim):
		self._y_dim = y_dim
		self._x_dim = x_dim
		self._tunnel_grid = np.zeros((self._y_dim, self._x_dim))
		# Definition of entry point can be changed subject to map generation
		self._entry_point = [int(self._y_dim/2), 0]
		self._current_position = self._entry_point
		self._action_dict = {"none": 0, "up": 1, "right": 2, "down": 3, "left": 4}
		self._action_coords = [(0, 0), (0, -1), (1, 0), (0, 1), (-1, 0)]
		self._reward = 0

	def _get_current_location(self):
		return self._current_position

	def _give_action(self, action):
		new_state = (self._current_position[0] + self._action_coords[self._action_dict[action]][0], self._current_position[1] + self._action_coords[self._action_dict[action]][1])
		self._update_location(new_state)

	def _update_location(self, state):
		self._current_position = state

	def _update_reward(self, found_artifact):
		if found_artifact:
			self._reward += 100
		else:
			self._reward -=1