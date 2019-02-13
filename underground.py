#!/usr/bin/env python3

""" Underground (environment) Module

This module initialises the main environment class

Author: Abhijeet Agnihotri
"""

import numpy as np


class Underground:

	def __init__(self, tunnel_filename, artifact_filename):
		self._tunnel_map = np.load(tunnel_filename)
		self._artifact_locations = [(x[0], x[1]) for x in np.load(artifact_filename).tolist()]
		self._x_dim, self._y_dim = self._tunnel_map.shape
		self._action_dict = {"none": 0, "up": 1, "right": 2, "down": 3, "left": 4}
		self._action_coords = [(0, 0), (0, -1), (1, 0), (0, 1), (-1, 0)]

	def _check_state_in_tunnel(self, state):
		return self._tunnel_map[state[1]][state[0]]

	def _get_allowed_actions(self, state):
		allowed_actions = []
		for key, value in self._action_dict.items():
			new_state = [state[0] + self._action_coords[value][0], state[1] + self._action_coords[value][1]]
			if self._check_state_in_tunnel(new_state):
				allowed_actions.append(key)
				# print('action: {} and new_state: {} and key: {}'.format(self._check_state_in_tunnel(new_state), new_state, key))
		# print('allowed action: {} and state {}'.format(allowed_actions, state))
		return allowed_actions

	def _found_artifact(self, state):
		if self._artifact_locations.count(state):
			return True
		else:
			return False
