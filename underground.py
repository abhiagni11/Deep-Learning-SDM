#!/usr/bin/env python3

""" Underground (environment) Module

This module initialises the main environment class

Author: Abhijeet Agnihotri
"""

import numpy as np


class Underground:

	def __init__(self, tunnel_filename, artifact_filename):
		# tunnel map is (y, x)
		self._tunnel_map = np.load(tunnel_filename)
		self._observation_radius = 1
		# artifact locations = (x, y)
		self._artifact_locations = [(x[1], x[0]) for x in np.load(artifact_filename).tolist()]
		self._updated_artifact_locations = self._artifact_locations[:]
		self._y_dim, self._x_dim = self._tunnel_map.shape
		# self._action_dict = {"none": 0, "up": 1, "right": 2, "down": 3, "left": 4}
		# self._action_coords = [(0, 0), (0, -1), (1, 0), (0, 1), (-1, 0)]
		self._action_dict = {"up": 0, "right": 1, "down": 2, "left": 3}  # Actions without a "none" option
		self._action_coords = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # Actions without a "none" option
		self._artifact_fidelity_map = np.zeros_like(self._tunnel_map)
		self._update_artifact_fidelity_map()
		# This helps debug. Only need to run once per tunnel map
		# np.savetxt("artifact_fidelity.csv", self._artifact_fidelity_map, delimiter=',')

	def _update_artifact_fidelity_map(self):
		self._artifact_fidelity_map = np.zeros_like(self._tunnel_map)
		for artifact in self._updated_artifact_locations:
			self._add_artifact_fidelity(artifact[0], artifact[1])
		self._artifact_fidelity_map = np.multiply(self._artifact_fidelity_map, self._tunnel_map)

	def _add_artifact_fidelity(self, artifact_x, artifact_y):
		for y in range(self._y_dim):
			for x in range(self._x_dim):
				self._artifact_fidelity_map[y][x] += 1.0/(np.sqrt((y - artifact_y)**2 + (x - artifact_x)**2) + 1)

	def _check_state_in_tunnel(self, state):
		# state = (x, y)
		if state[0] < 0 or state[1] < 0 or state[0] >= self._x_dim or state[1] >= self._y_dim:
			return 0
		else:
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
		if self._updated_artifact_locations.count(state):
			self._updated_artifact_locations.remove(state)
			self._update_artifact_fidelity_map()
			print('Artifact found at: {}'.format(state))
			return True
		else:
			return False

	def _get_observation(self, state):
		# state and current observation are (x, y)
		self._current_observation = np.zeros((self._observation_radius*2 + 1, self._observation_radius*2 + 1))
		for y in range(self._observation_radius*2 + 1):
			for x in range(self._observation_radius*2 + 1):
				_i_state = (state[0] + x - self._observation_radius, state[1] + y - self._observation_radius)
				if self._check_state_in_tunnel(_i_state):
					# Return the fidelity value at the observed points
					self._current_observation[x][y] = self._artifact_fidelity_map[_i_state[1]][_i_state[0]]
				else:
					self._current_observation[x][y] = 0
		return self._current_observation

	def _get_artifact_fidelity_map(self):
		return self._artifact_fidelity_map

	def _get_artifact_locations(self):
		return self._updated_artifact_locations
