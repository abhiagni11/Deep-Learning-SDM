#!/usr/bin/env python3

""" Underground (environment) Module

This module initialises the main environment class

Author: Abhijeet Agnihotri
"""

import torch
import numpy as np
from neural_network.artifact_prediction_cuda import Net


class Underground:

	def __init__(self, grid_size, tunnel_filename, artifact_filename):

		'''
		FOR MANISH
		Don't get confused with self._tunnel_map and self._artifact_locations
		Both of these are initialized by loading the numpy files from your flood fill. 
		Pay attention to the sketch I sent to you in WhatsApp, I had to change x to y and y to x...

		Everything else (states and points) are denoted by [x, y] and are consistent to the sketch I sent you in WhatsApp. 
		Take a look and let me know if that helps, I can talk more and clarify your doubts.

		'''
		# tunnel map is (y, x)
		self._tunnel_map = np.load(tunnel_filename)
		self._observation_radius = 1
		self._grid_size = grid_size
		# artifact locations = (x, y)
		self._predict_artifact(self._tunnel_map)
		self._artifact_locations = [(x[1], x[0]) for x in np.load(artifact_filename).tolist()]
		self._updated_artifact_locations = self._artifact_locations[:]
		self._y_dim, self._x_dim = self._tunnel_map.shape
		# self._action_dict = {"none": 0, "up": 1, "right": 2, "down": 3, "left": 4}
		# self._action_coords = [(0, 0), (0, -1), (1, 0), (0, 1), (-1, 0)]
		self._action_dict = {"up": 0, "right": 1, "down": 2, "left": 3}  # Actions without a "none" option
		self._action_coords = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # Actions without a "none" option
		self._artifact_fidelity_map = np.zeros_like(self._tunnel_map)
		self._update_artifact_fidelity_map()
		self._update_predicted_artifact_fidelity_map()
		# This helps debug. Only need to run once per tunnel map
		# np.savetxt("artifact_fidelity.csv", self._artifact_fidelity_map, delimiter=',')

	def _predict_artifact(self, image):
		image = torch.Tensor(image)
		image = image.reshape(1, 1, image.shape[0],image.shape[1])
		model = Net()
		model.load_state_dict(torch.load('neural_network/mytraining_{}.pth'.format(self._grid_size)))
		model.eval()
		outputs = model(image)
		a=2
		outputs = outputs.reshape(self._grid_size, self._grid_size).detach().numpy()
		outputs = 1.0/(1.0 + np.exp(-outputs))
		self._predicted_artifact_locations = []
		self._predicted_artifact_fidelity_map = np.zeros_like(self._tunnel_map)
		for x in range(len(outputs)):
			for y in range(len(outputs[0])):
				if outputs[x][y] > 0.55:
					self._predicted_artifact_locations.append((y,x))
					self._predicted_artifact_fidelity_map[y][x] += outputs[x][y]
		self._updated_predicted_artifact_locations = self._predicted_artifact_locations[:]

	def _get_predicted_artifact_locations(self):
		return self._predicted_artifact_locations

	def _get_artifact_locations(self):
		return self._updated_artifact_locations

	def _get_artifact_fidelity_map(self):
		return self._artifact_fidelity_map

	def _get_predicted_artifact_fidelity_map(self):
		return self._predicted_artifact_fidelity_map

	def _update_artifact_fidelity_map(self):
		self._artifact_fidelity_map = np.zeros_like(self._tunnel_map)

		for artifact in self._updated_artifact_locations:
			self._add_artifact_fidelity(artifact[0], artifact[1])
		self._artifact_fidelity_map = np.multiply(self._artifact_fidelity_map, self._tunnel_map)

	def _update_predicted_artifact_fidelity_map(self):
		self._predicted_artifact_fidelity_map = np.zeros_like(self._tunnel_map)

		for artifact in self._updated_predicted_artifact_locations:
			self._add_predicted_artifact_fidelity(artifact[0], artifact[1])
		self._predicted_artifact_fidelity_map = np.multiply(self._predicted_artifact_fidelity_map, self._tunnel_map)

	def _add_predicted_artifact_fidelity(self, artifact_x, artifact_y):
		for y in range(self._y_dim):
			for x in range(self._x_dim):
				# self._artifact_fidelity_map[y][x] += (self._x_dim + self._y_dim) - (np.sqrt((y - artifact_y)**2 + (x - artifact_x)**2) + 1)
				self._predicted_artifact_fidelity_map[y][x] += 5.0/(np.sqrt((y - artifact_y)**2 + (x - artifact_x)**2) + 1)

	# def _add_artifact_fidelity_2(self, artifact_x, artifact_y):
	# 	for y in range(self._y_dim):
	# 		for x in range(self._x_dim):
	# 			self._artifact_fidelity_map[y][x] += 5.0/(np.sqrt((y - artifact_y)**2 + (x - artifact_x)**2) + 1)
	# 			if x == artifact_x and y == artifact_y:
	# 				self._artifact_fidelity_map[y][x] += 5
	# 			elif np.sqrt((y - artifact_y)**2 + (x - artifact_x)**2) <= 1:
	# 				# print("Square Root Term", np.sqrt((y - artifact_y)**2 + (x - artifact_x)**2))
	# 				# print("artifact x y", artifact_x, artifact_y)
	# 				# print('x y', x, y)
	# 				self._artifact_fidelity_map += 0.5

	def _add_artifact_fidelity(self, artifact_x, artifact_y):
		for y in range(self._y_dim):
			for x in range(self._x_dim):
				# self._artifact_fidelity_map[y][x] += (self._x_dim + self._y_dim) - (np.sqrt((y - artifact_y)**2 + (x - artifact_x)**2) + 1)
				self._artifact_fidelity_map[y][x] += 5.0/(np.sqrt((y - artifact_y)**2 + (x - artifact_x)**2) + 1)

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
		if self._updated_predicted_artifact_locations.count(state):
			self._updated_predicted_artifact_locations.remove(state)
			self._update_predicted_artifact_fidelity_map()
		if self._updated_artifact_locations.count(state):
			self._updated_artifact_locations.remove(state)
			self._update_artifact_fidelity_map()
			# print('Artifact found at: {}'.format(state))
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

