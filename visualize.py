#!/usr/bin/env python3

""" Visualize Module

This module contains methods to visualize the underground tunnel system

Author: Abhijeet Agnihotri
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches


class Visualize:

	def __init__(self, tunnel_filename):
		self._tunnel_map = np.load(tunnel_filename)
		self._y_dim, self._x_dim = self._tunnel_map.shape
		self.fig, self.ax = plt.subplots()

	def _initialise_visualization(self, artifact_locations):
		self._artifact_locations = artifact_locations
		plt.tight_layout()
		plt.imshow(self._tunnel_map, cmap=plt.get_cmap('bone'))
		plt.ion()
		plt.show()

	def _check_state_in_tunnel(self, state):
		# state = (x, y)
		if state[0] < 0 or state[1] < 0 or state[0] >= self._x_dim or state[1] >= self._y_dim:
			return 0
		else:
			return self._tunnel_map[state[1]][state[0]]

	def _keep_visualizing(self, robot_states, updated_artifact_locations, current_observation):
		# TODO: Adapt for multiple robots
		self._artifact_locations = updated_artifact_locations
		plt.cla()
		plt.imshow(self._tunnel_map, cmap=plt.get_cmap('bone'))
		observation_radius = len(current_observation[0])//2

		for y in range(observation_radius*2 + 1):
			for x in range(observation_radius*2 + 1):
				_i_state = (robot_states[0] + x - observation_radius, robot_states[1] + y - observation_radius)
				if self._check_state_in_tunnel(_i_state):
					rect = patches.Rectangle((_i_state[0] - 0.5, _i_state[1] - 0.5), 1, 1, linewidth=1, edgecolor='g', facecolor='g')
					self.ax.add_patch(rect)

		rect = patches.Rectangle((robot_states[0] - 0.5, robot_states[1] - 0.5), 1, 1, linewidth=1, edgecolor='r', facecolor='r')
		self.ax.add_patch(rect)

		for artifact in self._artifact_locations:
			rect = patches.Rectangle((artifact[0] - 0.5, artifact[1] - 0.5), 1, 1, linewidth=1, edgecolor='b', facecolor='none')
			self.ax.add_patch(rect)
		self.ax.plot()
		# plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
		plt.draw()
		plt.pause(.01)