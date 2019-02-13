#!/usr/bin/env python3

""" Visualize Module

This module contains methods to visualize the underground tunnel system

Author: Abhijeet Agnihotri
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches


class Visualize:

	def __init__(self, tunnel_filename, artifact_filename):
		self._tunnel_map = np.load(tunnel_filename)
		self._artifact_locations = [(x[1], x[0]) for x in np.load(artifact_filename).tolist()]
		self._x_dim, self._y_dim = self._tunnel_map.shape
		self.fig, self.ax = plt.subplots()

	def _initialise_visualization(self):
		plt.tight_layout()
		plt.imshow(self._tunnel_map, cmap=plt.get_cmap('bone'))
		plt.ion()
		plt.show()

	def _keep_visualizing(self, robot_states):
		# TODO: Adapt for multiple robots
		plt.cla()
		plt.imshow(self._tunnel_map, cmap=plt.get_cmap('bone'))
		rect = patches.Rectangle((robot_states[0] - 0.5, robot_states[1] - 0.5), 1, 1, linewidth=1, edgecolor='r', facecolor='r')
		self.ax.add_patch(rect)
		for artifact in self._artifact_locations:
			rect = patches.Rectangle((artifact[0] - 0.5, artifact[1] - 0.5), 1, 1, linewidth=1, edgecolor='b', facecolor='none')
			self.ax.add_patch(rect)
		self.ax.plot()
		# plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
		plt.draw()
		plt.pause(.1)
