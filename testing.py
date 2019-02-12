#!/usr/bin/env python3

""" Testing Module

This module tests all the submodule and calls the visualizer

Author: Abhijeet Agnihotri
"""

import random

from visualize import Visualize
from underground import Underground
from robot import Robot

TUNNEL_FILE = './maps/tunnel.npy'
ARTIFACT_FILE = './maps/artifacts.npy'


if __name__ == "__main__":
	# Instantiate the environment
	tunnel = Underground(TUNNEL_FILE, ARTIFACT_FILE)
	y_dim, x_dim = tunnel._y_dim, tunnel._x_dim
	# Introduce a robot, only one for now
	wall_e = Robot(y_dim, x_dim)
	# To visualize
	graph = Visualize(TUNNEL_FILE, ARTIFACT_FILE)
	graph._initialise_visualization()

	for i in range(50):
		state = wall_e._get_current_location()
		graph._keep_visualizing(state)
		wall_e._update_reward(tunnel._found_artifact(state))
		allowed_actions = tunnel._get_allowed_actions(state)
		action = random.choice(allowed_actions)
		wall_e._give_action(action)




