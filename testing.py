#!/usr/bin/env python3

""" Robot Module

This module initializes the robot class

Author: Abhijeet Agnihotri
"""

import numpy as np
import random

from visualize import Visualize
from underground import Underground
from robot import Robot

TUNNEL_FILE = './maps/tunnel.npy'
ARTIFACT_FILE = './maps/artifacts.npy'


if __name__ == "__main__":
	# Instantiate the environment
	tunnel = Underground(TUNNEL_FILE, ARTIFACT_FILE)
	x_dim, y_dim = tunnel._x_dim, tunnel._y_dim
	# Introduce a robot, only one for now
	wall_e = Robot(x_dim, y_dim)
	# To visualize
	graph = Visualize(TUNNEL_FILE, ARTIFACT_FILE)
	graph._initialise_visualization()

	for i in range(20):
		state = wall_e._get_current_location()
		graph._keep_visualizing(state)
		wall_e._update_reward(tunnel._found_artifact(state))
		allowed_actions = tunnel._get_allowed_actions(state)
		action = random.choice(allowed_actions)
		wall_e._give_action(action)




