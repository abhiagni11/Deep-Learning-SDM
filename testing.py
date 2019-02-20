#!/usr/bin/env python3

""" Testing Module

This module tests all the submodule and calls the visualizer

Author: Abhijeet Agnihotri
"""

import random
import atexit
from visualize import Visualize
from underground import Underground
from robot import Robot

TUNNEL_FILE = './maps/tunnel_5.npy'
ARTIFACT_FILE = './maps/artifacts_5.npy'


def shutdown():
	print('\nGoodbye')

def main():
	atexit.register(shutdown)
	# Instantiate the environment
	tunnel = Underground(TUNNEL_FILE, ARTIFACT_FILE)
	x_dim, y_dim = tunnel._x_dim, tunnel._y_dim
	# Introduce a robot, only one for now
	wall_e = Robot(x_dim, y_dim)
	# To visualize
	graph = Visualize(TUNNEL_FILE)
	graph._initialise_visualization(tunnel._get_artifact_locations())

	for i in range(1000):
		state = wall_e._get_current_location()
		graph._keep_visualizing(state, tunnel._get_artifact_locations(), tunnel._get_observation(state), wall_e._get_explored_map(), tunnel._get_artifact_fidelity_map())
		wall_e._update_reward(tunnel._found_artifact(state))
		allowed_actions = tunnel._get_allowed_actions(state)

		# randomly give action
		action = random.choice(allowed_actions)

		# # user input action
		# direction = input("which action? input 1-up, 2-down, 3-right, 4-left")
		# if direction == '1':
		# 	action = 'up'
		# elif direction == '2':
		# 	action = 'down'
		# elif direction == '3':
		# 	action = 'right'
		# elif direction == '4':
		# 	action = 'left'

		# instead: query based on artifact_fidelity_matrix which will guide us which frontier to go to!! say if you have a planner class:
		#
		# fidelity_map = tunnel._get_artifact_fidelity_map()
		## in the future this fidelity map would be the output of the neural network.
		# explored_map = wall_e._get_explored_map()
		# current_observation = tunnel._get_observation(state)
		# action = frontier_get_action(fidelity_map, explored_map, current_observation, state)
		wall_e._give_action(action)
		# print(tunnel._get_artifact_fidelity_map())


if __name__ == "__main__":
	try:
		print('Started exploring\n')
		main()
	except (KeyboardInterrupt, SystemExit):
		raise



