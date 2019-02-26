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
from frontier2 import Frontier

TUNNEL_FILE = './maps/tunnel_5.npy'
ARTIFACT_FILE = './maps/artifacts_5.npy'


def shutdown():
	print('\nGoodbye')


def tick(tunnel, wall_e, action):
	wall_e._give_action(action)
	state = wall_e._get_current_location()
	wall_e._update_reward(tunnel._found_artifact(state))
	return state

def main():
	atexit.register(shutdown)

	# Instantiate the environment
	tunnel = Underground(TUNNEL_FILE, ARTIFACT_FILE)
	x_dim, y_dim = tunnel._x_dim, tunnel._y_dim

	steps = 0
	budget = 50

	# Introduce a robot, only one for now
	wall_e = Robot(x_dim, y_dim)

	# Ininialize frontier
	frontier = Frontier()

	# To visualize
	graph = Visualize(TUNNEL_FILE)
	graph._initialise_visualization(tunnel._get_artifact_locations())

	# Set current state
	state = wall_e._get_current_location()


	while steps < budget:
		print("####################")
		print("Loop counter:", steps)
		# Get matrix of observed frontier values around wall-e and update observed map
		observation = tunnel._get_observation(state)
		wall_e.update_observed_map(observation, tunnel._observation_radius)

		# Update visualization
		graph._keep_visualizing(state, tunnel._get_artifact_locations(), observation, wall_e._get_explored_map(), tunnel._get_artifact_fidelity_map())

		# Pick the next frontier and get a path to that point
		path = frontier.get_next_frontier(state, wall_e._observed_map, wall_e._frontiers)
		# Loop through the path and update the robot at each step
		for point in path:
			# print("Next point", point)
			distance = abs(wall_e._get_current_location()[0] - point[0]) + abs(wall_e._get_current_location()[1] - point[1])

			# While loop continues to move robot until point has been reached
			while distance > 0:
				# print("Robot Position", wall_e._get_current_location())
				# Find allowed actions
				allowed_actions = tunnel._get_allowed_actions(state)
				# Get the action that will take the robot to the next point
				action = wall_e._next_action(point, allowed_actions)
				# print("Action", action)
				# Move robot and update world
				state = tick(tunnel, wall_e, action)
				steps += 1
				print("Reward", wall_e._reward)
				graph._keep_visualizing(state, tunnel._get_artifact_locations(), observation,
										wall_e._get_explored_map(), tunnel._get_artifact_fidelity_map())

				# Update the distance to the next point
				distance = abs(wall_e._get_current_location()[0] - point[0]) + abs(wall_e._get_current_location()[1] - point[1])



if __name__ == "__main__":
	try:
		print('Started exploring\n')
		main()
	except (KeyboardInterrupt, SystemExit):
		raise


	# randomly give action
	# action = frontier.get_action(state, allowed_actions)

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
	# print(tunnel._get_artifact_fidelity_map())


