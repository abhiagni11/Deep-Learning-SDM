#!/usr/bin/env python3

"""
Frontier Algorithm to choose which direction to explore next

Author: Anna Nickelson
"""

import numpy as np
from map_generation import map_generation
from math import sqrt


class Frontier:

    def get_next_frontier(self, current, observed_map, frontiers):
        frontier_values = []
        paths = []

        # Find coordinates of each frontier
        frontier_indices = np.array(np.nonzero(frontiers))

        # Creates a list of current values of the frontiers.
        # This gets updated each time because frontier values change
        for i in range(frontier_indices.shape[1]):
            # Find shortest path to the next frontier and the length of that path
            path = np.array(map_generation.getPath(observed_map, current, frontier_indices[:, i]))
            path = np.flip(path, 0)
            path_length = np.shape(path)[0]

            # Append the frontier value at that point, scaled based on the cost to get there.
            # Current point is included in path, so length - 1
            frontier_values.append(frontiers[frontier_indices[0, i]][frontier_indices[1, i]] / sqrt(path_length-1))

            # Keep track of all paths to all frontiers
            paths.append(path)

        # Finds the largest value in the frontier values and returns the index
        choice_index = frontier_values.index(np.amax(frontier_values))

        print("Robot Position:", current)
        print("Chosen path", paths[choice_index])

        return paths[choice_index]
