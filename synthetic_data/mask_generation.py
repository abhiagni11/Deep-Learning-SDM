#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:22:18 2019

@author: Manish Saroya
Contact: saroyam@oregonstate.edu
"""
from x_map_gen import Exploration
import numpy as np 
import matplotlib.pyplot as plt

class Mask:
	"""Creates a mask over a explored map """
	def __init__(self, exploredMap, frontierVector):
		self.exploredMap = exploredMap
		self.frontierVector = frontierVector 
		
	def free_nbrs(self, point):
		dirs_motion = [
		    lambda x, y: (x-1, y),  # up
		    lambda x, y: (x+1, y),  # down
		    lambda x, y: (x, y - 1),  # left
		    lambda x, y: (x, y + 1),  # right
		]
		nbrsVector = []
		for d in dirs_motion:
			nx, ny = d(point[0], point[1])
			if 0 <= nx < len(self.exploredMap) and 0 <= ny < len(self.exploredMap[0]):
				if self.exploredMap[nx, ny]==0:
					nbrsVector.append([nx, ny])
		return nbrsVector
	
	def get_mask(self):
		mask = np.ones((len(self.exploredMap),len(self.exploredMap[0])))
		for x in range(len(self.exploredMap)):
			for y in range(len(self.exploredMap[0])):
				if self.exploredMap[x,y]==1:
					mask[x,y] = 0
					nbrsVector = self.free_nbrs([x,y])
					for p in nbrsVector:
						mask[p[0],p[1]] = 0
		
		for p in self.frontierVector:
			mask[p[0],p[1]] = 1
		
		return mask
	
######### Exploration Parameters #############	
GRID_SIZE = 32
numPOI = 20
filterRatio = 0.7
##############################################

explore = Exploration(GRID_SIZE, numPOI, filterRatio)
explore.generate_map()
map, frontierVector = explore.flood_fill_filter()
for p in frontierVector:
	map[p[0],p[1]] = 0.5
print map
plt.imshow(map)
plt.pause(0.0001)
maskobject = Mask(map,frontierVector)
mask = maskobject.get_mask()
plt.imshow(mask)