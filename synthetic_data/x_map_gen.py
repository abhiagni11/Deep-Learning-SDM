#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 22:32:42 2019

@author: manish
"""

from flood_fill import getTiles

class Exploration:
	"""Filters X% exploration map form completely known map 
	   with appropriate flood fill based masking"""
	      
	def __init__(self, gridSize, numPOI, percentFilter):
		self.gridSize = gridSize
		self.gridDimension = [self.gridSize, self.gridSize]
		self.numPOI = numPOI
		self.percentFilter = percentFilter
		self.occupancy_map = None
	
	def generate_map(self):
		feature_map, self.occupancy_map = getTiles(self.gridDimension, self.numPOI)
		print(self.occupancy_map)
	
	def nbrs(self, point):
		pass
	
	def flood_fill_filter(self):
		# adapted from map generation procedure
		startp = [0, int(self.gridDimension[1]/2)]
		latentPoints = []
		latentPoints.append(startp)
		while(len(latentPoints)>0):
			latent = latentPoints.pop(0)
			
		
GRID_SIZE = 16
numPOI = 9

explore = Exploration(GRID_SIZE, numPOI,0)
explore.generate_map()
