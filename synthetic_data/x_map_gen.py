#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 22:32:42 2019

@author: manish
"""

from flood_fill import getTiles

GRID_SIZE = 16
gridDimension = [GRID_SIZE, GRID_SIZE]
numPOI = 9
feature_map, occupancy_map = getTiles(gridDimension,numPOI)
print(occupancy_map)

# adapted from map generation procedure
start_point = [0, int(gridDimension[1]/2)]

latent_points = []
latent_points.append(start_point)

while(len(start_point)>0){
        latent_points[0]}