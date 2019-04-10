#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 00:06:14 2019

@author: manish

This module generates database of maps. The database is split into training and
test sets and stored into a pickle file.
"""

from flood_fill import getTiles
import pickle 
import numpy as np

######## Parameters for generating the database #############
GRID_SIZE = 16
numPOI = 15
trainRatio = 0.9
totalData = 10000
#############################################################

data = {}
data["training_data"] = []
data["training_labels"] = []
data["testing_data"] = []
data["testing_labels"] = []

gridDimension = [GRID_SIZE, GRID_SIZE]

for i in range(int(trainRatio * totalData)):
    m, n = getTiles(gridDimension,numPOI)
    data["training_data"].append(n)
    test = np.logical_or.reduce((m==31,m==32,m==33,m==34))
    data["training_labels"].append(test.astype(int))
    print(
    '\r[Generating Training Data {} of {}]'.format(
        i,
        int(trainRatio * totalData),
    ),
    end=''
    )
print('')

for i in range(int((1 - trainRatio) * totalData +1)):
    m, n = getTiles(gridDimension,numPOI)
    data["testing_data"].append(n)
    test = np.logical_or.reduce((m==31,m==32,m==33,m==34))
    data["testing_labels"].append(test.astype(int))
    #print("testing")
    print(
    '\r[Generating Testing Data {} of {}]'.format(
        i,
        int((1 - trainRatio) * totalData +1),
    ),
    end='',
    )
print('') 

with open('synthetic_dataset_{}.pickle'.format(GRID_SIZE), 'wb') as handle:
    pickle.dump(data, handle)
