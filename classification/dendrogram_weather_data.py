#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 18:20:56 2020

@author: a
"""

import numpy as np
import scipy.spatial.distance as sd
import scipy.cluster as sc
import matplotlib.pyplot as plt


## To generate distance matrix for dendrogram:
def distanceMat(data):

    distanceMat = np.zeros((data.shape[0], data.shape[0])) 
    ## (no. of rows) x (no. of rows) matrix

    ## Calculating Euclidean distance:
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            distanceMat[i, j] = np.linalg.norm(data[i] - data[j])
            ## i.e. np.sqrt(np.sum(data[i] - data[j]))
    
    return distanceMat

def loadData(filePath):
    dataRaw = []
    DataFile = open(filePath, "r")

    while True:
        theline = DataFile.readline()

        if len(theline) == 0:
            break

        theline = theline.rstrip()

        readData = theline.split(",")
        for i in range(len(readData)):
            readData[i] = float(readData[i]);

        dataRaw.append(readData)

    DataFile.close()

    data = np.array(dataRaw)

    return data

data = loadData("SCC403CWWeatherData.txt")


## Creating distance matrix and condensing to a vector:
condensedDist = sd.squareform(distanceMat(data))

## Creating linkage information:
linkageInfo = sc.hierarchy.linkage(condensedDist)

## Plotting dendrogram:
plt.figure(figsize = (6,4))
sc.hierarchy.dendrogram(linkageInfo, truncate_mode="lastp", p=12)
plt.savefig("dendrogram.pdf")