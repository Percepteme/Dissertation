#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 10:36:00 2022

@author: caiusgibeily
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as col
import pandas as pd
import glob, os

os.chdir("/home/caiusgibeily/Downloads/Training-Optocamp/test/output")
kymo = np.array(pd.read_csv("test.csv",header=None))

kymo = kymo[:,1:]

F0 = kymo[0,1:]
DF = np.subtract(kymo, F0)
DFNormF0 = np.divide(DF,F0)

## Normalise data
kymonorm = np.divide(kymo,4096)


data = np.random.rand(10, 10) * 20

# create discrete colormap





df = np.array(pd.read_csv("test.csv",header=None))

# variable to store the camera integration time (in seconds) as this always gets lost in the mail...
IntTime = 0.1
vid = len(df)*IntTime
# Convert frame number, stored in the first column, to time, by multiplying by IntTime. Start at time = 0 by subtracting IntTime.
time = np.multiply(df[:,0],IntTime)
data = df[:,1:]

F0 = df[0,1:]
DF = np.subtract(data, F0)
DFNormF0 = np.transpose(np.divide(DF,F0))

fig, ax = plt.subplots()
ax.imshow(DFNormF0, cmap="plasma")
plt.yticks(np.arange(len(DFNormF0),step=4))
