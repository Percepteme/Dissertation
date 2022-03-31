#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 10:40:05 2022
"""

## Script for taking all raw intensity csv files, normalising and converting them into a single macro-csv file for analysis of
## the trace data

##################
import numpy as np
import os 
import glob
import pandas as pd
import re
from natsort import natsorted, ns
##################
datdir = "..."
updatdir = "..."
os.chdir(updatdir)
########## Set directory
apset = False

########## Extract all csv files across multiple subdirectories 
subdirs = [name for name in os.listdir(".") if os.path.isdir(name)]  
for i,num in enumerate(subdirs):
    ints = sorted(
        glob.glob(os.path.join(os.path.abspath(subdirs[i]+"/" + "*.csv"))), key=os.path.getmtime)    
    if len(ints) != 0:
        for csv,j in enumerate(ints):
            os.rename(ints[csv],os.path.join(updatdir + os.path.basename(ints[csv])))
    else:
        pass
### sort the files using natural sorting 
ints = sorted(
    glob.glob(os.path.join(updatdir + "/" + "*.csv")), key=os.path.getmtime) 
ints = natsorted(ints, key=lambda y: y.lower())

### initialise an array of independent variables to categorise the data using
vartab = np.zeros((1,3),dtype="object")


for csv,i in enumerate(ints):
    #######################
    dat = np.array(pd.read_csv(ints[csv],header=None))
    
    inttime = 0.1
    time = np.multiply(dat[:,0],inttime)
    
    f0 = dat[0,1:]
    df = np.subtract(dat[:,1:], f0)
    datdiv = np.divide(df,f0)
    datnorm = np.c_[time,datdiv]
    ######################## normalise the data
    
    ########################
    numpulse = int(re.search("NumPulses_(.+?)-",ints[csv]).group(1)) 

    width = int(re.search("PW_(.+?)_",ints[csv]).group(1))
    fov = int(re.search("FOV_(.+?)-",ints[csv]).group(1))
    drug = int(re.search("D_(.+?)-",ints[csv]).group(1))
    ######################## identify the relevant variable level in the file name 
    if apset == True:
        ap = re.search("AP_(.+?)-",ints[csv]).group(1)
        meta = np.tile([numpulse,width,fov,ap],(len(datnorm),1)) ## separate selection on the aperture diameter, "full" or "small" 
    else:
        meta = np.tile([numpulse],(len(datnorm),1))
    datnorm = np.append(datnorm,meta,axis=1)
    for j in range(len(dat[0])-3):
        vartab = np.append(vartab,datnorm[:,[0,j+1,-1]],axis=0) ## append the raw data and suffixed metadata (meta) for each csv file
save = np.savetxt(os.path.join(updatdir + "/" + "....csv"), vartab,delimiter=",",fmt="%s") ## save the output

#####################



