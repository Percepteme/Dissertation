#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 10:40:05 2022

@author: caiusgibeily
"""

## High-level analysis

import numpy as np
import os 
import glob
import pandas as pd
import re
from natsort import natsorted, ns
datdir = "/home/caiusgibeily/Downloads/Training-Optocamp/test/Low-level-analysis/CNQX/after/session_2-after/FOV_0/"
updatdir = "/home/caiusgibeily/Downloads/Training-Optocamp/test/Low-level-analysis/Aβ/session_4-Control/"
os.chdir(updatdir)

"""
ints = sorted(
    glob.glob(os.path.join(datdir + "/" + "*.csv")), key=os.path.getmtime) 
"""
apset = False

subdirs = [name for name in os.listdir(".") if os.path.isdir(name)]  
for i,num in enumerate(subdirs):
    ints = sorted(
        glob.glob(os.path.join(os.path.abspath(subdirs[i]+"/" + "*.csv"))), key=os.path.getmtime)    
    if len(ints) != 0:
        for csv,j in enumerate(ints):
            os.rename(ints[csv],os.path.join(updatdir + os.path.basename(ints[csv])))
    else:
        pass

ints = sorted(
    glob.glob(os.path.join(updatdir + "/" + "*.csv")), key=os.path.getmtime) 
ints = natsorted(ints, key=lambda y: y.lower())
vartab = np.zeros((1,3),dtype="object")

#dat = np.array(pd.read_csv("5_0-FOV_1-NumPulse_4-AP_small-D_0-005_img_005_intensities.csv",sep=","))
#targetpulse = 1
drug = 25
for csv,i in enumerate(ints):    
    dat = np.array(pd.read_csv(ints[csv],header=None))
    
    inttime = 0.1
    time = np.multiply(dat[:,0],inttime)
    
    f0 = dat[0,1:]
    df = np.subtract(dat[:,1:], f0)
    datdiv = np.divide(df,f0)
    datnorm = np.c_[time,datdiv]
    #dat = dat[:,1:]
    
    numpulse = int(re.search("NumPulses_(.+?)-",ints[csv]).group(1))

    #width = int(re.search("PW_(.+?)_",ints[csv]).group(1))
    #fov = int(re.search("FOV_(.+?)-",ints[csv]).group(1))
    #drug = int(re.search("D_(.+?)-",ints[csv]).group(1))
    #if numpulse == targetpulse:
        
    if apset == True:
        ap = re.search("AP_(.+?)-",ints[csv]).group(1)
        meta = np.tile([numpulse,width,fov,ap],(len(datnorm),1)) 
    else:
        meta = np.tile([numpulse],(len(datnorm),1))
    datnorm = np.append(datnorm,meta,axis=1)
    for j in range(len(dat[0])-3):
        vartab = np.append(vartab,datnorm[:,[0,j+1,-1]],axis=0)
save = np.savetxt(os.path.join(updatdir + "/" + "session_4-Ab_control-FOV_2.csv"), vartab,delimiter=",",fmt="%s")


import matplotlib.pyplot as plt
dat = np.array(pd.read_csv("session_4-Ab_control-FOV_0.csv",header=None))
plt.plot(dat[:,0],dat[:,1])


#####################



