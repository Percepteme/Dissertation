#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pycro-manager integration

@author: caiusgibeily
"""

from pycromanager import Acquisition, multi_d_acquisition_events, Bridge
import numpy as np
import argparse as ag
import os


viddir = "/home/caiusgibeily/Downloads/Training-Optocamp/test/"

### Pipeline: 
    # 1. Initialise libraries for pycromanager and cellpose/analysis ONCE and then provide a customisable carte of analysis options. 
    # The script is intended to be run once per job:
        # a. timelapse of timelapse analysis - this would enable short temporal sequences for an array of FOVs to be acquired 
        # over multiple timepoints (Ab procedure)
        
        # b. engram analysis - this would initially be a manual operation per selected FOV. Its function would be two-fold: 
            # i - machine inference of cell pose by acquiring 100 frames with n pulses of stimulation to aid machine labelling. 
            # This will output coordinate data rather than intensities for each ROI centroid and temp mask data. Based on inference-analysis2
            # ia - The tmp mask data contain the coordinates of the masks which, following stage translation, will be updated using the 
            # Euclidean equation to avoid having to re-run inference. Change output loop in inference-analys2
            # ii - the data will be read from the object variable array and used in pycromanager to control XY position. Within Acquisition
            # The acquisition engine will then image 100 frames with specified pulse parameters in the arduino controls for each ROI 
            # It will then use the mask data to output iteratively the intensity data for FOVs (csvs), resulting in n csv files for n 
            # neurons, enabling high-dimensional engram reconstruction - use translated masks 

##########################################################
##########################################################
########## Timelapse of timelapse analysis ###############
##########################################################
##########################################################

bridge = Bridge() ## initialise java objects
mmc = bridge.get_core()
mmStudio = bridge.get_studio()

hour_duration = 10 ## total duration
imfrequency = 2 ## frequency of image acquisitions (h)

duration = 10 ## duration in seconds of image acquisitions. Default: 10s
frate = 10 ## frame rate. Default: 10 fps

total_duration = hour_duration * 360 # total duration in seconds
##### Set number of FOVs. Ideally, this would be automated but for now, this should be
## extracted from the coordinate manager in micro manager
num_FOV = 5
x = np.array((10,4,6,8,9))
y = np.array((2,3,7,8,9))
z = np.array((1,1,1,1,1))
xyz = np.hstack([x[:,None],y[:,None],z[:,None]])

subset_interval = imfrequency * 360 # number of separate timelapse intervals

num_subsets = int(np.ceil(total_duration/subset_interval))
num_time_points = duration * frate

events = [] # set empty event list to fill with dictionary of number of time points (100) per subset in time per field of view
for e in range(num_FOV):
    
    for s in range(num_subsets):
        
        for t in range(num_time_points):
            
            events.append(
                {
                    "axes": {"subset": s, "time": t}, 
                    'x' : xyz[e,0], 
                    'y' : xyz[e,1], 
                    'z' : xyz[e,2],
                    "min_start_time": s * subset_interval,
                }
            )
 
### Preparing for acquisition phase
## setting up camera
mmc.set_property("Camera", "Framerate", frate)
## z stage
z_stage = mmc.get_focus_device()
bridge.close()

### Acquisition
with Acquisition(directory=viddir, name="timelapse-analysis") as acq:
    acq.acquire(events)


##########################################################
##########################################################
###################Engram reconstruction##################
##########################################################
##########################################################


import matplotlib.pyplot as plt
import skimage
import skimage.io
import cv2


os.chdir('/home/caiusgibeily/Downloads/Training-Optocamp')
import functions3

viddir = "/home/caiusgibeily/Downloads/Training-Optocamp/function-test/"
imdir = "/home/caiusgibeily/Downloads/Training-Optocamp/function-test/"

os.chdir(viddir)


num_FOV = 1
x = np.array((500,2))
y = np.array((500,2))
z = np.array((1,2))
xyz = np.hstack([x[:,None],y[:,None],z[:,None]])
################################## 1 use inference to obtain coordinates 
def run_intensity(e,coords):
    functions3.get_intensities(viddir,coords = coords)  ## create intensity function instance to call in post acquisition hook

for e in range(num_FOV):
    coords = np.transpose(xyz[e,0:2]) ## transpose into form (1,2)
    inference_coords = functions3.get_coords(viddir,imdir=imdir,coords=xyz[0,0:2]) ## extract coordinate data of all neurons in the field of view


    for i in range(len(inference_coords)):
        name = "engram_acquisition" + str(i).zfill(3)
        with Acquisition(directory=viddir, name=name, 
                         post_camera_hook_fn=functions3.engram_intensities(viddir, imdir, coords,inference_coords[i])) as acq:
        
            events = multi_d_acquisition_events(
                                        min_start_time = 0, num_time_points=100, time_interval_s=0,
                                        channel_group='Channel',
                                        order='tp',
                                        xy_positions=inference_coords)
        
        acq.acquire(events)


for e in range(num_FOV):
    coords = np.transpose(xyz[e,0:2])
    inference_coords = functions3.get_coords(viddir,imdir=imdir,coords=coords)

    for i in range(len(inference_coords[:,0])):
        functions3.get_intensities(viddir, coords)







##########################################################################
#####################Data roving##########################################
##########################################################################

import random
import skimage as ski
num_FOV = 10

mask = np.zeros((1000,1000),dtype=np.uint8)

#########################################################################
def get_ROIs(coords):
    ROIs = functions3.get_intensities(viddir,coords)
    inference_coords = functions3.get_coords(viddir, coords, imdir)
    return ROIs,inference_coords

def run_acquisition(name,coords):
    with Acquisition(directory=viddir, name=name,post_camera_hook_fn=get_ROIs(coords)) as acq:
    
        events = multi_d_acquisition_events(
                                    min_start_time = 0, num_time_points=20, time_interval_s=0,
                                    channel_group='Channel',
                                    order='tp',
                                    xy_positions=test_coords)
    
    acq.acquire(events)
    return ROIs,inference_coords

def remove_FOV(coords):
    remove = np.transpose(np.array(ski.draw.circle_perimeter(test_coords[0],test_coords[1],5)))
    remove = np.array([list(zip(remove[:,0],remove[:,1]))])
    filled = cv2.fillPoly(mask, remove,color=(0))
    matrix = np.argwhere(mask!=0)
    return matrix

### Set radius of well ###
well = np.transpose(np.array(ski.draw.circle_perimeter(100,100,50))) ### 
well = np.array([list(zip(well[:,0],well[:,1]))]) ## Zip coordinates of circumference together
filled = cv2.fillPoly(mask, well,color=(1)) ### Fill in mask
matrix = np.argwhere(mask!=0) ## create a 2-column matrix of values where the mask != 0, i.e. the scan area still filled. Later searching
                              ## eliminates failed or completed FOVs from the search space by setting area equal to 0 

FOVs = np.zeros((num_FOV,3))
iterations = 0
while e <= num_FOV and iterations < 100:
    test_coords = tuple(random.choice(matrix))
    
    ### Move to random and inspect###
    name = "inspection-" + str(e).zfill(3)  
    
    ROIs,inference_coords = run_acquisition(name,test_coords)
    
    if ROIs <= 8:
        matrix = remove_FOV(test_coords)
    elif ROIs > 8:
        ROIb = ROIs
        test_coord = tuple(np.mean(inference_coords[:,0]),np.mean(inference_coords[:,1]))
        
        ROIs,inference_coords = run_acquisition(name,test_coord)
        
        if ROIs <= 8:
            name = "FOV_" + str(e).zfill(3)
            ROIs,inference_coords = run_acquisition(name,test_coords)
            FOVs[e,0] = e
            FOVs[e,1:3] = test_coords
            e += 1
            matrix = remove_FOV(test_coord)
            
        else:
            while ROIs > ROIb: 
                ROIb = ROIs
                test_coords = tuple(np.mean(inference_coords[:,0]),np.mean(inference_coords[:,1]))
                ROIs,inference_coords = run_acquisition(name)
                
                matrix = remove_FOV(test_coords)
                
            if ROIs >=  10:
                run_acquisition(name,test_coords)
                name = "FOV_" + str(e).zfill(3)
                FOVs[e,0] = e
                FOVs[e,1:3] = test_coords
                e += 1
                matrix = remove_FOV(test_coords)
     
    iterations += 1           
            