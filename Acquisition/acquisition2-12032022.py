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
from time import sleep
### OLD pycro-manager code ###
##############################
## Connect to running ZMQ server (port 4897) and micromanager
core = Bridge() ## from pycromanager import Core; core = Core()
c = core.get_core() ## core.snap_image()
mmStudio = core.get_studio() ## from pycromanager import Studio; mmStudio = Studio()
### Initialise connection with micro manager
##############################

### Autofocus function
z_stage = c.get_focus_device()
z_pos = c.get_position(z_stage)
afm = mmStudio.get_autofocus_manager()
c = core.get_core() ## core.snap_image()

#afm_method = afm.get_autofocus_method()
#afm.refresh()
#afm.set_autofocus_method_by_name('JAF(H&P)')
#c = afm.get_all_autofocus_methods()
#print(mmStudio.get_position_list_manager())

"""
def autofocus(coords):    
    c.set_xy_position(c.get_xy_stage_device(), float(coords[0]), float(coords[1]))
    sleep(5) 
    c.set_position(c.get_focus_device(),float(coords[2]))
    sleep(1)
    mmStudio.autofocus_now() ## run current Micro-manager autofocus pilot
    sleep(50)
"""
###
def tenengrad(img,ksize=3):
    Gx = cv2.Sobel(img,ddepth=cv2.CV_64F, dx=1, dy = 0, ksize=ksize)
    Gy = cv2.Sobel(img,ddepth=cv2.CV_64F, dx=0, dy = 1, ksize=ksize)
    FM = Gx**2 + Gy ** 2
    return FM

def fibonacci(size,**kwargs):
    fn_1 = 1
    fn_2 = 0
    fn_3 = 0
    fn = 0
    val = 0
    while fn < size:
        fn = fn_1 + fn_2
        fn_3 = fn_2
        fn_2 = fn_1
        fn_1 = fn
        val += 1
    if "mode" in kwargs and kwargs["mode"] == "range":     
        return val
    else:
        return fn_3/fn

def fib_autofocus(coords=[1000,1000],mode="timelapse",z_search=100):
    if mode == "timelapse":
        c.set_xy_position(c.get_xy_stage_device(), float(coords[0]), float(coords[1]))
        sleep(5)
    z = c.get_position(z_stage)
    z_range = np.arange(round(z-z_search,2),round(z+z_search,2),0.5)

    size = len(z_range)
    a = z_range[0]
    b = z_range[-1]
    var = np.zeros((2,1))


    for s in range(fibonacci(size,mode="range")):
        if size > 1:
            if s == 0:
                diff = b - a
                sep = fibonacci(size)
                
                a_pos = a + round((sep * (diff)) * 2)/2
                b_pos = b - round((sep * (diff)) * 2)/2
            
            testpos = list((a_pos,b_pos))
            for i,num in enumerate(testpos):    

                """
                c.set_position(c.get_focus_device(),float(testpos[i]))
                sleep(0.2)
                c.snap_image()
                
                tagged_image = c.get_tagged_image()
                image_height = tagged_image.tags['Height']
                image_width = tagged_image.tags['Width']
                image = tagged_image.pix.reshape((image_height, image_width))

                plt.imshow(image, cmap='gray')
                plt.show()
                """
                k  = randint(0,1)
                #var[i] = cv2.Laplacian(image,cv2.CV_64F).var()
                #var[i] = tenengrad(image)
                if k ==1:
                    var = [0,1]
                else:
                    var = [1,0]
            if var[0] < var[1]:
                a = a_pos
                b = b
                a_pos = b_pos
                diff = abs(b - a)
                size = len(np.arange(a,b,0.5))
                sep = fibonacci(size)
                b_pos = b - abs(round(sep * diff * 2)/2)
            
            elif var[0] > var[1]:
                a = a
                b = b_pos
                b_pos = a_pos
                diff = abs(b - a)
                size = len(np.arange(a,b,0.5))
                sep = fibonacci(size)
                a_pos = a + abs(round(sep * diff * 2)/2)

    if a_pos == b_pos:
        print(a_pos)
        #xyz[0,2] = a_pos
    elif var[0] < var[1]:
        print(b_pos)
        #c.set_position(c.get_focus_device(),float(b_pos))
        #xyz[0,2] = b_pos
    elif var[0] > var[1]:
        print(a_pos)
        #c.set_position(c.get_focus_device(),float(a_pos))
        #xyz[0,2] = a_pos 
    print(diff)
    s
    
### test functions
"""
c.snap_image()
c.set_relative_xy_position(c.get_xy_stage_device(), 200, 0)
c.set_property("Andor sCMOS Camera","FrameRate",10)
core.get_focus_device()
"""
##
tagged_image = core.get_tagged_image()

viddir = r"F:\\test-acquisition\\"

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
save = r'D:/cg281'

hour_duration = 24 ## total duration
imfrequency = 4 ## frequency of image acquisitions (h)

duration = 10 ## duration in seconds of image acquisitions. Default: 10s
frate = 10 ## frame rate. Default: 10 fps

total_duration = hour_duration * 360 # total duration in seconds
total_duration = 30
subset_interval = 10
##### Set number of FOVs. Ideally, this would be automated but for now, this should be
## extracted from the coordinate manager in micro manager

num_FOV = 6
x = np.array((7522.5,7129.3999,3040,68385,70683,66630)) ## Initial set of x coordinates from ROI list
y = np.array((-2016.9,-2254.1001,-4743.6001,-5347.2998,-2649.3999,-3944)) ## Initial set of y coordinates from ROI list
z = np.array((2142.225,2142.225,2142.225,2202.85,2209.95,2183.6)) ## Initial set of z coordinates from ROI list
xyz = np.hstack([x[:,None],y[:,None],z[:,None]]) ## combination

subset_interval = imfrequency * 360 # number of separate timelapse intervals

num_subsets = int(np.ceil(total_duration/subset_interval))
num_time_points = duration * frate

events = [] # set empty event list to fill with dictionary of number of time points (100) per subset in time per field of view
for s in range(num_subsets):
    
    for e in range(num_FOV):
        
        for t in range(num_time_points):
            
            events.append(
                {
                    "axes": {"time": t, "position": e,"subset":s}, 
                    'x' : xyz[e,0], 
                    'y' : xyz[e,1], 
                    'z' : xyz[e,2],
                    "min_start_time": s * subset_interval,
                }
            )
        if e ==0 and s==0:
            run = events[e*100 :e*100+ 100]
        if e == 0 and s!=0:
            run = events[e*100 + num_FOV*100 :e*100+num_FOV*100 + 100]
        else:
            run = events[e*100:e*100+100]
        name = "FOV_" + str(e).zfill(3) + "_time_" + str(s * subset_interval).zfill(3)
        with Acquisition(directory=save, name=name,post_hardware_hook_fn=fib_autofocus(xyz[e,0:3])) as acq:
            acq.acquire(run)


save_name = r'Acq-3F'
### Acquisition
v = np.zeros((1,2))
v[0] = 1641.7
v[0,1] = -135

##########################################################
##########################################################
###################Engram reconstruction##################
##########################################################
##########################################################


import matplotlib.pyplot as plt
import skimage
import skimage.io
import cv2
import os

os.chdir(r'D:/cg281')
import functions3

viddir = r'D:/cg281'
imdir = r'D:/cg281'

os.chdir(viddir)
c.snap_image()

tagged_image = c.get_tagged_image()
image_height = tagged_image.tags['Height']
image_width = tagged_image.tags['Width']
image = tagged_image.pix.reshape((image_height, image_width))

plt.imshow(image, cmap='gray')
plt.show()
laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
laplacian_var


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


##########################################################################
#####################Data roving##########################################
##########################################################################
import random
import skimage as ski
import matplotlib.pyplot as plt
########################################

def get_ROIs(coords,viddir,name):
    c.snap_image()
    
    tagged_image = c.get_tagged_image()
    image_height = tagged_image.tags['Height']
    image_width = tagged_image.tags['Width']
    image = tagged_image.pix.reshape((image_height, image_width))

    plt.imshow(image, cmap='gray')
    plt.show()
    
    
    #viddir = viddir + "/" + name + "/" + "Full resolution" + "/"
    ROIs = functions3.get_ROIs(viddir,image=image)
    inference_coords = functions3.get_coords(viddir, test_coords, imdir,image=image)
    return ROIs,inference_coords

def run_acquisition(viddir,name,coords):
    with Acquisition(directory=viddir, name=name,post_hardware_hook_fn=autofocus()) as acq:
        events = multi_d_acquisition_events(xy_positions=shape, order='pt',num_time_points=100)
        acq.acquire(events)
    vid = viddir + "/" + name + "/" + "Full resolution" + "/"
    ROIs,inference_coords = get_ROIs(coords,vid)
    #functions3.get_intensities(vid,coords)
def remove_FOV(coords):
    remove = np.transpose(np.array(ski.draw.circle_perimeter(test_coords[0],test_coords[1],50)))
    remove = np.array([list(zip(remove[:,0],remove[:,1]))])
    filled = cv2.fillPoly(mask, remove,color=(0))
    matrix = np.argwhere(mask!=0)
    return matrix

#########################################################################

num_FOV = 10

mask = np.zeros((5000,5000),dtype=np.uint8)

#########################################################################

### Set radius of well ###
well = np.transpose(np.array(ski.draw.circle_perimeter(2618,2192,400))) ### 
well = np.array([list(zip(well[:,0],well[:,1]))]) ## Zip coordinates of circumference together
filled = cv2.fillPoly(mask, well,color=(1)) ### Fill in mask
matrix = np.argwhere(mask!=0) ## create a 2-column matrix of values where the mask != 0, i.e. the scan area still filled. Later searching
                              ## eliminates failed or completed FOVs from the search space by setting area equal to 0 

FOVs = np.zeros((5,3))
iterations = 0
e = 0

z = c.get_position(z_stage)
while e <= num_FOV and iterations < 100:
    shape = np.zeros((1,2))
    test_coords = (random.choice(matrix))
    c.set_xy_position(c.get_xy_stage_device(), float(test_coords[0]), float(test_coords[1]))
    sleep(5)
    shape[0],shape[0,1] = test_coords[0],test_coords[1]
    ### Move to random and inspect###
    name = "inspection-" + str(0).zfill(3)  
    fib_autofocus(z,mode="roving")
    z = c.get_position(z_stage)
    c.set_position(c.get_focus_device(),float(z))
    ROIs,inference_coords = get_ROIs(test_coords, viddir, name)
    
    if ROIs == None:
        ROIs = 0

    if ROIs < 5 or ROIs == None:
        matrix = remove_FOV(test_coords)
    elif ROIs >= 5:
        ROIb = ROIs
        test_coord = tuple((np.mean(inference_coords[:,0]),np.mean(inference_coords[:,1])))
        
        ROIs,inference_coords = get_ROIs(test_coord,viddir,name)
        
        if ROIs <= 5:
            name = "FOV_" + str(e).zfill(3)
            ROIs,inference_coords = run_acquisition(name,test_coords)
            FOVs[e,0] = e
            FOVs[e,1:3] = test_coords
            e += 1
            matrix = remove_FOV(test_coords)
            
        else:
            while ROIs > ROIb: 
                ROIb = ROIs
                test_coords = tuple((np.mean(inference_coords[:,0]),np.mean(inference_coords[:,1])))
                ROIs,inference_coords = run_acquisition(name,test_coords)
                
                matrix = remove_FOV(test_coords)
                
            if ROIs >=  10:
                run_acquisition(name,test_coords)
                name = "FOV_" + str(e).zfill(3)
                FOVs[e,0] = e
                FOVs[e,1:3] = test_coords
                e += 1
                matrix = remove_FOV(test_coords)
     
    iterations += 1           
            