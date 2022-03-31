#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 20:55:41 2022

@author: caiusgibeily
"""

import numpy as np
import cv2
import pandas as pd
import os
import matplotlib.pyplot as plt
os.chdir('/home/caiusgibeily/Downloads')

img0 = cv2.imread('007_img.png')
img10 = cv2.imread('img_007-10.png')
img20 = cv2.imread('img_007-20.png')
img30 = cv2.imread('img_007-30.png')
img80 = cv2.imread('img_007-80.png')

img0_0 = cv2.imread('008_img.tif')

img0_10 = cv2.imread('img_008-10.png')
img0_15 = cv2.imread('img_008-15.png')
img0_30 = cv2.imread('img_008-30.png')

img1_1 = cv2.imread('img_009-1.png')
img1_10 = cv2.imread('img_009-10.png')
img1_20 = cv2.imread('img_009-20.png')

plt.imshow(img30)
    

var = 1/(1024*1024) * sum(0,sum(img0))
var = 0

img = img20
mean = np.average(img20[:,:,0])
img = img[:,:,0]
for i in range(len(img[:,0])):
    for j in range(len(img[0,:])):
        img[i,j] = (img[i,j] - mean)
        var += img[i,j]
        
var = var/(1024*1024)
var
vals = 191, 143, 131, 134, 126

img2 = np.sum(np.square(np.absolute(np.subtract(img,mean))))/(1024*1024)
img2
vals = 4742, 6846, 4060

cv2.Laplacian(img1_10,cv2.CV_64F).var()

############## autofocus function #############
import cv2
import pycromanager
core = Bridge().get_core()

def autofocus():
    z_pos = c.get_zstage_coordinates()
    z_range = np.arange(z_pos-50,z_pos+50,0.5) ## qualify a and b search extrema 
    
    #### initiate Fibonacci search ###
    i_length = 30 ## iteration number
    
    for k in range(i_length):
        if k == 0:
from random import randint, choice
            
def fibonacci(size):
    fn_2,fn_1 = 0,1
    fn = 0
    fn_3 = 0
    while fn < size:
        fn = fn_2 + fn_1
        fn_3 = fn_2
        fn_2 = fn_1
        fn_1 = fn
    return (fn_3/fn)
    
2150
2250
size = len(z_range)
   
#a_pos = a + round((sep_a * (size - 1)) * 2)/2
#b_pos = a + round((sep_b * (size - 1)) * 2)/2  

a = z_range[0]
b = z_range[-1]
var = np.zeros((2,1))

for s in range(144):
    if size > 1:
        if k == 0:
            diff = b - a
            sep = fibonacci(size-1)
            
            a_pos = a + round((sep * (diff)) * 2)/2
            b_pos = b - round((sep * (diff)) * 2)/2
        
        testpos = list((a_pos,b_pos))
        for i,num in enumerate(testpos):    
            """
            z = change_z_pos(float(testpos[i])), ### code to o
            sleep(0.4)
            core.snap_image()
            core.snap_image()
            image = 23
            var[i] = cv2.Laplacian(image,cv2.CV_64F).var()
            """
            k = randint(0,1)
            if k == 0:
                var[0],var[1] = 0,1
            else:
                var[0],var[1] = 1,0
            ### code to process image into 1024x1024 array
        if var[0] < var[1]:
            a = a_pos
            b = b
            a_pos = b_pos
            diff = abs(b - a)
            sep = fibonacci(size)
            b_pos = b - abs(round(sep * diff * 2)/2)
            
        elif var[0] > var[1]:
            a = a
            b = b_pos
            b_pos = a_pos
            diff = abs(b - a)
            sep = fibonacci(size)
            a_pos = a + abs(round(sep * diff * 2)/2)
            
if a_pos == b_pos:
    print(a_pos)
        
elif var[0] < var[1]:
    print(b_pos)
    #change_z_pos(float(b_pos))
    #return b_pos

elif var[0] > var[1]:
    print(a_pos)
    #change_z_pos(float(a_pos))
    #return a_pos

print(diff)
s
size = len(z_range)
   
#a_pos = a + round((sep_a * (size - 1)) * 2)/2
#b_pos = a + round((sep_b * (size - 1)) * 2)/2  

a = z_range[0]
b = z_range[-1]
var = np.zeros((2,1))

for s in range(144):
    if size > 1:
        if k == 0:
            diff = b - a
            sep = fibonacci(size-1)
            
            a_pos = a + round((sep * (diff)) * 2)/2
            b_pos = b - round((sep * (diff)) * 2)/2
        
        testpos = list((a_pos,b_pos))
        for i,num in enumerate(testpos):    
            """
            z = change_z_pos(float(testpos[i])), ### code to o
            sleep(0.4)
            core.snap_image()
            core.snap_image()
            image = 23
            var[i] = cv2.Laplacian(image,cv2.CV_64F).var()
            """
            k = randint(0,1)
            if k == 0:
                var[0],var[1] = 0,1
            else:
                var[0],var[1] = 1,0
            ### code to process image into 1024x1024 array
        if var[0] < var[1]:
            a = a_pos
            b = b
            a_pos = b_pos
            diff = abs(b - a)
            sep = fibonacci(size)
            b_pos = b - abs(round(sep * diff * 2)/2)
            
        elif var[0] > var[1]:
            a = a
            b = b_pos
            b_pos = a_pos
            diff = abs(b - a)
            sep = fibonacci(size)
            a_pos = a + abs(round(sep * diff * 2)/2)
            
if a_pos == b_pos:
    print(a_pos)
        
elif var[0] < var[1]:
    print(b_pos)
    #change_z_pos(float(b_pos))
    #return b_pos

elif var[0] > var[1]:
    print(a_pos)
    #change_z_pos(float(a_pos))
    #return a_pos

print(diff)
s