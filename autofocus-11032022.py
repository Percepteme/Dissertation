#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 12:15:25 2022

@author: caiusgibeily
"""
import numpy as np
import cv2
from random import randint
from time import sleep
from pycromanager import Acquisition, multi_d_acquisition_events, Bridge
import matplotlib.pyplot as plt

core = Bridge() ## from pycromanager import Core; core = Core()
c = core.get_core() ## core.snap_image()

z_stage = c.get_focus_device()

z = c.get_position(z_stage)
z_range = np.arange(round((z-50),2),round((z+50),2))

def modifiedLaplacian(img):
    M = np.array([-1,2,-1])
    G = cv2.getGaussianKernel(ksize=3,sigma=-1)
    Lx = cv2
def tenengrad(img,ksize=3):
    Gx = cv2.Sobel(img,ddepth=cv2.CV_64F, dx=1, dy = 0, ksize=ksize)
    Gy = cv2.Sobel(img,ddepth=cv2.CV_64F, dx=0, dy = 1, ksize=ksize)
    FM = Gx**2 + Gy ** 2
    
    return cv2.mean(FM)[0]
def fibonacci(size):
    fn_1 = 1
    fn_2 = 0
    fn_3 = 0
    fn = 0
    while fn < size:
        fn = fn_1 + fn_2
        fn_3 = fn_2
        fn_2 = fn_1
        fn_1 = fn
    return fn_3/fn

z = c.get_position(z_stage)
z_range = np.arange(round((z-100),2),round((z+100),2))

size = len(z_range)
a = z_range[0]
b = z_range[-1]
var = np.zeros((2,1))


for s in range(20):
    if size > 1:
        if s == 0:
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
            
            c.set_position(c.get_focus_device(),float(testpos[i]))
            sleep(0.5)
            c.snap_image()
            
            tagged_image = c.get_tagged_image()
            image_height = tagged_image.tags['Height']
            image_width = tagged_image.tags['Width']
            image = tagged_image.pix.reshape((image_height, image_width))

            plt.imshow(image, cmap='gray')
            plt.show()
            #var[i] = cv2.Laplacian(image,cv2.CV_64F).var()
            var[i] = tenengrad(image)
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
    c.set_position(c.get_focus_device(),float(b_pos))
    #return b_pos

elif var[0] > var[1]:
    print(a_pos)
    c.set_position(c.get_focus_device(),float(b_pos))
    #return a_pos

print(diff)
s