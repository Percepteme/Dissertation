#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 12:15:25 2022

@author: caiusgibeily
"""
import numpy as np
import cv2
from random import randint

z = set_z_position
z_range = np.arange(z-50,z+50,0.5)


#a_pos = a + round((sep_a * (size - 1)) * 2)/2
#b_pos = a + round((sep_b * (size - 1)) * 2)/2  

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

size = len(z_range)
a = z_range[0]
b = z_range[-1]
var = np.zeros((2,1))


for s in range(144):
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