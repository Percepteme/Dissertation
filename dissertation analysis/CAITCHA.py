#!/usr/bin/env python3
# -*- coding: utf-8 -*-

############### CAITCHA analysis ######################

### import
import numpy as np
import pandas as pd
import os
import random as rd
###
datdir = "..." 
os.chdir(datdir)
## read csv file

dat = pd.read_csv("intensities.csv",sep=",",header=0)


### Mean absolute percentage error calculation using ra####
## For every image, compute the mean absolute error for every participant
### 1. Least difference between reference and test mean intensities are determined and the test data reordered to match the reference
### 2. The absolute test values are then subtracted elementwise from the reference values and divided by the reference.
### 3. These values are then summed and divided by the length of the reference array to give the mean absolute error
### 4. The mean total error is given by summing across all imaging trials and dividing by the number of test images.
rd.seed(56) # set seed to ensure deterministic solution

def meanperror(arr,part):
    array = np.array(arr)
    img = np.unique(np.array(array[1:,6]))
    error = np.zeros((len(img),part))
    for i in range(len(img)):
        c = np.array(arr.loc[arr["img"] == i+1])
        
        #part_error = []
        for j in range(part):
            c = np.array(arr.loc[arr["img"] == i+1])
            parr = c[:,j+1]
            parlen = len([i for i in parr if i!=0])
            conlen = len([i for i in c[:,0] if i!=0])
            maxval = max(parlen,conlen)
            
            c = c[0:maxval,:]
            parr = c[:,j+1]
            
            ordered = []
            control_error = []
            
            for k in range(len(c[:,0])):
                control = np.copy(c[:,0])
                abindex = np.abs(parr - control[k]).argmin()
                ordered.append(parr[abindex])
                control[k] = 0
                parr[abindex] = 0
                
                
                control = np.copy(c[:,0])
                abindex = np.absolute(control - control[k]).argmin()

                control_error.append(control[abindex])
                control[k] = 0
            print(control_error)
            print(ordered)
            ### test ###
            abvalue = np.absolute(np.subtract(c[:,0],np.array(ordered)))

            setval = np.ones(abvalue.shape)
            diff = np.divide(abvalue,c[:,0],out=setval, where=abvalue!=0)        
            test_sum = (np.sum(diff))/(len(c[:,0])) 
            
            ### control ###
            abvalue = np.absolute(np.subtract(c[:,0],np.array(control_error)))

            diff = np.divide(abvalue,c[:,0],out=abvalue, where=c[:,0]!=0)
            control_sum = (np.sum(diff)/len(c[:,0]))

            error[i,j] = test_sum - control_sum           
    return error

d = meanperror(dat,5)
d
