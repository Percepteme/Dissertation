#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 16:59:39 2022

@author: caiusgibeily
"""

############### CAITCHA analysis ######################

### import
import numpy as np
import pandas as pd
import os
import random as rd
###
datdir = "/home/caiusgibeily/Downloads/Training-Optocamp/test/CAITCHA/" 
os.chdir(datdir)
## read csv file

dat = pd.read_csv("intensities.csv",sep=",",header=0)
dat2 = pd.read_csv("test.csv",sep=",")
c = dat.loc[dat["img"] == 1]
dat = dat[1:,:]
    

### Mean absolute percentage error calculation ####
### 1. shuffle test arrays and subtract each value pairwise from the reference array
### 2. the absolute values are then divided by the reference value 
### 3. the values are summed and divided by the length of the reference array
### 4. the value is subtracted from 1 and multiplied by 100 to achieve an accuracy score
rd.seed(56)
def meanperror(arr,shuffle,part):
    array = np.array(arr)
    img = np.unique(np.array(array[1:,6]))
    error = np.zeros((len(img),part))
    for i in range(len(img)):
        c = np.array(dat.loc[arr["img"] == i+1])
        
        part_error = []
        for j in range(part):
            parr = c[:,j+1]
            shuffle_error = []
            control_error = []
            for k in range(shuffle):
                rd.shuffle(parr)
                ### test_dat ###
                abvalue = np.absolute(np.subtract(c[:,0],parr))
                diff = np.divide(abvalue,c[:,0],out=abvalue, where=c[:,0]!=0)
                summed = (np.sum(diff)/len(c[:,0]))*100
                shuffle_error.append(summed)
                ### control_dat ###
                control = np.copy(c[:,0])
                rd.shuffle(control)
                abvalue = np.absolute(np.subtract(c[:,0],control))
                diff = np.divide(abvalue,c[:,0],out=abvalue, where=c[:,0]!=0)
                summed = (np.sum(diff)/len(c[:,0]))*100
                control_error.append(summed)
            min_control_error = min(control_error)
            min_error = min(shuffle_error)
            print(min_error)
            print(min_control_error)
            part_error.append(abs(min_error-min_control_error))
        error[i] = part_error
    return error

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
            """
            for l in range(len(control)):
                if control_error[l] == 0 and ordered[l] == 0:
                    false_positive.append(1)
                else:
                    false_positive.append(control_error[l])
            """
            #print(abvalue)
            #setval = np.ones(abvalue.shape)
            setval = np.full(abvalue.shape,0.5)
            diff = np.divide(abvalue,c[:,0],out=setval, where=abvalue!=0)        
            test_sum = (np.sum(diff))/(len(c[:,0])) 
            ### control ###
            abvalue = np.absolute(np.subtract(c[:,0],np.array(control_error)))

            diff = np.divide(abvalue,c[:,0],out=abvalue, where=c[:,0]!=0)
            control_sum = (np.sum(diff)/len(c[:,0]))


            
            #part_error.append(test_sum-control_sum)
            error[i,j] = test_sum - control_sum           
        #error[i] = part_error
    return error

d = meanperror(dat,5)
e = meanperror(dat2,5)
f = meanperror(dat2,5)
