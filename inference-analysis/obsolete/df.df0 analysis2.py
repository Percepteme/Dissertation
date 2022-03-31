#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 14:18:00 2020
Run this script to explore data from a Results.csv generated from ImageJ Multi Measure or Time Series Analyzer intensity vs time for multiple ROIs 
@author: am380
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import pandas as pd
from cellpose import models, io, plot
import os

#### Import data
datdir = "/home/caiusgibeily/Downloads/Training-Optocamp/test/"
os.chdir(datdir)

dat_num = 0
dat = os.path.join(str(dat_num).zfill(3) + "_intensities.csv")

df = np.array(pd.read_csv(dat,header=None))

# variable to store the camera integration time (in seconds) as this always gets lost in the mail...
IntTime = 0.1

# Convert frame number, stored in the first column, to time, by multiplying by IntTime. Start at time = 0 by subtracting IntTime.
time = np.multiply(df[:,0],IntTime)
data = df[:,1:]

#####################################################################
######111111111111####### DF/D0 analysis ############################
#####################################################################
# Specify which columns hold data (note this can vary depending on exactly the parameters of the analysis; this dataset has an additional column with labels which should be ignored for plotting)
# Plot the intensity vs time data for all ROIs
plt.plot(time,data)
plt.xlabel("Time (s)")
plt.ylabel("Mean intensity")
plt.title("Raw data")
#plt.savefig('raw-OLED1.pdf')
plt.close()

# Next, process the data to be in the form delta F / F0
F0 = df[0,1:]
DF = np.subtract(data, F0)

# Plot delta F
plt.plot(time,DF)
plt.xlabel("Time (s)")
plt.ylabel("$\Delta$ F")
plt.title("$\Delta$ F (i.e. F-F_0)")
#plt.savefig('dF-OLED1.pdf')
plt.close()

# Normalise the data to F0
DFNormF0 = np.divide(DF,F0)

# Plot delta F / F0
plt.plot(time,DFNormF0)
plt.xlabel("Time (s)")
plt.ylabel("$\Delta F / F_0$")
plt.title("$\Delta F / F_0$")
plt.ylim(-0.5,2)
#plt.axvspan(9.0, 10.0, facecolor='dodgerblue')
#plt.savefig('dFNormF0-OLED1.pdf')
plt.close()

####################################################################
####################################################################
####################################################################

fig, axs = plt.subplots(int(len(DFNormF0[0])),figsize=(10,10))
fig.suptitle("Grid view of neurons ($\Delta F / F_0$)")
plt.xlabel('Time (s)')
### 1 column stack
for i in range(len(DFNormF0[0])):
    if i == len(DFNormF0[0])-1:
        axs[i].plot(time,DFNormF0[:,i])
    else:
        axs[i].plot(time,DFNormF0[:,i])
        axs[i].yaxis.set_visible(False)
        axs[i].xaxis.set_visible(False)
    
        
plt.ylim(-0.5, 2)


### nx2 columns
fig, axs = plt.subplots(int(len(DFNormF0[0])/2),2,figsize=(10,10))
for i in range(len(DFNormF0[0])):
    
    if i >= int((len(DFNormF0[0])/2)):
        axs[(i-int(len(DFNormF0[0])/2)),1].plot(time,DFNormF0[:,i])
        axs[(i-int(len(DFNormF0[0])/2)),1].yaxis.set_visible(False)
        axs[(i-int(len(DFNormF0[0])/2)),1].xaxis.set_visible(False)
    else:
        axs[i,0].plot(time,DFNormF0[:,i])
        axs[i,0].yaxis.set_visible(False)
        axs[i,0].xaxis.set_visible(False)
plt.ylim(-0.5, 2)

#####################################################################
#####################################################################
## Pearson's correlation - correlogram 

from scipy import stats
import seaborn as sb
import cv2
stats.pearsonr(DFNormF0[:,0], DFNormF0[:,1])

## transliterate array into Pandas dataframe for correlogram analysis
pdata = pd.DataFrame(DFNormF0)
cordata = round(pdata.corr(),2)

## Clustered correlogram
sb.clustermap(cordata,cmap="plasma") 
## Heatmap
sb.heatmap(cordata,cmap="plasma")

## Partial correlation, accounting for covariates
#pip install pingouin
import pingouin as pg
## Between two neurons, given a third:
# partial_corr(data,x,y,covar)

## Pairwise partial correlation, when all other variables are accounted for
part_correl = pdata.pcorr().round(3)

## Partial correlogram 
sb.heatmap(part_correl,cmap="plasma")

## Set idempotent adjacencies = 0 
partcorrelmatrix = np.fill_diagonal(np.array(part_correl), 0)

## alternative method of acquiring partial correlations
x = pg.pairwise_corr(pdata)

######## Adjacency graph functions #####
import networkx as nx

### Create graph plots ###
def show_graph_with_labels(adjacency_matrix,coordata,thresh,**kwargs): # takes an adjacency matrix, as above and a threshold coefficient
    amatrix = np.array(adjacency_matrix) # convert to np.array 
    if "fill" in kwargs and kwargs["fill"] == True:    
        np.fill_diagonal(amatrix,-0.001) 
    rows, cols = np.where(amatrix >= thresh)
    edges = zip(rows.tolist(), cols.tolist())
    G = nx.Graph()
    G.add_edges_from(edges)
    pos = {coordata[i,0]:tuple(coordata[i,1:]) for i in range(len(coordata))}
    nx.set_node_attributes(G, pos, 'coord')
    nx.draw(G, node_size=0, labels=None, with_labels=False,edge_color="red", pos=pos,style='--')
    ax = plt.gca()
    plt.axis([0, 1024, 0, 1024])
    ax.set_ylim(ax.get_ylim()[::-1])
    fig.patch.set_facecolor("black")
    plt.imshow(plt.imread(image))
    ### adapted from https://stackoverflow.com/questions/29572623/plot-networkx-graph-from-adjacency-matrix-in-csv-file

### fit graph plots to neuronal projection images using the ROI coordinate data
image = os.path.join(str(dat_num).zfill(3) + "_img.png") ### Load image based on number for data analysis from inference machine

def maproi_to_image(mode):
    fig = plt.figure(figsize=(10,10)) ### set figure layout
    fig.patch.set_facecolor("black")
    txt = open(os.path.join(str(dat_num).zfill(3) + "_img_cp_outlines.txt"),"r")
    
    ROIlist = [line.rstrip('\n') for line in txt]
    coordata = np.array([[0 for l in range(3)] for m in range((len(ROIlist)))])
    
    txt = open(os.path.join(str(dat_num).zfill(3) + "_img_cp_outlines.txt"),"r")
    for ROIs, line in enumerate(txt):   ### Adapted from inference machine
        xy = map(int, line.rstrip().split(","))
        X = list(xy)[::2]
        xy = map(int, line.rstrip().split(","))
        Y = list(xy)[1::2]
        
        mask = np.zeros((1024,1024),dtype=np.uint8) ## prepare 0 array with same dimensions as image
        roi_vertices = np.array([list(zip(X,Y))]) ## prepare coordinate data for vertices of ROI
        cv2.fillPoly(mask, roi_vertices,color=(1)) # fill masks 
        
        boolean = np.where(mask!=0) ## Boolean operation
        coordata[ROIs,0] = ROIs
        coordata[ROIs,1] = X[0]
        coordata[ROIs,2] = Y[0]    
        
        
        ax = plt.gca()
        plt.axis([0, 1024, 0, 1024])
        
        
        ### Plot masks onto loaded max_intensity projection image
        ax.set_ylim(ax.get_ylim()[::-1]) ## Invert axis to have origin in top left corner
        plt.axis("off") ## Hide axes
        plt.annotate(ROIs, (X[0],Y[0]),color="white",size=15) ## Add numerical annotation based on order of ROI unpacking from 
        ## txt file
        if mode == "graph":
            plt.plot(boolean[1],boolean[0],color="red") # plot X and Y masks
        elif mode == "mask":
            plt.plot(boolean[1],boolean[0]) # plot X and Y masks
        plt.plot(X,Y,color="black") ## plot ROI outlines for visibility
    
    #plt.imshow(plt.imread("000_img.png"))
    if mode == "graph":
        show_graph_with_labels(cordata,coordata=coordata,thresh=0.0,fill=True)
    elif mode == "mask":
        fig.patch.set_facecolor("black")
        plt.imshow(plt.imread(image))
    plt.show()
##################################
## Visualise with neurons

