## Import packages
import numpy as np
import matplotlib.pyplot as plt
import skimage
import skimage.io
from cellpose import models, io, plot
import glob, os
import cv2
import argparse as ag
from natsort import natsorted, ns
import pandas as pd
### Use from the command line:
parser = ag.ArgumentParser(description="Using the Inference Machine. This script enables one to analyse output Optocamp videos (.tiff) by: (1) creating a max intensity projection for all available videos, (2) cell pose segmentation using the pretrained model (3) consecutive intensity extraction for each ROI per video, outputting a csv datafile. This script is unlimited in the number of videos it can process. One is advised to check inference for any errors and save corrected _cp_masks.png file for ROI extraction" )
parser.add_argument("--mdir",
                    type=str, required=False, default="/home/caiusgibeily/Downloads/Training-Optocamp/training/models/cellpose_residual_on_style_on_concatenation_off_training_2022_02_05_13_44_49.849970", help="This is where one sets the directory to the pretrained model")
parser.add_argument("--viddir",
                    type=str, required=False,default="/home/caiusgibeily/Downloads/Training-Optocamp/test/",help="This where the input directory to the videos is specified. WARNING - the script will rename files numerically")
parser.add_argument("--save_im",
                    required=False,help="Choose whether to save the pipeline images (these show original image, outlines, masks and predicted cell poses). Set True to save in imdir")
parser.add_argument("--imdir",
                    type=str,required=False, default = "/home/caiusgibeily/Downloads/Training-Optocamp/test/",help="This is where the output directory to the images is specified")
parser.add_argument("--imformat",
                    type=str,required=False,choices=[".png",".jpg",".tif"],default=".png",help="Choose the extension type of your images for segmentation. Default = .png")
parser.add_argument("--dia",
                    type=int,required=False, default=30,help="Select an average diameter value for cellpose inference. Default = 30")
parser.add_argument("--mask_thresh",
                    type=float,required=False,default=0.0,help="Select a mask threshold to alter sensitivity of mask detection. \nDefault = 0.0")
parser.add_argument("--rename_videos",
                    required=False,choices=["True","False"],default="False",help="Choose whether to rename entire stem of video to numeric system: 000, 001 etc. Default = False")
parser.add_argument("--subdir", 
                    required=False,choices=["True","False"],default="False",help="Choose whether the software should search the subdirectories in the video directory for video files")
parser.add_argument("--fos", 
                    required=False,choices=["True","False"],default="False",help="Choose whether the software should search the subdirectories in the video directory for video files")

args = parser.parse_args()

## Current directory
os.getcwd()
####### Load pre-trained Optocamp model data
mdir = args.mdir
model= models.CellposeModel(pretrained_model=mdir, gpu=False, diam_mean=30)
####### Prepare max_intensity images here ######
## Load video and image directory
viddir = args.viddir
## image directory and format
imdir = args.imdir
imformat = args.imformat
viddir = args.viddir
## change directory to video directory
os.chdir(viddir)
## Extract videos from subdirectories if necessary
subdir = args.subdir
if subdir == "True":
    subdirs = [name for name in os.listdir(".") if os.path.isdir(name)]  
    for i,num in enumerate(subdirs):
        vid = sorted(
            glob.glob(os.path.join(os.path.abspath(subdirs[i]+"/" + "*.tif"))), key=os.path.getmtime)
            
        os.rename(vid[0],os.path.join(viddir + subdirs[i] + ".tif"))
else:
    pass


tifdir = os.path.join(viddir + "/" + "*.tif")
## check for any existing images
#imdirs = os.path.join(imdir + "*_img" + imformat)
## Obtain raw output of videos in compatbile .tif format for processing
viddirs_raw = sorted(
    glob.glob(tifdir), key=os.path.getmtime)

## should be empty if initialised on fresh video directory
#imdirs_raw = sorted(glob.glob(imdirs),
#                    key=os.path.getmtime)
## Process raw tif videos into processible file names ({int}_vid.tif)
rename_videos = args.rename_videos
for i, vid in enumerate(viddirs_raw, 0):
    if rename_videos == "True":
        try:
            os.rename(vid, os.path.join(str(i).zfill(3) + "_vid.tif"))
        except OSError:
            print('Improper procedure. Please check videos or directory')
    elif rename_videos == "True" and os.path.isfile("".join(glob.glob(os.path.join(viddir + "*" + 
                                                     str(i).zfill(3) + "_vid" + ".tif"))))==False:
        try:
            pass
        except OSError:
            print('Improper procedure. Please check videos or directory')


## Obtain list of processed video directories
viddirs = sorted(
    glob.glob(os.path.join(viddir + "/" + "*.tif")), key=os.path.getmtime) 
## Obtain list of videos
vidfiles = [os.path.basename(str(viddirs[i])) for i in range(len(viddirs))]



print("##############################\nVideo loading complete\n##############################\nInitiating cell inference")
## Loop over images to create max_intensity images for cellpose inference

for i in range(len(vidfiles)):
    imname = os.path.splitext(os.path.basename(str(viddirs_raw[i])))[0]
    if os.path.isfile("".join(glob.glob(os.path.join(imdir + "*" + 
                                                     str(i).zfill(3) + "_img" + imformat))))==False:
        IM = skimage.io.imread(viddirs[i],plugin='pil')
        IM_MAX= np.max(IM, axis=0)
        if rename_videos == "False":
            im_output = os.path.join(imdir + str(i) + "_" + imname + "_" + str(i).zfill(3) + '_' + "img" + imformat)
            cv2.imwrite(im_output, IM_MAX) 
        else:
            im_output = os.path.join(imdir + str(i).zfill(3) + '_' + "img" + imformat)
            cv2.imwrite(im_output, IM_MAX) 
            
print("##############################\nImage extraction complete\n##############################")
####### Load extracted max_intensity image files here ######

os.chdir(imdir)

## imdir = image directory ; ext = image extension used: .tif, .png and .jpg supported in function
## Compile list of files to process
def get_imglist(imdir):
    files = []
    for i in glob.glob("*_img" + imformat):
        files = np.append(files,i)
    return files

imfiles = list(get_imglist(imdir))
imfiles = natsorted(imfiles, key=lambda y: y.lower())
######## Cell position inference ########### 
## Channel data - only 1, so nucleus channel is set to 0
channels = [0,0]
## Run inference on all data in directory
diameter = args.dia # recommended diameter
mask_threshold = args.mask_thresh
#save_im = args.save_im
save_im = args.save_im

for image in range(len(imfiles)):
    im = os.path.splitext(imfiles[image])[0]
    
    #imname = os.path.basename(os.path.splitext(str(viddirs_raw[image]))[0])

    if os.path.isfile("".join(glob.glob(os.path.join(imdir + "*" + 
                                                     str(image).zfill(3) + "_img_cp_outlines.txt"))))==False:
        img = io.imread(imfiles[image])
        masks, flows, styles = model.eval(img, diameter=diameter, 
                                                 channels=channels,mask_threshold=mask_threshold)
        
        #io.masks_flows_to_seg(img, masks, flows, imdir, channels)
        io.save_to_png(img, masks, flows, im + imformat)
        
        if imformat != ".png":
            png_img = cv2.imread(im + "_cp_masks.png")
            cv2.imwrite(os.path.join(imdir + im + "_cp_masks" + imformat),png_img)
            #os.remove(im + ".png")

        ## Save pipeline image (original > predicted outlines > predicted masks > predicted cell pose)
        if save_im == "True":
            fig = plt.figure(figsize=(12,5))
            plot.show_segmentation(fig, img, masks, flows[0], channels=channels)
            plt.tight_layout()
            plt.savefig(os.path.join(im + "_pipeline" + imformat))
        plt.close()
    else:
        pass

print("Cell inference complete \nCheck segmentation png(s) \nin directory\n##############################")
######### Data extraction ##############

## Initialise ROIs from cellpose text file
ROIdir = imdir
os.chdir(ROIdir)
### Collect ROIs ####

print("ROI initialisation complete\n##############################")

### Data analysis functions
import networkx as nx
import seaborn as sb
import pingouin as pg
import skimage as ski
import scipy.signal as sci
from shapely.geometry import Point, Polygon
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
    plt.imshow(plt.imread(imfiles[video]))
    plt.savefig(os.path.join(imdir + os.path.splitext(os.path.basename(str(imfiles[video])))[0] + '_graph' + '.pdf'))
    ### adapted from https://stackoverflow.com/questions/29572623/plot-networkx-graph-from-adjacency-matrix-in-csv-file

### fit graph plots to neuronal projection images using the ROI coordinate data
### Load image based on number for data analysis from inference machine

### Function - appends graph data to image and ROI coordinates, creates an annotated mask image from the data, shows 
### regions of stimulation (ROS) and checks whether ROIs fall within the region
## mode - graph: plots graph function, mask: plots mask functions, checkROS: when in mask mode, set to True to 
## enable photostimulation spot
def maproi_to_image(mode,dat_num,**kwargs): 
    img_name = os.path.splitext(imfiles[video])[0]
    image = os.path.join(img_name + imformat) 
    fig = plt.figure(figsize=(10,10))
    fig.patch.set_facecolor("black")
    check = False

    txt = open("".join(glob.glob(os.path.join(imdir + "*" + str(video).zfill(3) + "_img_cp_outlines.txt"))),"r")
    ROIlist = [line.rstrip('\n') for line in txt]
    coordata = np.array([[0 for l in range(3)] for m in range((len(ROIlist)))])
    if "checkROS" in kwargs and kwargs["checkROS"] == True:
        c = np.array(ski.draw.circle_perimeter(kwargs["xcoord"],kwargs["ycoord"],kwargs["radius"]))
        d = list(zip(c[0],c[1]))
        col = []
        insideROS = np.delete(coordata,1,1)
        ROS = Polygon(d)
        check = True 
    

    txt = open("".join(glob.glob(os.path.join(imdir + "*" + str(video).zfill(3) + "_img_cp_outlines.txt"))),"r")
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
        
        if check == True:
            #point = zip(coordata[ROIs,1],coordata[ROIs,2])
            test_coords = Point(coordata[ROIs,1],coordata[ROIs,2])
            insideROS[ROIs,0] = ROIs
            if ROS.intersects(test_coords)==True:
                col.append("green")
                insideROS[ROIs,1] = 1
            else:
                col.append("red")
                insideROS[ROIs,1] = 0
        ax = plt.gca()
        plt.axis([0, 1024, 0, 1024])
        
        
        ### Plot masks onto loaded max_intensity projection image
        ax.set_ylim(ax.get_ylim()[::-1]) ## Invert axis to have origin in top left corner
        #plt.axis("off") ## Hide axes
        plt.annotate(ROIs, (X[0],Y[0]),color="white",size=15) ## Add numerical annotation based on order of ROI unpacking from 
        ## txt file
        if mode == "graph":
            plt.plot(boolean[1],boolean[0], color="red") # plot X and Y masks
        elif mode == "mask":
            if check == True:
                circ = plt.Circle((kwargs["xcoord"],kwargs["ycoord"]),kwargs["radius"],color="green",fill=False,lw=5)
                ax.add_patch(circ)
                plt.plot(boolean[1],boolean[0],c=col[ROIs])
                
            else:
                plt.plot(boolean[1],boolean[0]) # plot X and Y masks
        plt.plot(X,Y,color="black") ## plot ROI outlines for visibility
    
    #plt.imshow(plt.imread("000_img.png"))
    if mode == "graph":
        show_graph_with_labels(kwargs["cordata"],coordata=coordata,thresh=0.0,fill=True)
    elif mode == "mask":
        #fig.patch.set_facecolor("black")
        plt.imshow(plt.imread(image))
        plt.savefig(os.path.join(imdir + os.path.splitext(os.path.basename(str(imfiles[video])))[0] + '_labelled-masks' + '.pdf'))
    plt.close()
    if check == True:
        return insideROS
    
##############################################
##############################################
### Raster plots #############################

## Takes all trace data and identifies maxima in each 1D array and returns an binary output. The plt.eventplot 
## then converts this into a rasterplot
## If showpeaks==True, the function will also output a spaghetti plot with annotated maxima points
## The function takes the input csv file as a numpy array, the height threshold at which peaks are expected and the
## expected width of peaks
"""def rasterplot(data,height,width,**kwargs):
    spikearray = np.empty(data.shape)
    for neuron, i in enumerate(data[0]):    
        spikedat = data[:,neuron]
        peaks, _ = sci.find_peaks(spikedat, height=height, width=width)  
        spikearray[peaks,neuron] = peaks*IntTime
        if "showpeaks" in kwargs and kwargs["showpeaks"] == True:
            plt.plot(spikedat)
            plt.plot(peaks, spikedat[peaks], "x")
    plt.show()
    plt.eventplot(np.transpose(spikearray),linelengths=0.7)  
    plt.xlim((0,vid))
    plt.yticks(np.arange(0,len(data[0]),2))
    plt.xlabel("Time (s)")
    plt.ylabel("Neuron ID")
    plt.title("Raster plot of spike trains")
    plt.savefig(os.path.join(imdir + os.path.splitext(os.path.basename(str(imfiles[video])))[0] + '_raster-plot_' + '.pdf'))
    plt.close()"""

def rasterplot(data,**kwargs):
    raster = np.empty(data.shape)
    index = np.zeros(len(data[0,:]))
    for i,neuron in enumerate(data[0,:]):    
        val = i
        thresh = np.max(data[:,val])*0.6
        x = sci.find_peaks(data[:,val],height = thresh,distance=15)
        raster[x[0],i] = x[0]*IntTime
        x100 = data[x[0],i]; val50 = data[(np.abs(data[list(range(0,int(x[0]+1))),i] - (x100*0.5))).argmin(),i]; index50 = list(data[:,i]).index(val50)
        
        index[i] = index50
        if "showpeaks" in kwargs and kwargs["showpeaks"] == True:
            plt.plot(x[0], DFNormF0[x[0],val], "ob"); plt.plot(DFNormF0[:,val])
            plt.plot(index[i],DFNormF0[int(index[i]),val],"ob"); plt.plot(DFNormF0[:,val])
            plt.show()
            
    plt.show()
    plt.eventplot(np.transpose(raster), linelengths=0.7)  
    plt.xlim((0.1,10))
    plt.yticks(np.arange(0,len(data[0]),2))
    plt.xlabel("Time (s)")
    plt.ylabel("Neuron ID")
    plt.title("Raster plot of spike trains")
    plt.savefig(os.path.join(imdir + os.path.splitext(os.path.basename(str(imfiles[video])))[0] + '_raster-plot_' + '.pdf'))
    plt.close()
    return raster
  
IntTime = 0.1

fos = args.fos
for video in range(len(vidfiles)):
    if os.path.isfile("".join(glob.glob(os.path.join(imdir + "*" + 
                                                     str(video).zfill(3) + "_intensities.csv"))))==False:
        vids = skimage.io.ImageCollection(viddirs[video], conserve_memory=True)
        img_name = os.path.splitext(imfiles[video])[0]
        txt = open("".join(glob.glob(os.path.join(imdir + "*" + str(video).zfill(3) + "_img_cp_outlines.txt"))),"r")
        with txt:
            ROIlist = [line.rstrip('\n') for line in txt]
            output_tab = np.array([[0 for l in range(len(ROIlist))] for m in range(len(vids))])
            x_axis = np.array(list(range(0,len(vids))))
            output_tab = np.column_stack((x_axis,output_tab))
            img_name = os.path.splitext(imfiles[video])[0]
            txt = open("".join(glob.glob(os.path.join(imdir + "*" + str(video).zfill(3) + "_img_cp_outlines.txt"))),"r")
            for ROIs, line in enumerate(txt):   
                xy = map(int, line.rstrip().split(","))
                X = list(xy)[::2]
                X_points = np.array(X)
                xy = map(int, line.rstrip().split(","))
                Y = list(xy)[1::2]
                Y_points = np.array(Y) 
            
                img_dimensions = vids[0].shape  ## input dimensions of videos
                mask = np.zeros(img_dimensions,dtype=np.uint8) ## prepare 0 array with same dimensions as image
                roi_vertices = np.array([list(zip(X,Y))]) ## prepare coordinate data for vertices of ROI
                ignore_mask_color = (256,)*1024 # prepare masks with arbitrary colour to fill
                cv2.fillPoly(mask, roi_vertices,color=(1)) # fill masks 
                
                img = skimage.io.imread(imfiles[video],plugin='pil').astype(np.uint8)
                masked_image=cv2.bitwise_and(mask,img) # create an output int array with 0s for areas outside the ROI 
    
                vid_input = skimage.io.imread(viddirs[video],plugin='pil')
                boolean = np.where(mask!=0,1,0).astype(np.uint8) ## Boolean operation
                masked_video=np.multiply(mask,vid_input)
                for i in range(len(vid_input)):
                    intensity = np.average(masked_video[i],weights=(masked_video[i] != 0))
                    output_tab[i,ROIs+1] = intensity
            if rename_videos == "True":
                save = np.savetxt(os.path.join(imdir + str(video).zfill(3) + '_intensities.csv'), output_tab,delimiter=",")
            elif rename_videos == "False":
                save = np.savetxt(os.path.join(imdir + os.path.splitext(os.path.basename(str(imfiles[video])))[0] + "_" + str(video).zfill(3) + '_intensities.csv'), output_tab,delimiter=",")   
            print("Video", video+1, "completed!\n##############################")
            
    ### DF/F0 plots
    
        vid = len(output_tab)*IntTime
        # Convert frame number, stored in the first column, to time, by multiplying by IntTime. Start at time = 0 by subtracting IntTime.
        time = np.multiply(output_tab[:,0],IntTime)
        data = output_tab[:,1:]
        
        F0 = output_tab[0,1:]
        DF = np.subtract(data, F0)
        DFNormF0 = np.divide(DF,F0)
        ## Graph plot and clustermap        
        
        
        
        plt.plot(time,DFNormF0)
        plt.xlabel("Time (s)")
        plt.ylabel("$\Delta F / F_0$")
        plt.title("$\Delta F / F_0$")
        plt.ylim(-0.5,2)
        plt.savefig(os.path.join(imdir + os.path.splitext(os.path.basename(str(imfiles[video])))[0] + '_dFNormF0' + '.pdf'))
        plt.close()
        
        ## Grid view ##
        fig, axs = plt.subplots(int(len(DFNormF0[0])/2),2,figsize=(10,10))
        """
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
        plt.savefig(os.path.join(imdir + os.path.splitext(os.path.basename(str(imfiles[video])))[0] + '_dFNormF0-grid' + '.pdf'))
        """
        plt.close()
        
        maproi_to_image(mode="mask",dat_num=video)
        
        if fos == "True":
            if output_tab.any() != None or np.max(output_tab) != 0: 
                pdata = pd.DataFrame(DFNormF0)
                cordata = round(pdata.corr(),2)
                sb.clustermap(cordata,cmap="plasma") 
                plt.savefig(os.path.join(imdir + os.path.splitext(os.path.basename(str(imfiles[video])))[0] + '_clustermap' + '.pdf'))
                plt.close()
                FOS = maproi_to_image(mode="mask",checkROS=True,xcoord=512,ycoord=512,radius=303, cordata=cordata,dat_num=video)
                save = np.savetxt(os.path.join(imdir + os.path.splitext(os.path.basename(str(imfiles[video])))[0] + "_" + str(video).zfill(3) + '_fos.csv'), FOS,delimiter=",")   
            else:
                pass
            
        
        rasterplot(DFNormF0,0,10,showpeaks=False)
        kymo = np.transpose(DFNormF0)
        fig, ax = plt.subplots()
        ax.imshow(kymo, cmap="plasma")
        plt.yticks(np.arange(len(DFNormF0[0]),step=4))
        plt.savefig(os.path.join(imdir + os.path.splitext(os.path.basename(str(imfiles[video])))[0] + '_kymograph' + '.pdf'))
        plt.close()

print("All videos analysed\n##############################")

