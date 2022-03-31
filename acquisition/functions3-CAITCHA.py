## Import packages
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage.transform import rescale
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
                    type=str, required=False, default="/home/caiusgibeily/Downloads/Training-Optocamp/training/models/cellpose_residual_on_style_on_concatenation_off_training_2022_03_26_13_48_54.341824", help="This is where one sets the directory to the pretrained model")

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
args = parser.parse_args()

## Current directory
os.getcwd()
####### Load pre-trained Optocamp model data
####### Prepare max_intensity images here ######
## Load video and image directory
#viddir = args.viddir
## image directory and format
#imdir = args.imdir
imformat = args.imformat
## change directory to video directory
#os.chdir(viddir)


## Extract videos from subdirectories if necessary

def subdir(viddir):
    subdirs = [name for name in os.listdir(".") if os.path.isdir(name)]  
    for i,num in enumerate(subdirs):
        vid = sorted(
            glob.glob(os.path.join(os.path.abspath(subdirs[i]+"/" + "*.tif"))), key=os.path.getmtime)
        os.rename(vid[0],os.path.join(viddir + os.path.basename(vid[0])))

def get_viddirs(viddir):
    tifdir = os.path.join(viddir + "/" + "*.tif")
## check for any existing images
#imdirs = os.path.join(imdir + "*_img" + imformat)
## Obtain raw output of videos in compatbile .tif format for processing
    viddirs_raw = sorted(
        glob.glob(tifdir), key=os.path.getmtime)
    vidfiles = [os.path.basename(str(viddirs_raw[i])) for i in range(len(viddirs_raw))]
    vidfiles = natsorted(vidfiles, key=lambda y: y.lower())
    return viddirs_raw, vidfiles
## should be empty if initialised on fresh video directory
#imdirs_raw = sorted(glob.glob(imdirs),
#                    key=os.path.getmtime)
## Process raw tif videos into processible file names ({int}_vid.tif)

def rename_videos(rename_videos,viddirs_raw,imdir,viddir):
    for i, vid in enumerate(viddirs_raw, 0):
        try:
            os.rename(vid, os.path.join(str(i).zfill(3) + "_vid.tif"))
        except OSError:
            print('Improper procedure. Please check videos or directory')
        if os.path.isfile("".join(glob.glob(os.path.join(viddir + "*" + 
                                                         str(i).zfill(3) + "_vid" + ".tif"))))==False:
            try:
                pass
            except OSError:
                print('Improper procedure. Please check videos or directory')
    viddirs = sorted(
        glob.glob(os.path.join(viddir + "/" + "*.tif")), key=os.path.getmtime) 
    vidfiles = [os.path.basename(str(viddirs[i])) for i in range(len(viddirs))]
    return viddirs, vidfiles


## Loop over images to create max_intensity images for cellpose inference
def create_max_intensity(viddirs, vidfiles,imdir):
    for i in range(len(vidfiles)):
        imname = os.path.splitext(os.path.basename(str(viddirs[i])))[0]
        if os.path.isfile("".join(glob.glob(os.path.join(imdir + "*" + 
                                                         str(i).zfill(3) + "_img" + imformat))))==False:
            IM = io.imread(viddirs[i])
            IM_MAX= np.max(IM, axis=0)
            
            #micro_dimensions = 500
            #scale_factor = 1024/micro_dimensions # pixel dimensions (1024x1024) over micrometer dimensions
            #img = rescale(IM_MAX, 1/scale_factor)
            
            im_output = os.path.join(imdir + str(i) + "_" + imname + "_" + str(i).zfill(3) + '_' + "img" + imformat)
            cv2.imwrite(im_output, IM_MAX) 


####### Load extracted max_intensity image files here ######



## imdir = image directory ; ext = image extension used: .tif, .png and .jpg supported in function
## Compile list of files to process
def get_imglist(imdir):
    os.chdir(imdir)
    files = []
    for i in glob.glob("*" + imformat):
        files = np.append(files,i)
    
    imfiles = list(files)
    imfiles = natsorted(imfiles, key=lambda y: y.lower())
    return imfiles

######## Cell position inference ########### 
## Channel data - only 1, so nucleus channel is set to 0

def inference(imfiles,imdir):    
    mdir = args.mdir
    imformat = ".png"
    model= models.CellposeModel(pretrained_model=mdir, gpu=False)
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

            #x_offset = coords[0] - (micro_dimensions/2)
            #y_offset = coords[1] - (micro_dimensions/2)
            
            #plt.imshow(img,extent=(x_offset,x_offset + micro_dimensions,y_offset,y_offset + micro_dimensions))
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
    

######### Data extraction ##############

## Initialise ROIs from cellpose text file
### Collect ROIs ####



### Data analysis functions
import networkx as nx
import seaborn as sb
import pingouin as pg
import skimage as ski
import scipy.signal as sci
from shapely.geometry import Point, Polygon
### Create graph plots ###

### fit graph plots to neuronal projection images using the ROI coordinate data
### Load image based on number for data analysis from inference machine

### Function - appends graph data to image and ROI coordinates, creates an annotated mask image from the data, shows 
### regions of stimulation (ROS) and checks whether ROIs fall within the region
## mode - graph: plots graph function, mask: plots mask functions, checkROS: when in mask mode, set to True to 
## enable photostimulation spot
def maproi_to_image(mode,dat_num,imfiles,**kwargs): 
    img_name = os.path.splitext(imfiles[video])[0]
    image = os.path.join(img_name + imformat) 
    fig = plt.figure(figsize=(10,10))
    fig.patch.set_facecolor("black")
    check = False

    txt = open("".join(glob.glob(os.path.join(imdir + "*" + str(0).zfill(3) + "_img_cp_outlines.txt"))),"r")
    ROIlist = [line.rstrip('\n') for line in txt]
    coordata = np.array([[0 for l in range(3)] for m in range((len(ROIlist)))])
    if "checkROS" in kwargs and kwargs["checkROS"] == True:
        #c = np.array(ski.draw.circle_perimeter(kwargs["xcoord"],kwargs["ycoord"],kwargs["radius"]))
        c = np.array(ski.draw.circle_perimeter(10,10,5))
        d = list(zip(c[0],c[1]))
        col = []
        insideROS = np.delete(coordata,1,1)
        ROS = Polygon()
        check = True 
    

    txt = open("".join(glob.glob(os.path.join(imdir + "*" + str(0).zfill(3) + "_img_cp_outlines.txt"))),"r")
    for ROIs, line in enumerate(txt):   ### Adapted from inference machine
        xy = map(int, line.rstrip().split(","))
        X = list(xy)[::2]
        xy = map(int, line.rstrip().split(","))
        Y = list(xy)[1::2]
        
        mask = np.zeros((1024,1024),dtype=np.uint8) ## prepare 0 array with same dimensions as image
        roi_vertices = np.array([list(zip(X,Y))]) ## prepare coordinate data for vertices of ROI
        cv2.fillPoly(mask, roi_vertices,color=(1)) # fill masks 
        ROS = Polygon(list(zip(X,Y)))
        boolean = np.where(mask!=0) ## Boolean operation
        
        micro_dimensions = 500
        scale_factor = 1024/micro_dimensions
        x_offset = coords[0] - (micro_dimensions/2)
        y_offset = coords[1] - (micro_dimensions/2)
        
        coordata[ROIs,0] = ROIs
        coordata[ROIs,1] = X[0]/1024 * micro_dimensions - x_offset
        coordata[ROIs,2] = Y[0]/1024 * micro_dimensions - y_offset   
        
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

IntTime = 0.1
def extract_intensities(viddirs,vidfiles,imfiles,imdir):
    for video in range(5):
        #vids = skimage.io.ImageCollection(viddirs[video], conserve_memory=True)

        img_name = os.path.splitext(imfiles[video])[0]
        vid = skimage.io.ImageCollection(vidfiles[video])
        img = io.imread(imfiles[video]).astype(np.uint16)
        #x_offset = 512/inference_coords[1,1]
        #y_offset = 512/inference_coords[1,2]
        txt = open("".join(glob.glob(os.path.join(imdir + "*" + str(video+1).zfill(3) + "b_cp_outlines.txt"))),"r")
        with txt:
            ROIlist = [line.rstrip('\n') for line in txt]
            output_tab = np.array([0 for l in range(len(ROIlist))])
            


            txt = open("".join(glob.glob(os.path.join(imdir + "*" + str(video+1).zfill(3) + "b_cp_outlines.txt"))),"r")
            for ROIs, line in enumerate(txt):   
                xy = map(int, line.rstrip().split(","))
                X = list(xy)[::2]

                xy = map(int, line.rstrip().split(","))
                Y = list(xy)[1::2]

            
                img_dimensions = np.zeros((1024,1024)).shape ## input dimensions of videos
                mask = np.zeros(img_dimensions,dtype=np.uint16) ## prepare 0 array with same dimensions as image
                roi_vertices = np.array([list(zip(X,Y))]) ## prepare coordinate data for vertices of ROI
                ignore_mask_color = (256,)*1024 # prepare masks with arbitrary colour to fill
                cv2.fillPoly(mask, roi_vertices,color=(1)) # fill masks 
                

                masked_image=cv2.bitwise_and(mask,img) # create an output int array with 0s for areas outside the ROI 
                boolean = np.where(mask!=0,1,0).astype(np.uint16) ## Boolean operation
                masked_video=np.multiply(mask,vid)
                for i in range(1):
                    intensity = np.average(masked_video,weights=(masked_video != 0)).astype(float)
                    output_tab[ROIs] = intensity
            save = np.savetxt(os.path.join(imdir + os.path.splitext(os.path.basename(str(imfiles[video])))[0] + "_" + str(video).zfill(3) + '_intensities.csv'), output_tab,delimiter=",")   
    return ROIs

def enget_intensities(viddirs,vidfiles,imfiles,imdir,coords,inference_coords):
    for video in range(len(vidfiles)):
        vids = skimage.io.ImageCollection(viddirs[video], conserve_memory=True)

        img_name = os.path.splitext(imfiles[video])[0]
        
        micropix = 0.63 # 0.63 um per pixel 
        x0,y0 = coords[0] - (512 * micropix), coords[1] - (512 * micropix) 

        x_adjust,y_adjust = coords[0]-inference_coords[0],coords[1]-inference_coords[1]
        txt = open("".join(glob.glob(os.path.join(imdir + "*" + str(0).zfill(3) + "_img_cp_outlines.txt"))),"r")
        with txt:
            ROIlist = [line.rstrip('\n') for line in txt]
            output_tab = np.array([[0 for l in range(len(ROIlist))] for m in range(len(vids))])
            x_axis = np.array(list(range(0,len(vids))))
            output_tab = np.column_stack((x_axis,output_tab))
            img_name = os.path.splitext(imfiles[video])[0]
            txt = open("".join(glob.glob(os.path.join(imdir + "*" + str(0).zfill(3) + "_img_cp_outlines.txt"))),"r")
            for ROIs, line in enumerate(txt):   
                xy = map(int, line.rstrip().split(","))
                #X = list(xy)[::2]
                X = list(map(lambda x: (x + x_adjust)/micropix, xy))[::2]
                xy = map(int, line.rstrip().split(","))
                #Y = list(xy)[1::2]
                Y = list(map(lambda x: (x + y_adjust)/micropix, xy))[::2]
                if np.array(X).any() > 1024 or np.array(X).any() < 0 or np.array(Y).any() > 1024 or np.array(Y).any() < 0:
                    X, Y = [0,0], [0,0] 
                img_dimensions = vids[0].shape  ## input dimensions of videos
                mask = np.zeros(img_dimensions,dtype=np.uint8) ## prepare 0 array with same dimensions as image
                roi_vertices = np.array([list(zip(X,Y))]) ## prepare coordinate data for vertices of ROI
                ignore_mask_color = (256,)*1024 # prepare masks with arbitrary colour to fill
                cv2.fillPoly(mask, roi_vertices,color=(1)) # fill masks 
                
                img = io.imread(imfiles[video]).astype(np.uint8)
                masked_image=cv2.bitwise_and(mask,img) # create an output int array with 0s for areas outside the ROI 
    
                vid_input = io.imread(viddirs[video])
                boolean = np.where(mask!=0,1,0).astype(np.uint8) ## Boolean operation
                masked_video=np.multiply(mask,vid_input)
                for i in range(len(vid_input)):
                    intensity = np.average(masked_video[i],weights=(masked_video[i] != 0))
                    output_tab[i,ROIs+1] = intensity
                save = np.savetxt(os.path.join(imdir + os.path.splitext(os.path.basename(str(imfiles[video])))[0] + "_" + 
                                               str(video).zfill(3) + "_" + str(ROIs).zfill(3) + "_intensities.csv"), output_tab,delimiter=",")   
    return ROIs



def extract_coords(viddirs,vidfiles,imfiles, coords,imdir):

    
    for video in range(len(vidfiles)):
        img_name = os.path.splitext(imfiles[video])[0]
        image = os.path.join(img_name + imformat) 
        if os.path.isfile("".join(glob.glob(os.path.join(imdir + "*" + 
                                                         str(video).zfill(3) + "_intensities.csv"))))==False:
            vids = skimage.io.ImageCollection(viddirs[video], conserve_memory=True)
            img_name = os.path.splitext(imfiles[video])[0]
            
            micropix = 0.63 # 0.63 um per pixel 
            x0,y0 = coords[0] - (512 * micropix), coords[1] - (512 * micropix) 
        
            txt = open("".join(glob.glob(os.path.join(imdir + "*" + str(0).zfill(3) + "_img_cp_outlines.txt"))),"r")
            ROIlist = [line.rstrip('\n') for line in txt]
            coordata = np.array([[0 for l in range(3)] for m in range((len(ROIlist)))])
            
            
            txt = open("".join(glob.glob(os.path.join(imdir + "*" + str(0).zfill(3) + "_img_cp_outlines.txt"))),"r")
            for ROIs, line in enumerate(txt):   ### Adapted from inference machine
                xy = map(int, line.rstrip().split(","))
                X = list(map(lambda x: x*micropix,xy))[0::2]
                xy = map(int, line.rstrip().split(","))
                Y = list(map(lambda x: x*micropix,xy))[1::2]
                
                #roi_vertices = np.array([list(zip(X,Y))]) ## prepare coordinate data for vertices of ROI
                 
                ROS = Polygon(list(zip(X,Y)))
                X_centroid,Y_centroid = list(ROS.centroid.coords)[0]
                micro_dimensions = 500
                scale_factor = 1024/micro_dimensions
                
                coordata[ROIs,0:3] = ROIs,X_centroid + x0,Y_centroid + y0
                save = np.savetxt(os.path.join(imdir + os.path.splitext(os.path.basename(str(imfiles[video])))[0] + "_" + str(video).zfill(3) + '_coords.csv'), coordata,delimiter=",")
    return coordata


def get_coords(viddir,coords,**kwargs):
    os.chdir(viddir)
    if "imdir" in kwargs:
        imdir = kwargs["imdir"]
    else:
        imdir = viddir
    if "subdir" in kwargs and kwargs["subdir"] == True:
        subdir(viddir)

    viddirs, vidfiles = get_viddirs(viddir)
    
    if "rename_videos" in kwargs and kwargs["rename_videos"] == True:
        rv = kwargs["rename_videos"]
        viddirs, vidfiles = rename_videos(rv,viddirs)
    
    create_max_intensity(viddirs,vidfiles,imdir)
    
    imfiles = get_imglist(imdir)
    
    inference(imfiles,imdir)
    
    coords = extract_coords(vidfiles,viddirs,imfiles,coords,imdir)
    #extract_intensities(vidfiles, viddirs, imfiles)
    return coords

def get_intensities(viddir,**kwargs):
    os.chdir(viddir)
    if "imdir" in kwargs:
        imdir = kwargs["imdir"]
    else:
        imdir = viddir
    
    if "subdir" in kwargs and kwargs["subdir"] == True:
        subdir(viddir)

    viddirs, vidfiles = get_viddirs(viddir)
    
    if "rename_videos" in kwargs and kwargs["rename_videos"] == True:
        rv = kwargs["rename_videos"]
        viddirs, vidfiles = rename_videos(rv,viddirs)
    
    create_max_intensity(viddirs,vidfiles,imdir)
    
    imfiles = get_imglist(imdir)
    
    inference(imfiles,imdir)
    
    #extract_coords(vidfiles,viddirs,imfiles)
    ROIs = extract_intensities(vidfiles, viddirs, imfiles,imdir)
    return ROIs
#get_coords(viddir = "/home/caiusgibeily/Downloads/Training-Optocamp/function-test",rename_videos=False)

def engram_intensities(viddir, imdir, coords,inference_coords):
    os.chdir(viddir)
    viddirs, vidfiles = get_viddirs(viddir)
    
    imfiles = get_imglist(imdir)
    enget_intensities(viddirs,vidfiles,imfiles,imdir,coords,inference_coords)


viddir = "/home/caiusgibeily/Downloads/Training-Optocamp/test/CAITCHA/data/machine15"
imdir = viddir
get_intensities(viddir)
c = get_coords(viddir, coords=[100,100])
