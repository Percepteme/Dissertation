## Import packages
import numpy as np
import matplotlib.pyplot as plt
import skimage
import skimage.io
from cellpose import models, io
import glob, os
from PIL import Image
import cv2
from skimage import data
from cellpose import plot

## Current directory
os.getcwd()
####### Load pre-trained Optocamp model data
mdir = '/home/caiusgibeily/Downloads/Training-Optocamp/training/models/cellpose_residual_on_style_on_concatenation_off_training_2022_02_05_13_44_49.849970'
#model = models.CellposeModel(pretrained_model = mdir, model_typ gpu = True, diam_mean=30)
#mod = models.Cellpose(model_dir=mdir,gpu=True, model_type=None, torch = True)
model2= models.CellposeModel(pretrained_model=mdir, gpu=True, diam_mean=30)
####### Prepare max_intensity images here ######
## Load video directory
viddir = '/home/caiusgibeily/Downloads/Training-Optocamp/test'
os.chdir(viddir)
tifdir = os.path.join(viddir + "/" + "*.tif")

## Obtain raw output of videos in compatbile .tif format for processing
viddirs_raw = sorted(
    glob.glob(tifdir), key=os.path.getmtime)

## Process raw tif videos into processible file names ({int}_vid.tif)
for i, vid in enumerate(viddirs_raw, 0):
    try:
        files = []
        os.rename(vid, os.path.join(str(i).zfill(3) + '_' + 'vid.tif'))
    except OSError:
        print('Improper procedure. Please check videos or directory')
        
## Obtain list of processed video directories
viddirs = sorted(glob.glob(os.path.join(viddir + "/" + "*.tif")), key=os.path.getmtime) 

## Obtain list of videos
def get_vidlist(viddir, ext):
    os.chdir(viddir)
    files = []
    if ext == ".tif":
        for i in glob.glob("*_vid.tif"):
            files = np.append(files, i)
    return files
vidfiles = list(sorted(get_vidlist(viddir, ".tif")))

print("##############################\nVideo loading complete\n##############################\nInitiating cell inference")
## Loop over images to create max_intensity images for cellpose inference
for i in range(len(vidfiles)):
    IM = io.imread(viddirs[i])[0]
    #IM_MAX= np.max(IM, axis=0) 
    plt.imshow(io.imread('000_img.png'))
    im_output = os.path.join(viddir + '/' + str(i).zfill(3) + '_' + 'img.png')
    cv2.imwrite(im_output, IM)

print("##########################\nImage extraction complete\n##########################")
####### Load extracted max_intensity image files here ######
imdir = viddir
os.chdir(imdir)

## imdir = image directory ; ext = image extension used: .tif, .png and .jpg supported in function
## Compile list of files to process
def get_imglist(imdir, ext):
    files = []
    if ext == ".jpg":
        for i in glob.glob("*_img.jpg"):
            files = np.append(files, i)
    if ext == ".png":
        for i in glob.glob("*_img.png"):
            files = np.append(files,i)
    if ext == ".tif":
        for i in glob.glob("*_img.tif"):
            files = np.append(files,i)
    return files

imfiles = list(get_imglist(imdir, ".png"))
#imgs = [skimage.io.imread(f) for f in imfiles]

## Inspect the images
"""
img = io.imread(imfiles[0]) ## change image axis here
plt.figure(figsize=(10,10))
plt.imshow(img)
plt.axis('off')
plt.show()
"""
c=0
######## Cell position inference ########### 
## Channel data - only 1, so nucleus channel is set to 0
channels = [0,0]

## Run inference on all data in directory
diameter = 30 # recommended diameter
mask_threshold = 0.2
import cv2
for image in range(len(imfiles)):
    img = io.imread(imfiles[image])
    masks, flows, styles = model2.eval(img, diameter=diameter, 
                                             channels=channels,mask_threshold=mask_threshold)
    
    #io.masks_flows_to_seg(img, masks, flows, imdir, channels)
    io.save_to_png(img, masks, flows, imfiles[image])
    
    ## Save pipeline image (original > predicted outlines > predicted masks > predicted cell pose)
    fig = plt.figure(figsize=(12,5))
    plot.show_segmentation(fig, img, masks, flows[0], channels=channels)
    plt.tight_layout()
    plt.savefig(os.path.join("pipeline_" + imfiles[image]))
    plt.show()

print("Cell inference complete \nCheck segmentation png(s) \nin directory\n############################")
######### Data extraction ##############

## Initialise the movies 
vids = skimage.io.ImageCollection(vidfiles[0], conserve_memory=True)
# len(vids)
#fps = 10
#vids_s = np.arange(0,len(vids))/fps


## Initialise ROIs from cellpose text file
ROIdir = viddir
os.chdir(ROIdir)
ROIs = 0
### Collect ROIs ####
img_name = os.path.splitext(imfiles[0])[0]
txt = open(os.path.join(img_name + "_cp_outlines.txt"),"r")
for line in txt:   
    xy = map(int, line.rstrip().split(","))
    X = list(xy)[::2]
    X_points = np.array(X)
    xy = map(int, line.rstrip().split(","))
    Y = list(xy)[1::2]
    Y_points = np.array(Y) 
    ROIs += 1


print("ROI initialisation complete\n############################")

for video in range(len(vidfiles)):
    vids = skimage.io.ImageCollection(vidfiles[video], conserve_memory=True)
    output_tab = np.array([[0 for l in range(ROIs+1)] for m in range(len(vids))])
    x_axis = np.array(list(range(0,len(vids))))
    output_tab = np.column_stack((x_axis,output_tab))
    ROIs = 0 ## reinitialise ROI counter
    img_name = os.path.splitext(imfiles[video])[0]
    txt = open(os.path.join(img_name + "_cp_outlines.txt"),"r")
    for line in txt:   
        xy = map(int, line.rstrip().split(","))
        X = list(xy)[::2]
        X_points = np.array(X)
        xy = map(int, line.rstrip().split(","))
        Y = list(xy)[1::2]
        Y_points = np.array(Y) 
        ROIs += 1
        #ax=plt.gca()
        #plt.axis([0, 1024, 0, 1024])   
        #ax.set_ylim(ax.get_ylim()[::-1])
        #ax.xaxis.tick_top()     
        #plt.plot(X,Y)
        #print(Y)
    
        img_dimensions = vids[0].shape  ## input dimensions of videos
        mask = np.zeros(img_dimensions,dtype=np.uint8) ## prepare 0 array with same dimensions as image
        roi_vertices = np.array([list(zip(X,Y))]) ## prepare coordinate data for vertices of ROI
        ignore_mask_color = (256,)*1024 # prepare masks with arbitrary colour to fill
        cv2.fillPoly(mask, roi_vertices,color=(1)) # fill masks 
        
        img = io.imread(imfiles[video]).astype(np.uint8)
        masked_image=cv2.bitwise_and(mask,img) # create an output int array with 0s for areas outside the ROI 
        #print(masked_image)
        vid_input = io.imread(vidfiles[video])
        boolean = np.where(mask!=0,1,0).astype(np.uint8) ## Boolean operation
        masked_video=np.multiply(mask,vid_input)
    
        for i in range(len(vid_input)):
            intensity = np.average(masked_video[i],weights=(masked_video[i] != 0))
            output_tab[i,ROIs] = intensity
        save = np.savetxt(os.path.join(viddir + '/' + str(video).zfill(3) + '_intensities.csv'), output_tab,delimiter=",")
    print("Video", video+1, "completed!\n############################")


