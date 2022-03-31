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

## Current directory
os.getcwd()
####### Load pre-trained Optocamp model data
mdir = '/home/caiusgibeily/Downloads/Training-Optocamp/training/models/optocamp-model.118985'
model = models.Cellpose(model_dir=mdir,gpu=False)

####### Prepare max_intensity images here ######

## Load video directory
viddir = '/home/caiusgibeily/Downloads/Training-Optocamp/test/test_run'
os.chdir(viddir)
tifdir = os.path.join(viddir + "/" + "*.tif")

## Obtain raw output of videos in compatbile .tif format for processing
vidfiles_raw = sorted(
    glob.glob(tifdir), key=os.path.getmtime)

## Process raw tif videos into processible file names ({int}_vid.tif)
for i, vid in enumerate(vidfiles_raw, 0):
    try:
        files = []
        os.rename(vid, os.path.join(str(i).zfill(3) + '_' + 'vid.tif'))
    except OSError:
        print('Improper procedure. Please check videos or directory')
        
## Obtain list of processed video directories
vidfiles = sorted(glob.glob('/home/caiusgibeily/Downloads/*.tif'), key=os.path.getmtime) 

## Obtain list of videos
def get_vidlist(viddir, ext):
    os.chdir(viddir)
    files = []
    if ext == ".tif":
        for i in glob.glob("*_vid.tif"):
            files = np.append(files, i)
    return files
vidfile = list(sorted(get_vidlist(viddir, ".tif")))

## Loop over images to create max_intensity images for cellpose inference
for i in range(len(vidfile)):
    IM = io.imread(vidfiles[i])[0]
    #IM_MAX= np.max(IM, axis=0) 
    plt.imshow(IM)
    im_output = os.path.join(viddir + '/' + str(i).zfill(3) + '_' + 'img.png')
    cv2.imwrite(im_output, IM)
    
####### Load extracted max_intensity image files here ######
imdir = '/home/caiusgibeily/Downloads/Training-Optocamp/test/test_run'
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
imgs = [skimage.io.imread(f) for f in imfiles]

## Inspect the images

img = io.imread(imfiles[0]) ## change image axis here
plt.figure(figsize=(10,10))
plt.imshow(img)
plt.axis('off')
plt.show()


######## Cell position inference ########### 
## Channel data - only 1, so nucleus channel is set to 0
channels = [0,0]

## Run inference on all data in directory
diameter = 20 # recommended diameter
mask_threshold = 0.0

for image in range(len(imfiles)):
    img = io.imread(imfiles[image])
    masks, flows, styles, diams = model.eval(img, diameter=diameter, 
                                             channels=channels,mask_threshold=mask_threshold,model_loaded=model)
    io.masks_flows_to_seg(img, masks, flows, diams, imdir, channels)
    io.save_to_png(img, masks, flows, imfiles[image])


## Plot resultant masks and images 
from cellpose import plot
fig = plt.figure(figsize=(12,5))
plot.show_segmentation(fig, img, masks, flows[0], channels=channels)
plt.tight_layout()
plt.show()

print(type(img[0]), img[0].dtype,img[0].shape) # properties of image

######### Data extraction ##############
## Initialise the movies 

#vidfiles = os.path.join(viddir,'*.tif')
vids = skimage.io.ImageCollection(vidfiles, conserve_memory=True)
# len(vids)
fps = 10
vids_s = np.arange(0,len(vids))/fps


## Initialise ROIs from cellpose text file
os.chdir('/home/caiusgibeily/Downloads/Training-Optocamp/test')
txt = open("00a_img1_cp_outlines.txt","r")
ROIs = 0
### Collect ROIs ####
for line in txt:   
    xy = map(int, line.rstrip().split(","))
    X = list(xy)[::2]
    X_points = np.array(X)
    xy = map(int, line.rstrip().split(","))
    Y = list(xy)[1::2]
    Y_points = np.array(Y) 
    ROIs += 1

output_tab = np.array([[0 for l in range(45)] for m in range(len(vid_input))])
x_axis = np.array(list(range(0,100)))
output_tab = np.column_stack((x_axis,output_tab))
plt.plot(x_axis,output_tab[:,])
txt = open("00a_img1_cp_outlines.txt","r")

for video in range(len(vidfile)):
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
        #roi_corners = np.array([[(10,10), (300,300), (10,300)]], dtype=np.int32)
        ignore_mask_color = (255,)*1024 # prepare masks with arbitrary colour to fill
        cv2.fillPoly(mask, roi_vertices,color=(200,200,200)) # fill masks 
        masked_image=cv2.bitwise_and(img1, mask) # create an output int array with 0s for areas outside the ROI 
        print(masked_image)
        vid_input = io.imread(vidfiles[video])
        boolean = np.where(masked_image==0,0,1) ## Boolean operation
        masked_video=np.multiply(boolean,vid_input)
    
        for i in range(len(vid_input)):
            intensity = np.average(masked_video[i],weights=(masked_video[i] != 0))
            output_tab[i,ROIs] = intensity
        save = np.savetxt(os.path.join(viddir + '/' + str(video).zfill(3) + '_intensities.csv'), output_tab,delimiter=",")


