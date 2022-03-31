## Import packages
import numpy as np
import matplotlib.pyplot as plt
import skimage
import skimage.io
from cellpose import models, io, plot
import glob, os
import cv2
import argparse as ag
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

## change directory to video directory
os.chdir(viddir)
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
    else:
        try:
            os.rename(vid, os.path.join(os.path.splitext(os.path.basename(str(viddirs_raw[i])))[0]
                                        + "_" + str(i).zfill(3) + "_vid.tif"))
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
    if rename_videos == "False":
        imname = os.path.splitext(os.path.basename(str(viddirs_raw[i])))[0]
        if os.path.isfile("".join(glob.glob(os.path.join(imdir + "*_" + 
                                                         str(1).zfill(3) + "_img" + imformat))))==False:
            IM = io.imread(viddirs[i])
            IM_MAX= np.max(IM, axis=0)
            im_output = os.path.join(imdir + imname + "_" + str(i).zfill(3) + '_' + "img" + imformat)
            cv2.imwrite(im_output, IM_MAX) 
        else:
            pass
    else: 
        if os.path.isfile(os.path.join(imdir + str(i).zfill(3) + "_img" + imformat))==False:
            IM = io.imread(viddirs[i])
            IM_MAX= np.max(IM, axis=0)
            im_output = os.path.join(imdir + str(i).zfill(3) + '_' + "img" + imformat)
            cv2.imwrite(im_output, IM_MAX) 
        else:
            pass

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
######## Cell position inference ########### 
## Channel data - only 1, so nucleus channel is set to 0
channels = [0,0]
## Run inference on all data in directory
diameter = args.dia # recommended diameter
mask_threshold = args.mask_thresh
save_im = args.save_im
for image in range(len(imfiles)):
    im = os.path.splitext(imfiles[image])[0]
    imname = os.path.basename(str(viddirs_raw[image]))[0]

    if rename_videos == "False":
        if os.path.isfile(os.path.join(imdir + imname + im + "_cp_outlines.txt"))==False:
            img = io.imread(imfiles[image])
            masks, flows, styles = model.eval(img, diameter=diameter, 
                                                     channels=channels,mask_threshold=mask_threshold)
            
            #io.masks_flows_to_seg(img, masks, flows, imdir, channels)
            io.save_to_png(img, masks, flows, im)
            
            if imformat != ".png":
                png_img = cv2.imread(im + "_cp_masks.png")
                cv2.imwrite(os.path.join(imdir + imname + im + "_cp_masks" + imformat),png_img)
                #os.remove(im + ".png")
    
            ## Save pipeline image (original > predicted outlines > predicted masks > predicted cell pose)
            if save_im == "True":
                fig = plt.figure(figsize=(12,5))
                plot.show_segmentation(fig, img, masks, flows[0], channels=channels)
                plt.tight_layout()
                plt.savefig(os.path.join(im + imname + "_pipeline" + imformat))
            plt.close()
        else:
            pass
    
    if rename_videos == "True":
        if os.path.isfile("".join(glob.glob(os.path.join(imdir + "*_" + 
                                                         str(image).zfill(3) + "_img_cp_outlines.txt"))))==False:
            img = io.imread(imfiles[image])
            masks, flows, styles = model.eval(img, diameter=diameter, 
                                                     channels=channels,mask_threshold=mask_threshold)
            
            #io.masks_flows_to_seg(img, masks, flows, imdir, channels)
            io.save_to_png(img, masks, flows, im)
            
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

for video in range(len(vidfiles)):
    if os.path.isfile("".join(glob.glob(os.path.join(viddir + "*_" + 
                                                     str(video).zfill(3) + "_intensities.csv"))))==False:
        vids = skimage.io.ImageCollection(viddirs[video], conserve_memory=True)
        img_name = os.path.splitext(imfiles[video])[0]
        txt = open(os.path.join(img_name + "_cp_outlines.txt"),"r")
        with txt:
            ROIlist = [line.rstrip('\n') for line in txt]
            output_tab = np.array([[0 for l in range(len(ROIlist))] for m in range(len(vids))])
            x_axis = np.array(list(range(0,len(vids))))
            output_tab = np.column_stack((x_axis,output_tab))
            img_name = os.path.splitext(imfiles[video])[0]
            txt = open(os.path.join(img_name + "_cp_outlines.txt"),"r")
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
                
                img = io.imread(imfiles[video]).astype(np.uint8)
                masked_image=cv2.bitwise_and(mask,img) # create an output int array with 0s for areas outside the ROI 
    
                vid_input = io.imread(viddirs[video])
                boolean = np.where(mask!=0,1,0).astype(np.uint8) ## Boolean operation
                masked_video=np.multiply(mask,vid_input)
                for i in range(len(vid_input)):
                    intensity = np.average(masked_video[i],weights=(masked_video[i] != 0))
                    output_tab[i,ROIs+1] = intensity
                if rename_videos == "True":
                    save = np.savetxt(os.path.join(viddir + '/' + str(video).zfill(3) + '_intensities.csv'), output_tab,delimiter=",")
                elif rename_videos == "False":
                    save = np.savetxt(os.path.join(viddir + '/' + os.path.splitext(os.path.basename(str(viddirs_raw[video])))[0] + "_" + str(video).zfill(3) + '_intensities.csv'), output_tab,delimiter=",")
                    
           
            print("Video", video+1, "completed!\n##############################")
print("All videos analysed\n##############################")

