#### Code for automating group allocation for preparing training dataset. Multi-level masks, where each mask is marked with a different pixel value
## requires manually labelled masks in different groups in ImageJ. This loops over all ROIs and assigns them to a unique group

for (i=0; i<roiManager("count"); i++){  
roiManager("Select", i);
RoiManager.setGroup(i);
RoiManager.setPosition(0);
roiManager("Set Line Width", 0)}
