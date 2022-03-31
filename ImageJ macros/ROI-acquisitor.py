
#Source: https://cellpose.readthedocs.io/en/latest/outputs.html [accessed March 2021]

####### ImageJ macro designed to map ROIs onto loaded image files using the cellpose cp_masks_outlines.txt file #### 
## See test data for examples of the output format ##
from ij import IJ
from ij.plugin.frame import RoiManager
from ij.gui import PolygonRoi
from ij.gui import Roi
from java.awt import FileDialog

fd = FileDialog(IJ.getInstance(), "Open", FileDialog.LOAD)
fd.show()
file_name = fd.getDirectory() + fd.getFile()
print(file_name)

RM = RoiManager() ## ROI manager object wrapper
rm = RM.getRoiManager() ## select ROI manager

imp = IJ.getImage() ## get imaged currently loaded in ImageJ

textfile = open(file_name, "r") ## open the txt file
for line in textfile:
    xy = map(int, line.rstrip().split(",")) # for each pair of lines in the text file, extract the coordinate data of the masks
    X = xy[::2] ## select every other even line as x-coordinates 
    Y = xy[1::2]
    imp.setRoi(PolygonRoi(X, Y, Roi.POLYGON))
    # IJ.run(imp, "Convex Hull", "")
    roi = imp.getRoi() 
    print(roi)
    rm.addRoi(roi) ## add ROI to the ROI manager
textfile.close()
rm.runCommand("Associate", "true")
rm.runCommand("Show All with labels") ## show with labels 
