var tab = " \t";

macro "make spreadsheet of intensity through time [F4]" {
  // could add error cecking here for valid image and ROImanager window
  t = getTitle;
  run("Set Measurements...", "mean redirect=None decimal=2");
  output = newArray(nSlices);  for (s=0; s<nSlices; s++) output[s] = "";
  
  for (i=0; i<roiManager("count"); i++) {
    run("Clear Results");
    roiManager("select", i);
    for (s=1; s<=nSlices; s++) {    // measure each slice
      setSlice(s);
      run("Measure");
      output[s-1] = output[s-1] + d2s(getResult("Mean", nResults-1), 2) + tab;
    }  // for each slice
  }  // for each region of interest
  print("\\Clear");
  print(t);
  for (s=0; s<nSlices; s++) 
    print(output[s]);
  

}