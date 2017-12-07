## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a 
classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, 
to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and 
testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and 
create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./writeup/Load.jpg
[image2]: ./writeup/Hog_Channels.jpg
[image3]: ./writeup/Sliding_Window.jpg
[image4]: ./writeup/Hmap1.JPG
[image5]: ./writeup/Hmap2.JPG
[image6]: ./writeup/Hmap3.JPG
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
The python notebook is structured will all function definitions at the beginning, and implementation of the pipeline at 
the bottom. The main function called for the pipeline is `Vehicle_Finder()` in cell 26.


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

Cell 16: Loading Images
I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` 
and `non-vehicle` classes:

![Loading Images][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and 
`cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the 
`skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and 
`cells_per_block=(2, 2)` for both a vehicle and non-vehicle image:


![Hog Features][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and eventually settled on `cspace=YCrCb`,`orientations=9`, `pixels_per_cell=
(8, 8)` and `cells_per_block=(2, 2)`. Different color spaces were used, as well as a `pixels_per_cell=(32,32)` and 
`pixels_per_cell=(64,64)`. I was only able to crack the 99%+ accuracy mark of the SVC with the final stated parameter 
combination.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

[Cells 18 / 19]
This is where I built the features to train the classifier. 

[Cell 20]
This cell verifies that I have the sets the I think, by checking the output shape of the previous cell, and continues to 
scale the data appropriately before splitting into training and test sets with the `train_test_split()` function. After 
the split, the size of each set is then verified for clarity.

[Cell 21]
I trained a linear SVM using a combination of all available features (Hog, spatial, and color histogram). Each 
combination was tried, but the highest accuracy was attained with the combination of all (99.2%) 

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what
scales to search and how much to overlap windows?

[Cell 7] (Sorry it's jumping around)
The find cars function calculates the range to search in X and Y for the window to process data. The sliding window 
parameters were first set to focus on a large vehicle detection. This was something like a 32,32 sliding window. 
However, I found that I was detecting the vehicles properly but the bounding boxes were not as tight as I had hoped. It
was at this point that I started to dial back the size of the window in an attempt to get more individual hits 
(increasing the FA rate), but with the intention of relying more heavily on the heatmap later on to filter these out.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

[Cell 22]
Instead of hard-coding pixel coordinates to window, I decided to generalize the percentage of Y distance to start and 
stop the search on. This should scale better when varying resolutions of image data is fed into the pipeline. 

I also output the individually detected bounding boxes at this stage. Below is an export for an image with vehicles 
and an image without. This was done to validate that I was not detecting an unacceptable amount of false-positive boxes

![Sliding Window][image3]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

[Cell 31] `Vehicle_Finder()`
In order to curb the rate of false positives, a heatmap and thresholding was implemented. All bounding boxes over the
previous 10 cycles are combined. They are then fed into the `add_heat()` function [Cell ] that accumulates the number 
of updates within the given boundaries. These are then thresholded with the `apply_threshold()` function to only count
the pixels iff we have seen at least 5 updates in the past 10 samples. I found that if an object is there > 50% of the 
time, it tended to be a valid detection. 

### Here are some frames with their final labelled outputs next to the internal corresponding heatmaps:

![alt text][image4]
![alt text][image5]
![alt text][image6]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I did most of the original development, with the exception of the heatmaps and labelling during the actual learning
modules. Most this project was just repurposed functionality. The one place that frustratingly causes issues is in the 
training of the SVM. There is a window of accuracy that ranges from ~98% to 99%, which is directly dependent on the 
way the training set is distributed. I could force the split to always be identical, but this doesn't solve any issues
when it comes to playing new data so I left it as is.

The pipline doesn't severly fail in any point. It does sometimes have false positive detections, but they last only a 
limited number of frames.
