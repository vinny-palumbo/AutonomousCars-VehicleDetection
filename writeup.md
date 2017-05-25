# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images
* Apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
* Train a SVM classifier
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/hog-features.png
[image2]: ./output_images/sliding-windows.png
[image3]: ./output_images/hot-windows-test1.jpg
[image4]: ./output_images/hot-windows-test3.jpg
[image5]: ./output_images/hot-windows-test4.png
[image6]: ./output_images/hot-windows-test5.png
[image7]: ./output_images/heat-map.png
[image8]: ./output_images/labels.png
[image9]: ./output_images/vehicle-detection-output.png

[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 11th code cell of the IPython notebook.

I explored different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images of cars in the dataset and displayed the HOG Image to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image1]

#### 2. Explain how you settled on your final choice of Feature Extraction parameters.

I tried various combinations of parameters and settled on my final choice of Binned Color, HOC and HOG parameters by looking at the speed of the feature extraction, by looking at the test accuracy of the classifier, and by looking at the pipeline output on test images.

Here is my final choice of Feature Extraction parameters, which gives the Classifier a Test Accuracy of 99.47%:

`
color_space='LUV'

spatial_size=(16, 16)

hist_bins=32

orient=8

pix_per_cell=8

cell_per_block=2

hog_channel=0
`

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a Support Vector Machine Classifier using the `sklearn` library, in the 26th code cell of the IPython Notebook. 

I used the `model_selection.GridSearchCV()` function to try different combinations of parameters and choose the best combination for the model:

`
parameters = {'kernel':('linear', 'rbf', 'sigmoid'), 'C':[1, 10, 15]}

svr = svm.SVC()

clf = model_selection.GridSearchCV(svr, parameters)

clf.fit(X_train, y_train)
`

The best parameters turned out to be `kernel='rbf'` and `C=10`. These gave the classifier a test accuracy of 99.47%

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I implemented a sliding window search in the 27th and 29th code cells of my IPython Notebook.

Instead of searching for cars in windows positioned all over the image, which can really slow down the vehicle detection pipeline, I took advantage of the fact that cars only appear in the lower half of the images. 

Moreover, to decrease the false-positive detections due to too-small windows and to increase the pipeline's speed, I took advantage of the effect of perspective on the size of a vehicle in relation to its distance from our car, and searched for windows of 3 different sizes simultaneously: searching smaller-sized windows near the middle of the image (where the cars are far) and searching larger-sized windows near the bottom of the image (where the cars are closer).

An overlap of 85% gave me the best compromise between detecting vehicles accurately and not slowing down the pipeline too much.

![alt text][image2]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 3 scales of windows using LUV 0-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image3] 
![alt text][image4]

![alt text][image5] 
![alt text][image6]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video_output.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The code for this step is contained in the 31th, 32nd, 34th and 35th code cells of the IPython notebook.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions and reject most areas affected by false positives.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the thresholded heatmap of each frame.  I then assumed each blob corresponded to each vehicle in the frame. I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a frame of the video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the frame of video:

##### Here are the hot windows of a frame of the video, and their corresponding thresholded heatmap:

![alt text][image7]

##### Here is the output of `scipy.ndimage.measurements.label()` on the heatmap of the frame:

![alt text][image8]

##### Here, the resulting bounding boxes are drawn onto the vehicles in the frame:

![alt text][image9]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust? 

My current pipeline would fail to detect vehicles from a live stream of images from the front camera of an autonomous car, because it is not fast enough. To speed it up, here are some things I could try:

I could limit it to search sliding windows only between the left-most and the right-most lane lines of the road. 

Moreover, I could reduce the overlap factor, and thus the number of sliding windows, by optimizing the size of the windows depending on their position in the image (effect of perspective). 

Instead of extracting features from each individual window as I search across the image, I could extract features just once for the entire region of interest (i.e. lower half of each frame of video) and subsample that array for each sliding window.

In addition, I could do a sliding window search on the entire lower-half of the image only when I lose a vehicle detection. But when I have a vehicle detected, I could do a sliding window search only near the position of the vehicles on the previous frame. 


To improve the accuracy and/or speed of the classifier, I could try using a Decision Tree or some other model.

To optimize my classifier, I could devise a train/test split that avoids having nearly identical images in both my training and test sets. This means extracting the time-series tracks from the GTI data and separating the images manually to make sure train and test images are sufficiently different from one another.


To make the detection windows less wobbly and unstable, and to reduce the false-positives, I could integrate the heatmaps of several previous frames before labeling the integrated heatmap.

 

