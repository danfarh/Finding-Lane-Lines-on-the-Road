# Finding Lane Lines on the Road

<img src="examples/laneLines_thirdPass.jpg" width="480"/>


## Finding Lane Pipeline

In this work, we'll try to detect lanes in test images and video using OpenCV Library.


To find Lane we'll use Canny to detect edges in a pre-process gray image, with these Edges we going to use Hough Transformation to find lines, which we'll use to draw lines in image. 


### To reach our objective we going through some steps:

* Create a Grayscale image
* Apply blur to avoid noise
* Find Edges Using Canny
* Define Vertices to create a Region of Interest
* Use Hough Transformation to find Lines
* Merge Lines and original image
* Create a pipeline function
* Test with test images


## 1. Reflection

My pipeline consisted of 3 parts:

### Part One

First, I converted the images to grayscale, then I applied a blur to avoid noise in the edges detection process. With the processed image I used Canny detection to find edges, this returned many irrelevant edges, so I defined vertices to make a region of interest. After that, I used Hough transformation to get lines from these edges. Finaly I drawed this lines in the original image. 

### Part Two

In the second part, I changed draw_lines testing each row to see if the slope was positive or negative, then I created 2 groups of lines based on your slopes. Then, I calculated start and ends points to make lines with a standar lenght. Finally, I used numpy to get mean and make  the function return only 2 lines with opposites slopes.

### Part Three

In the final part, I tested and tuned the parameters looking for the best fit in images and videos, the results were very satisfactory, but I couldn't perform well with the challenge video, I belive that more advanced and complex algorithims are needed to find lanes in that conditions. 


## 2. Potential shortcomings 


In a situation where many edges were detected, the function Draw_lines calculated the mean of these lines, and the result could be wrong, this happens and can be very visible in the challenge video. 


## 3. Improvements

I changed the Draw_lines function to consider only lines with the slope inside a determined range, which is the most likely value to slope, avoiding unnecessary line detection.

I did a fine-tuning of the parameters to find the optimum balance to the challenge video, and this returned a better result. 

## 3. Possible improvements

Use Color Selection together with the actual method to improve lane detection and improve the algorithm to avoid or fit curves. 

Make the algorithm consider a mean of n last lines to calculate the next line will smooth and help avoid errors.
