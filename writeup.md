# **Finding Lane Lines on the Road** 

## Writeup

---

[//]: # (Image References)

[image1]: ./test_images/solidWhiteRight.jpg "solidWhiteRight"
[image2]: ./test_images_output/grayscaleWhiteRIght.jpg "grayscale"
[image3]: ./test_images_output/blurWhiteRight.jpg "gaussian blur"
[image4]: ./test_images_output/cannyWhiteRight.jpg "canny edge"
[image5]: ./test_images_output/roiWhiteRight.jpg "ROI"
[image6]: ./test_images_output/houghWhiteRight.jpg "hough line"
[image7]: ./test_images_output/resultWhiteRIght.jpg "result"


![alt text][image1]

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I used Gaussian blur to 
smooth the noise signal in each frame. 
![alt text][image3]
Next, I used Canny Edge to detect all possible edges in the image.
![alt text][image4]
The forth step is to use a polygon mask to the region of interesting (ROI), in where the lane lines should
![alt text][image5]
be found. The fifth (final) step is to use Hough Line to collect lines that is possbile to be lane lines.
In addition, inside the function of Hough Line, I also use draw_line to lines on the weighted_img.
![alt text][image6]

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by following steps:
1. Classify the left and right lines according to the sign of slope.
2. Filter out mis-detected line by their slope and offset. The formula of a line is `y=mx+b`. x and y are position, 
m is the slope of line and b is the offset from x-axis. The slope and offset of a correct lane line will lay on certain
range. Therefore, I can remove incorrect lines by the values of slope and offset.
3. Computer the average slope and offset for left and right lanes.
4. Use average slope and offset with the maximum y coordination to extrapolate the expected x coordination.
5. Use the coordination I get from extrapolation to draw the lane line.

Finally, I overlap the original image with the result of Hough line by weighted_img function. Here is the result: 

![alt text][image7]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when the camera is mounted differently inside the car, such as the
camera in challenge video. Therefore, the region of interest (ROI) may need to adjust depend on the position of camera. Also, 
the parameter of slope and offset need to adjust.

Another shortcoming could be the difference of environment outside the car. Different weather and the shadow of trees or 
buildings will affect the results of canny edge and Hough line processing.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to adjust ROI and the parameters of lane line manually or automatically. With some training
images, we can use regression or binary classify to get more accuracy parameter for ROI and lane lines.

Another potential improvement could be to preprocessing (adjust the brightness and contract) the image depend on the weather,
and change image from RGB to other domain to get rid of the shadow in the image. After that, we can do canny edge and Hough line.
 
