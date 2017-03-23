#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

"""
Helper Functions
Below are some helper functions to help get you started. They should look
familiar from the lesson!
"""
def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    #print('number of line: ', len(lines))
    left_para = []  # record (m, b)
    right_para = [] # same as left
    y_para = []     # record (y1, y2)
    # filter to get left and right lane parameter
    for line in lines:
        x1, y1, x2, y2 = line.flatten()
        #for x1,y1,x2,y2 in line:
        m = (y2-y1)/(x2-x1)  # find the slope of line
        b = y1 - m*x1
        m_abs = abs(m)
        if m >= 0.5 and m <= 0.87 and abs(b) <= 70:          # right lane
            #cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            right_para.append([m, b])
            y_para.append([y1, y2])
        elif m <= -0.5 and m >= -0.87 and abs(b-650) <= 100: # left lane
            #cv2.line(img, (x1, y1), (x2, y2), [0, 255, 0], thickness)
            left_para.append([m, b])
            y_para.append([y1, y2])
        else:
            #print('x1, y1, x2, y2', x1,y1,x2,y2)
            #print('slope, offset: ', m, b)
            #print("Posssible not a lane line")
            #cv2.line(img, (x1, y1), (x2, y2), [0, 0, 255], thickness)
            continue
    # initial parameters for slope, offset, ymax, ymin
    (ymax, xmax, _) = img.shape
    ymin = ymax//2
    left_m, left_b = -0.65, (650/960)*xmax
    right_m, right_b = 0.65, 25

    # compute average slope and offset
    if len(left_para):
        left_m, left_b = np.mean(left_para, axis=0).flatten()
    else:
        pass
        #print("no left_para in this frame!")

    if len(right_para):
        right_m, right_b = np.mean(right_para, axis=0).flatten()
    else:
        pass
        #print("no right_para in this frame!")
    # get maximum and minimum of y value
    if len(y_para):
        y1max, y2max = np.amax(y_para, axis=0).flatten()
        y1min, y2min = np.amin(y_para, axis=0).flatten()
        ymax = int(max(y1max, y2max))
        ymin = int(min(y1min, y2min))
    else:
        pass
        #print("no lane detect!")

    # use extrapolation to get expected x value and plot lane lines
    xmax = int((ymax - left_b)/left_m)
    xmin = int((ymin - left_b)/left_m)
    #print(type(xmax), type(xmin))
    #print(type(ymax), type(ymin))
    cv2.line(img, (xmin, ymin), (xmax, ymax), [0, 255, 0], thickness)

    xmax = int((ymax - right_b)/right_m)
    xmin = int((ymin - right_b)/right_m)
    cv2.line(img, (xmin, ymin), (xmax, ymax), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


"""
Image Processing Function of finding lane lines
"""
def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)

    # read image
    #init_img = mpimg.imread(image)
    #plt.imshow(image)
    #plt.show()
    init_img = image
    #print('Name of Image:', image)
    #plt.imshow(init_img)
    #plt.show()
    # transfer to gray scale
    gray = grayscale(init_img)

    # Use Gaussain_blur to smooth the noise
    kernel_size = 5
    gray_blur = gaussian_blur(gray, kernel_size)

    # Use Canny Edge to detect edge
    low_threshold = 50
    high_threshold = 150
    edges = canny(gray_blur, low_threshold, high_threshold)

    # Find the region of interest (ROI)
    imshape = gray.shape
    vertices = np.array([[(0, imshape[0]), (450, 330), (490, 330), (imshape[1], imshape[0])]], dtype=np.int32)
    mask_edges = region_of_interest(edges, vertices)
    #plt.imshow(mask_edges)
    #plt.show()

    # Use Hough Line to find the straight line
    rho = 2  # distance resolution in pixels of the Hough grid
    theta = np.pi/180   # angular resolutions in radians of the Hough grid
    threshold = 10      # minimum number of votes
    min_line_length = 35
    max_line_gap = 10
    line_image = hough_lines(mask_edges, rho, theta, threshold, min_line_length, max_line_gap)
    #plt.imshow(line_image)
    #plt.show()

    # Combine to original image
    result = weighted_img(line_image, init_img)
    #plt.imshow(result_img)
    #plt.show()

    return result


if __name__ == '__main__':
    import os
    images = os.listdir('./test_images/')

    for image in images:
        # read image
        init_img = mpimg.imread('./test_images/' + image)

        result_img = process_image(init_img)
        plt.imshow(result_img)
        plt.show()

