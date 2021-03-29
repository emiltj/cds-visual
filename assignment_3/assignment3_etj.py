#!/usr/bin/env python

# Importing libraries
import os
import sys
sys.path.append(os.path.join(".."))
import cv2
import numpy as np
from utils.imutils import jimshow
from utils.imutils import jimshow_channel
import matplotlib.pyplot as plt

# Define behaviour when called from command line
if __name__=="__main__":
    # Loading the image in
    fname = os.path.join("..", "data", "jefferson", "jefferson.jpg")
    image = cv2.imread(fname)

    # Recreating the image with a rectangle on top
    start_point = (1385, 885)
    end_point = (2890, 2790)
    color = (0, 255, 0)
    thickness = 2
    image_rectangle = cv2.rectangle(image.copy(), start_point, end_point, color, thickness)

    # Cropping the image
    image_cropped = image[885:2790, 1385:2890] # The first index indicating the range of x-values and the second the y-range

    # Writing the cropped image
    cv2.imwrite(os.path.join("..", "data", "jefferson", "image_cropped.jpg"), image_cropped)

    # Flattening the image to black and white
    grey_image = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2GRAY) # flatten image to black and white

    # Blurring the image
    blurred = cv2.GaussianBlur(grey_image, (5,5), 0)

    # Using the canny edge detection software
    canny = cv2.Canny(blurred, 75, 150)

    # Using the "findContours" cv2 function to find all the contours from the canny image
    contours, _ =cv2.findContours(canny.copy(), # .copy() is just so that we don't overwrite the original image, but rather do it on a copy
    cv2.RETR_EXTERNAL, # This takes only the external structures (we don't want edges within each coin)
    cv2.CHAIN_APPROX_SIMPLE, ) # The method of getting approximated contours

    # The original cropped image with contour overlay
    image_letters =cv2.drawContours(image_cropped.copy(), # draw contours on original
    contours, # our list of contours
    -1, #which contours to draw -1 means all. -> 1 would mean first contour 
    (0, 255, 0), # contour color
    2) # contour pixel width

    # Writing the cropped image with contour overlay
    cv2.imwrite(os.path.join("..", "data", "jefferson", "image_letters.jpg"), image_letters)
    print("Two new files have been created within the \"cds-visual/data/jefferson\" folder.")