#!/usr/bin/env python

# Importing libraries
import os, sys, cv2, argparse
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.join(".."))
from utils.imutils import jimshow
from utils.imutils import jimshow_channel

def main(inpath, startpoint, endpoint):
    # Loading image
    image = cv2.imread(inpath)

    # Recreating the image with a rectangle on top
    color = (0, 255, 0) # green
    thickness = 2
    image_rectangle = cv2.rectangle(image.copy(), startpoint, endpoint, color, int(thickness))

    # Cropping the image
    image_cropped = image[startpoint[1]:endpoint[1], startpoint[0]:endpoint[0]] # The first index indicating the range of x-values and the second the y-range

    # Specifying cropped outpath
    cropped_outpath = os.path.join(os.path.split(inpath)[0],"image_cropped.jpg")
    cropped_outpath

    # Writing cropped image
    cv2.imwrite(cropped_outpath, image_cropped)
    print(f"A new file has been saved succesfully: \"{cropped_outpath}\"")

    # Flattening the image to black and white
    grey_image = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2GRAY)

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

    # Specifying cropped, contour overlay image outpath
    cropped_contour_outpath = os.path.join(os.path.split(inpath)[0], "image_letters.jpg")

    # Writing the cropped image with contour overlay
    cv2.imwrite(cropped_contour_outpath, image_letters)
    print(f"A new file has been saved succesfully: \"{cropped_contour_outpath}\"")

# Define behaviour when called from command line
if __name__=="__main__":
    # Initialise ArgumentParser class
    parser = argparse.ArgumentParser(description = "Takes a target image - creates a copy with a green rectangle specifying boundary of crop. Also creates a cropped version with contours of letters")
    
    # Add inpath argument
    parser.add_argument(
        "-i",
        "--inpath", 
        type = str,
        default = os.path.join("..", "data", "jefferson", "jefferson.jpg"),
        required = False,
        help= "str - path to image file to be processed")
    
    # Add outpath argument
    parser.add_argument(
        "-s",
        "--startpoint",
        type = tuple, 
        default = (1385, 885),
        required = False,
        help = "tuple - tuple of two numbers specifying bottom left pixel of crop")
        
    # Add outpath argument
    parser.add_argument(
        "-e",
        "--endpoint",
        type = tuple, 
        default = (2890, 2790),
        required = False,
        help = "tuple - tuple of two numbers specifying bottom left pixel of crop")
    
    # Taking all the arguments we added to the parser and input into "args"
    args = parser.parse_args()

    main(args.inpath, args.startpoint, args.endpoint)
    