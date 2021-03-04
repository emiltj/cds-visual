#!/usr/bin/env python

# Import libraries
import os
import glob
import sys
import pandas as pd
import cv2
import numpy as np

# filepath
filepath = os.path.join("..", "data", "flowers", "*02.jpg")
# targetpath
targetpath = os.path.join("..", "data", "flowers", "image_0730.jpg")

# Define function
def rgb_dist(targetpath, filepath):
    
    file_name = []
    distance_to_image_0730 = []

    target_image = cv2.imread(targetpath)
    target_hist = cv2.calcHist([target_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    target_hist = cv2.normalize(target_hist, target_hist, 0,255, cv2.NORM_MINMAX)
    
    for file in glob.glob(filepath):
        filename = os.path.split(file)[1]
        
        file_name.append(filename)

        img = cv2.imread(file)

        hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

        hist = cv2.normalize(hist, hist, 0,255, cv2.NORM_MINMAX)

        dist_to_target = round(cv2.compareHist(target_hist, hist, cv2.HISTCMP_CHISQR), 2)

        distance_to_image_0730.append(dist_to_target)

    df = pd.DataFrame(list(zip(file_name, distance_to_image_0730)), 
                columns =['filename', 'dist_to_target'])

    target_image_filename = os.path.split(targetpath)[1][:-4]

    df.to_csv(f"chisqr_dist_to_{target_image_filename}.csv", index = False)

# Define behaviour when called from command line
if __name__=="__main__":
    rgb_dist(targetpath, filepath)
    print("A new file has now been created within the src folder: \"chisqr_dist_to_image_0730.csv\"")
