#!/usr/bin/env python

# Import libraries
import os, glob, argparse, cv2
import pandas as pd

# Defining main function
def main(targetpath, filepath):
    # Empty lists, for appending to
    filenames = []
    distances_to_target = []
    
    # Getting the filename of the target image
    target_name = os.path.split(targetpath)[-1]

    # Load target image, calculate histogram and normalize
    target = cv2.imread(targetpath)
    target_hist = cv2.calcHist([target], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    target_hist_norm = cv2.normalize(target_hist, target_hist, 0,255, cv2.NORM_MINMAX)
    
    # Info for terminal
    print(f"[INFO] Calculating distances from corpus to \"{target_name}\" ...") # Info for user in terminal
    # For each of the non-target files, get filename and calculate distance to target
    for file in glob.glob(filepath):
        # Get filename and append to list
        filenames.append(os.path.split(file)[-1])

        # For each file, read image and calculate histogram distance
        img = cv2.imread(file)
        img_hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        img_hist_norm = cv2.normalize(img_hist, img_hist, 0,255, cv2.NORM_MINMAX)
        dist = round(cv2.compareHist(target_hist, img_hist_norm, cv2.HISTCMP_CHISQR), 2) # Using the normalization values from the target histogram - rounded to 2 decimal places
        distances_to_target.append(dist)
    
    print(f"\n[INFO] Distances between the 3D color histogram of \"{target_name}\" and the corpus in \"{filepath}\" have been calculated using the chi-square method.") # Info for user in terminal
    
    # Create a df with the information on distances
    df = pd.DataFrame(list(zip(filenames, distances_to_target)), 
                columns = ["filename", "distance"])
    
    # Find the row with the shortest chisquare distance to target image
    closest_image = df.loc[df['distance'].idxmax()]
    
    # Create outpath for df
    outpath = f"distances_to_{target_name[:-4]}.csv"    
    
    # Save df
    df.to_csv(outpath, index = False)
    print(f"\nA new file with the distances has been created succesfully: \"{outpath}\" \n NOTE: The image \"{closest_image[0]}\" has the shortest chi-square distance: {closest_image[1]}") # Info for user in terminal

# Defining behaviour when called from command line
if __name__=="__main__":
    # Initialize ArgumentParser class
    parser = argparse.ArgumentParser(description = "Calculates rgb-distance from image corpus to a specified target image")
    
    # Add inpath argument
    parser.add_argument(
        "-f",
        "--filepath", 
        type = str,
        default = os.path.join("..", "data", "flowers", "*.jpg"), # Default path to corpus, when none is specified
        required = False,
        help= "str - path to image corpus")
    
    # Add outpath argument
    parser.add_argument(
        "-t",
        "--targetpath",
        type = str, 
        default = os.path.join("..", "data", "flowers", "image_0001.jpg"), # Default path to a target image, when none is specified
        required = False,
        help = "str - path to target file from which to calculate distance to the other images")
    
    # Taking all the arguments we added to the parser and input into "args"
    args = parser.parse_args()
    
    # Perform main function
    main(args.targetpath, args.filepath)