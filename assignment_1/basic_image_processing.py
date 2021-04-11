#!/usr/bin/env python
import os, cv2, argparse
import pandas as pd
from pathlib import Path

# Function that splits img into 4 quadrants
def img_split_quadrants(img):
    
    # Retrieve dimensions of image
    height, width, n_channels = img.shape
    
    # Crop image into quadrants
    top_left = img[0:int(round(height/2)),0:int(round(width/2))]
    top_right = img[0:int(round(height/2)),int(round(width/2)):-1]
    bottom_left = img[int(round(height/2)):-1,0:int(round(width/2))]
    bottom_right = img[int(round(height/2)):-1,int(round(width/2)):-1]
    
    # Return the quadrants
    return (top_left, top_right, bottom_left, bottom_right)

def main(inpath, outpath):
    # Create empty lists
    quadrant_filenames = []
    quadrant_heights = []
    quadrant_widths = []
    quadrant_n_channels = []

    # Get list of all image filenames
    filenames = os.listdir(inpath)

    # For each image, in this list
    for filename in filenames:
        # Loading images
        filepath = os.path.join(inpath, filename)
        img = cv2.imread(filepath)

        # For each of the images
        # Split into 4 quadrants
        # Put into dict, with a key with the quadrant name
        quadrant_names = ["top_left", "top_right", "bottom_left", "bottom_right"]
        quadrants = dict(zip(quadrant_names, img_split_quadrants(img)))
        
        # For each of the 4 quadrants, save them and get info to append to empty df
        for quadrant_name, quadrant_img in quadrants.items():
            # Getting info on newly created quadrant
            height, width, n_channels = quadrant_img.shape
            
            # Create a new filename for the quadrant
            quadrant_filename = f"{filename}_{quadrant_name}.jpg"
            
            # Create an outpath
            if os.path.exists(outpath) == False:
                os.mkdir(outpath)
            quadrant_outpath = os.path.join(outpath, quadrant_filename)
            
            # Saving the quadrants
            cv2.imwrite(quadrant_outpath, quadrant_img)
            
            # Appending information on the quadrant to empty lists
            quadrant_filenames.append(quadrant_filename)
            quadrant_heights.append(height)
            quadrant_widths.append(width)
            quadrant_n_channels.append(n_channels)
    print(f"All images from \"{inpath}\" have been split and saved to \"{outpath}\" succesfully")

    # Create a dataframe containing relevant information on quadrants
    meta_data = pd.DataFrame(
        {'filename': quadrant_filenames, 
        'height': quadrant_heights, 
        'width': quadrant_widths})

    # Saving df
    meta_data_outpath = os.path.join(outpath, "meta_data.csv")
    meta_data.to_csv(meta_data_outpath)
    print(f"A new file has been created succesfully: \"{meta_data_outpath}\"")

# Define behaviour when called from command line
if __name__=="__main__":
    # Initialise ArgumentParser class
    parser = argparse.ArgumentParser(description = "Splits all images within a folder, and outputs the splitted images, as well as a .csv file with metadata on the output")
    
    # Add inpath argument
    parser.add_argument(
        "-i",
        "--inpath", 
        type = str,
        default = os.path.join("..", "data", "pokemon_1st_gen", "images"),
        required= False, 
        help= "str - path to data folder")
    
    # Add outpath argument
    parser.add_argument(
        "-o",
        "--outpath",
        type = str, 
        default = os.path.join("..", "data", "pokemon_1st_gen", "image_quadrants"),
        required = False, 
        help = "str - path to output folder")
    
    # Taking all the arguments we added to the parser and input into "args"
    args = parser.parse_args()

    # Perform main function
    main(args.inpath, args.outpath)