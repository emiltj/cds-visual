#!/usr/bin/env python
"""
Load image, grab and manipulate image region
Parameters:
    image: str <path-to-image>
Usage:
    getting_and_setting.py --image <path-to-image>
Example:
    $ python getting_and_setting.py --image ../img/trex.png
## Explain
- RGB and BGR triplets: BGR is the horse's ass in OpenCV." OpenCV reads in images in 
BGR format (instead of RGB) because when OpenCV was first being developed, 
BGR color format was popular among camera manufacturers and image software. 
- slicing arrays
## Task
Manipulate color of pixel region
Move pixel region
Loop over BGR triplet of pixel region
"""#!/usr/bin/env python
"""
Count total and unique words in directory
Parameters:
    path: str <path-to-folder>
Usage:
    word_counts_rdkm.py --path <path-to-folder>
Example:
    $ python word_counts_rdkm.py --path data/100_english_novels/corpus
"""

import os
from pathlib import Path
import argparse

# Define main function
def main():
    # Initialise ArgumentParser class
    ap = argparse.ArgumentParser()
    # CLI parameters
    ap.add_argument("-i", "--path", required=True, help="Path to data folder")
    ap.add_argument("-o", "--outfile", required=True, help="Output filename")
    # Parse arguments
    args = vars(ap.parse_args())

    # Output filename
    out_file_name = args["outfile"]
    # Create directory called out, if it doesn't exist
    if not os.path.exists("out"):
        os.mkdir("out")

    # Output filepath
    outfile = os.path.join("out", out_file_name)
    # Create column headers
    column_headers = "filename,height, width"
    # Write column headers to file
    with open(outfile, "a", encoding="utf-8") as headers:
        # add newling after string
        headers.write(column_headers + "\n")

    # Create explicit filepath variable
    filenames = Path(args["path"]).glob("*.jpg")
    # Iterate over images
    for image in filenames:
        # load image
        image = cv2.imread(image)
        (height, width, channels) = image.shape
        # Get novel name
        name = os.path.split(novel)[1]
        # Formatted string
        out_string = f"{name}, {height}, {width}"
        # Append to output file using with open()
        with open(outfile, "a", encoding="utf-8") as results:
            # add newling after string
            results.write(out_string+"\n")
        
# Define behaviour when called from command line
if __name__=="__main__":
    main()