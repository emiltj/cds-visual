# Assignment 1 - Visual Analytics

## Content of assignment

This folder contains the following:

| File | Description|
|--------|:-----------|
```basic_image_processing.py```| Script that splits all images within a folder (inpath) and outputs the splitted images, as well as a .csv file with metadata on the output into "outpath".

basic_image_processing.py arguments:
- --inpath (str - path to input folder.  Default = os.path.join("..", "data", "pokemon_1st_gen", "images"))
- --outpath (str - path to output .csv file. Default = os.path.join("..", "data", "pokemon_1st_gen", "image_quadrants") )

## Running my scripts - MAC/LINUX/WORKER02
Setup
```bash
git clone https://github.com/emiltj/cds-visual.git
cd cds-visual
bash ./create_vis_venv.sh
```
Running this assignment:
```bash
cd cds-visual/assignment_1
source ../cv101/bin/activate 
python basic_image_processing.py
```

## Running my scripts - WINDOWS
Setup
```bash
git clone https://github.com/emiltj/cds-visual.git
cd cds-visual
bash ./create_vis_venv_win.sh
```
Running this assignment:
```bash
cd cds-visual/assignment_1
source ../cv101/Scripts/activate 
python basic_image_processing.py
``` 

## Contact

Feel free to write me, Emil Jessen for any questions (also regarding the reviews). 
You can do so on [Slack](https://app.slack.com/client/T01908QBS9X/D01A1LFRDE0) or on [Facebook](https://www.facebook.com/emil.t.jessen/).
