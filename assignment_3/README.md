# Assignment 3 - Visual Analytics

## Content of assignment

This folder contains the following:

| File | Description|
|--------|:-----------|
```edge_detection.py```| Calculates distance from a target image, to a set of images within a folder

edge_detection.py arguments:
- --inpath (str - path to target image.  Default = os.path.join("..", "data", "jefferson", "jefferson.jpg")
- --startpoint (tuple - tuple with start point of cropping (from top left))
- --endpoint (tuple - tuple with end point of cropping (bottom right))

## Running my scripts - MAC/LINUX/WORKER02
Setup
```bash
git clone https://github.com/emiltj/cds-visual.git
cd cds-visual
bash ./create_vis_venv.sh
```
Running this assignment:
```bash
cd cds-visual/assignment_3
source ../cv101/bin/activate 
python edge_detection.py
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
cd cds-visual/assignment_3
source ../cv101/Scripts/activate 
python edge_detection.py
``` 

## Contact

Feel free to write me, Emil Jessen for any questions (also regarding the reviews). 
You can do so on [Slack](https://app.slack.com/client/T01908QBS9X/D01A1LFRDE0) or on [Facebook](https://www.facebook.com/emil.t.jessen/).
