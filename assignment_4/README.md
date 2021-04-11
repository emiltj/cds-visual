# Assignment 4 - Visual Analytics

## Content of assignment

This folder contains the following:

| File | Description|
|--------|:-----------|
```lr-mnist.py```| 
```nn-mnist.py```| 

lr_mnist.py arguments:
- --outfilename (str - containing name of classification report)
- --save (bool - specifying whether to save classification report)
- --individual (str - specifying a .png file which is to be classified using this logistic regression model. For trying For trying it out, use: "../data/cf_test/test.png")

nn_mnist.py arguments:
- --inpath (str - path to target image.  Default = os.path.join("..", "data", "jefferson", "jefferson.jpg")
- --startpoint (tuple - tuple with start point of cropping (from top left))
- --endpoint (tuple - tuple with end point of cropping (bottom right))
- --hiddenlayers (list specifying the hidden layers, each element in the list corresponds to number of nodes in layer. index in list corresponds to hiddenlayer number. E.g. [2, 4])
- --epochs (int - specifying number of epochs for training the model. Default = 5)

## Running my scripts - MAC/LINUX/WORKER02
Setup
```bash
git clone https://github.com/emiltj/cds-visual.git
cd cds-visual
bash ./create_vis_venv.sh
```
Running this assignment:
```bash
cd cds-visual/assignment_4
source ../cv101/bin/activate 
python lr_mnist.py
python nn_mnist.py
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
cd cds-visual/assignment_4
source ../cv101/Scripts/activate 
python lr_mnist.py
python nn_mnist.py
``` 

## Contact

Feel free to write me, Emil Jessen for any questions (also regarding the reviews). 
You can do so on [Slack](https://app.slack.com/client/T01908QBS9X/D01A1LFRDE0) or on [Facebook](https://www.facebook.com/emil.t.jessen/).
