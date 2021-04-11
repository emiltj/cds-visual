#!/usr/bin/env python

# Importing libraries
import sys,os, argparse
sys.path.append(os.path.join(".."))
import utils.classifier_utils as clf_util
import numpy as np 
import pandas as pd
from sklearn import metrics
from sklearn import datasets
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

def main(outfilename, save, individual):

    # Importing data; y = what the image depicts, X = values for all pixels (from top right, moving left)
    print("[INFO] loading MNIST (sample) dataset...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

    X = np.array(X)
    y = np.array(y)

    # Make a test-train split of some of the data
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        random_state=9, # just for replication purposes
                                                        train_size=7500, # absolute size of test and train set to avoid too much data
                                                        test_size=2500)

    # Min-max scaling:
    scaler = MinMaxScaler()
    scaler = scaler.fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train))
    X_test_scaled = pd.DataFrame(scaler.transform(X_test))

    # Fitting a model to the training data
    print("[INFO] training classifier...")
    clf = LogisticRegression(penalty='none', 
                            tol=0.1, 
                            solver='saga',
                            multi_class='multinomial').fit(X_train_scaled, y_train)

    # Predicting the test data, using the model fitted on the training data
    print(["[INFO] evaluating network..."])
    y_pred = clf.predict(X_test_scaled)

    # Get classification report
    cm = pd.DataFrame(metrics.classification_report(y_test, y_pred, output_dict = True))

    # Print to terminal
    print(cm)

    # Save as csv in "out"-folder, if save == True
    if save == True:
        outpath = os.path.join("..", "data", "mnist", "out", outfilename)
        cm.to_csv(outpath, index = False)
        print(f"\n \n The classification benchmark report has been saved: \"{outpath}\"")

    if individual != None: 
        print(["[INFO] Predicting the individual image..."])
        classes = sorted(set(y))
        nclasses = len(classes)
        import cv2
        image = cv2.imread(individual)
        gray = cv2.bitwise_not(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        compressed = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
        individual_pred = clf_util.predict_unseen_assignment4(compressed, clf, classes)
        print(f"\n \n Image prediction for {individual}: {individual_pred}")

# Define behaviour when called from command line
if __name__=="__main__":
    # Initialize parser
    parser = argparse.ArgumentParser(
        description = "Script that trains a linear regression classifier on a subset of the mnist dataset. Tests on another part of the mnist dataset and outputs classification report. Can also be used to predict individual images, using the argument --individual.") 

    # Add parser arguments
    parser.add_argument(
        "-o",
        "--outfilename", 
        type = str,
        default = "classif_report_logistic_regression.csv", # Default when not specifying name of outputfile
        required = False, # Since we have a default value, it is not required to specify this argument
        help = "str - containing name of classification report")

    parser.add_argument(
        "-s",
        "--save", 
        type = bool,
        default = True, # Default when not specifying 
        required = False, # Since we have a default value, it is not required to specify this argument
        help = "bool - specifying whether to save classification report")

    parser.add_argument(
        "-i",
        "--individual", 
        type = str,
        default = None, # Default when not specifying anything in the terminal
        required = False, # Since we have a default value, it is not required to specify this argument
        help = "str - specifying a .png file which is to be classified using this logistic regression model. \n For trying it out, use: \n \"../data/cf_test/test.png\"")
    
    args = parser.parse_args()

    main(args.outfilename, args.save, args.individual)