#!/usr/bin/env python

# Importing libraries
import sys, os, argparse
sys.path.append(os.path.join(".."))
import utils.classifier_utils as clf_util
import numpy as np 
import pandas as pd
from sklearn import metrics
from sklearn import datasets
from sklearn.datasets import fetch_openml
from utils.neuralnetwork import NeuralNetwork
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

def main(outfilename, save, individual, hiddenlayers, epochs):
    print("[INFO] Loading the MNIST dataset ...")
    
    ############### DOESN'T RUN CORRECTLY: ###############
    # Importing data; y = what the image depicts, X = values for all pixels (from top right, moving left)
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

    X = np.array(X)
    y = np.array(y)

    # Make a test-train split of some of the data
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        random_state=9, # just for replication purposes
                                                        train_size=7500, # absolute size of test and train set to avoid too much data
                                                        test_size=2500)
  
    # Min-max scaling (doing it after the split, to avoid any fitting of the training data from the testing data)
    scaler = MinMaxScaler()
    scaler = scaler.fit(X_train)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train))
    X_test_scaled = pd.DataFrame(scaler.transform(X_test)) 

    # Binarize the labels (getting from e.g. [3,1,2] to [[0,0,1],[1,0,0],[0,1,0], instead of course with numbers from 0-10) 
    y_train = LabelBinarizer().fit_transform(y_train) 
    y_test = LabelBinarizer().fit_transform(y_test)

    # Assigning more layers in the neural network:
    hiddenlayers.insert(0, X_train.shape[1]) # Inserting the number of features as the input layer
    hiddenlayers.append(10) # Inserting the number of possible outcomes as the output layer
    
    # Defining a model (with the specified number of nodes and layers)
    nn = NeuralNetwork(hiddenlayers)
    
    print("[INFO] Training the neural networks classifier ...")
    # Fit the model to the training data
    nn.fit(X_train, y_train, epochs = epochs)

    # Using the fitted model to predict the test data
    predictions = nn.predict(X_test)

    # The "predictions" object contains certainties that the given image contains a 0, 1, 2, etc. Instead we want a single prediction
    predictions = predictions.argmax(axis=1)
    
    print("[INFO] Evaluating the neural networks classifier ...")
    # Getting a classification report:
    classif_report = pd.DataFrame(classification_report(y_test.argmax(axis=1), predictions, output_dict = True))

    # Print to terminal
    print(classif_report)

    # Save as csv in "out"-folder, if save == True
    if save == True:
        outpath = os.path.join("..", "data", "mnist", "out", outfilename)
        classif_report.to_csv(outpath, index = False)
        print(f"\nThe classification benchmark report has been saved: \"{outpath}\"")

    # If an individual image path is specified, then predict the number this image is meant to represent.
    if individual != None:
        import cv2
        image = cv2.imread(individual)
        gray = cv2.bitwise_not(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        compressed = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
        compressed_flattened = [float(item) for sublist in compressed for item in sublist] # Converting a list of lists into a list 
        compressed_flattened = np.array(compressed_flattened) # Converting to array
        compressed_flattened = pd.DataFrame(scaler.transform([compressed_flattened])) # Scaling the features of the individual image
        individual_pred = nn.predict(compressed_flattened) # Predicting the individual image (output = 10 probabilities - one for each class (0:9))
        individual_pred = individual_pred.argmax(axis=1) # Getting the highest probability as the prediction
        print(f"\nImage prediction for \"{individual}\": \"{individual_pred}\"") # Printing into terminal, the prediction

# Define behaviour when called from command line
if __name__=="__main__":
    # Initialize ArgumentParser class
    parser = argparse.ArgumentParser(
        description = "Script that trains a neural networks classifier on a subset of the mnist dataset. Tests on another part of the mnist dataset and outputs classification report. Number and depth of hidden layers can be specified using the -hiddenlayers argument. The trained model can also be used to predict individual images, using the argument --individual.") 

    # Add argument specifying name of classification report
    parser.add_argument(
        "-o",
        "--outfilename", 
        type = str,
        default = "classif_report_neural_networks.csv", # Default when not specifying name of outputfile
        required = False, # Since we have a default value, it is not required to specify this argument
        help = "str - containing name of classification report")

    # Add argument specifying whether we want the classification report saved
    parser.add_argument(
        "-s",
        "--save", 
        type = bool,
        default = True, # Default when not specifying 
        required = False, # Since we have a default value, it is not required to specify this argument
        help = "bool - specifying whether to save classification report")

    # Add argument specifying an individual image that is wanted predicted
    parser.add_argument(
        "-i",
        "--individual", 
        type = str,
        default = os.path.join("..", "data", "cf_test", "test.png"), # Default when not specifying anything in the terminal
        required = False, # Since we have a default value, it is not required to specify this argument
        help = "str - specifying a .png file which is to be classified using this neural networks model.")

    # Add argument specifying the hidden layers
    parser.add_argument(
        "-H",
        "--hiddenlayers", 
        type = list,
        default = [2, 4], # Default when not specifying anything in the terminal
        required = False, # Since we have a default value, it is not required to specify this argument
        help = "list - specifying the hidden layers, each element in the list corresponds to number of nodes in layer. index in list corresponds to hiddenlayer number. E.g. [2, 4]")
    
    # Add argument specifying number of epochs
    parser.add_argument(
        "-e",
        "--epochs", 
        type = int,
        default = 5, # Default when not specifying anything in the terminal
        required = False, # Since we have a default value, it is not required to specify this argument
        help = "int - specifying number of epochs for training the model. Default = 5")
    
    # Taking all the arguments we added to the parser and input into "args"
    args = parser.parse_args()

    # Execute main function
    main(args.outfilename, args.save, args.individual, args.hiddenlayers, args.epochs)
