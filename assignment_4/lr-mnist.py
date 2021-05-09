#!/usr/bin/env python

# Importing libraries
import sys, os, cv2, joblib, argparse
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

# Defining main function
def main(outname, save, individual, penalty, c):

    # Importing data; y = what the image depicts, X = values for all pixels (from top right, moving left)
    print("[INFO] Loading the MNIST dataset ...")
    X, y = fetch_openml('mnist_784', version = 1, return_X_y = True)

    # Converting to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Make a test-train split of some of the data
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        random_state = 9, # for replication purposes
                                                        train_size = 20000, # absolute size of test and train set to avoid too much data
                                                        test_size = 2000)

    # Min-max scaling:
    scaler = MinMaxScaler()
    scaler = scaler.fit(X_train) # Important to scale not only train data but also test data information from train
    X_train_scaled = pd.DataFrame(scaler.transform(X_train))
    X_test_scaled = pd.DataFrame(scaler.transform(X_test)) 

    # Fitting a model to the training data
    print("[INFO] Training a logistic regression classifier ...")
    clf = LogisticRegression(penalty = penalty, # Type of penalty, if any
                            tol = 0.1, # Tolerance for stopping
                            solver = 'saga', # Solver method
                            C = c, # C-parameter value (smaller value stronger regularization)
                            multi_class = 'multinomial').fit(X_train_scaled, y_train) # Multiclass classification

    # Predicting the test data, using the model fitted on the training data
    print("[INFO] Evaluating the logistic classifier ...") 
    y_pred = clf.predict(X_test_scaled)

    # Get classification report as a data frame
    classif_report = pd.DataFrame(metrics.classification_report(y_test, y_pred, output_dict = True))

    # Print to terminal
    print(classif_report)

    # Save as csv in "out"-folder, if save == True
    if save == True:
        # If the folder does not already exist, create it
        if not os.path.exists("out"):
            os.makedirs("out")
        
        # Create outpath and save
        outpath = os.path.join("out", outname)
        classif_report.to_csv(outpath, index = False)
        print(f"[INFO] The classification benchmark report has been saved: \"{outpath}\".")
        
        # Saving model
        outpath_lr_model = os.path.join("out", "lr_model.pkl")
        joblib.dump(clf, outpath_lr_model)
        print(f"[INFO] The trained logistic regression classifier model has been saved: \"{outpath_lr_model}\".")
        
    # If the argument for individual prediction is not "None", then predict the individual image
    if individual != None: 
                
        # Get the possible labels, as well as the number of possible labels
        classes = sorted(set(y))
        nclasses = len(classes)
        
        # Read individual image and convert to grayscale
        image = cv2.imread(individual) 
        gray = cv2.bitwise_not(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        
        # Compress image to fit the input size of the model 
        compressed = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Predict the individual image, using the previously trained model
        individual_pred = clf_util.predict_unseen_assignment4(compressed, clf, classes)
        print(f"[IMAGE PREDICTION] Image prediction for \"{individual}\": {individual_pred}")

# Define behaviour when called from command line
if __name__=="__main__":
    # Initialize ArgumentParser class
    parser = argparse.ArgumentParser(
        description = "Script that trains a linear regression classifier on a subset of the mnist dataset. Tests on another part of the mnist dataset and outputs classification report. Can also be used to predict individual images, using the argument --individual.") 

    # Add outname argument
    parser.add_argument(
        "-o",
        "--outname", 
        type = str,
        default = "classif_report_logistic_regression.csv", # Default when not specifying name of outputfile
        required = False, # Since we have a default value, it is not required to specify this argument
        help = "str - containing name of classification report.")

    # Add save argument
    parser.add_argument(
        "-s",
        "--save", 
        type = bool,
        default = True, # Default when not specifying anything else
        required = False, # Since we have a default value, it is not required to specify this argument
        help = "bool - specifying whether to save classification report.")

    # Add individual image prediction argument
    parser.add_argument(
        "-i",
        "--individual", 
        type = str,
        default = os.path.join("..", "data", "cf_test", "test.png"), # Default when not specifying anything in the terminal
        required = False, # Since we have a default value, it is not required to specify this argument
        help = "str - specifying a .png file which is to be classified using this logistic regression model.")
    
    # Add penalty argument
    parser.add_argument(
        "-p",
        "--penalty", 
        type = str,
        default = "none", # Default when not specifying anything in the terminal
        required = False, # Since we have a default value, it is not required to specify this argument
        help = "str - specifying penalty for the classifier - possible values: \"none\", \"l1\", \"l2\", \"elasticnet\"")
    
    # Add c-parameter argument
    parser.add_argument(
        "-c",
        "--c", 
        type = float,
        default = 1.0, # Default when not specifying anything in the terminal
        required = False, # Since we have a default value, it is not required to specify this argument
        help = "int - specifying c-parameter for the classifier. recommended values: 0.01, 0.1, 1.0, 10, 100, where lower values mean stronger regularization")
    
    # Taking all the arguments we added to the parser and input into "args"
    args = parser.parse_args()
    
    # Execute main function
    main(args.outname, args.save, args.individual, args.penalty, args.c)