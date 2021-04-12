#!/usr/bin/env python

# data tools
import os, cv2, glob, argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# sklearn tools
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# tf tools
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (Conv2D, 
                                     MaxPooling2D, 
                                     Activation, 
                                     Flatten, 
                                     Dense)

# Define function for retrieving alphabetically sorted artist list:
def get_artists(artists_path):
    artists = os.listdir(artists_path)
    artists = artists[0:4] + artists[5:]
    artists = sorted(artists)
    return artists

# Define function for retrieving test/train data:
def get_train_test(artists):
    # Make empty lists, which are to be appended to
    train_paintings, train_paintings_artists = [], []
    test_paintings, test_paintings_artists = [], []

    # For every artist, generate a list of paintings. 
    # For every painting in list of paintings. 
    # Take the artist name and append it. Take the painting and append it
    # //Repeat for test

    for artist in artists:
        print(f"[INFO] Importing paintings from: {artist}")
        # Training
        for train_painting in glob.glob(os.path.join("..", "data", "paintings", "training", f"{artist}", "*.jpg")):
            train_paintings_artists.append(artist)
            train_paintings.append(cv2.imread(train_painting))
        # Testing
        for test_painting in glob.glob(os.path.join("..", "data", "paintings", "validation", f"{artist}", "*.jpg")):
            test_paintings_artists.append(artist)
            test_paintings.append(cv2.imread(test_painting))
    
    # Return the lists
    return train_paintings, train_paintings_artists, test_paintings, test_paintings_artists

# Define function for resizing and making into array
def get_resized_arrays(paintings, width, height):
    paintings_resized = []
    for painting in paintings:
        # Resize painting
        resized = cv2.resize(painting, (width, height), interpolation = cv2.INTER_AREA)
        # Normalize painting
        resized = resized.astype("float") / 255.
        # Append to list
        paintings_resized.append(resized)
    
    # Make into arrays with same dimensions instead of lists
    paintings_resized = np.array(paintings_resized).reshape(len(paintings_resized), width, height, 3)
    
    # Return
    return paintings_resized

# Define function for plotting accuracy/loss over epochs
def plot_history(H, epochs):
    # visualize performance
    plt.style.use("fivethirtyeight")
    plt.figure()
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("out", 'training_history.png'), format='png', dpi=100)
    plt.show()


def main(cnn, resizedim,  batch_size):
    # Make a alphabetically sorted list of all the artists
    artists_path = os.path.join("..", "data", "paintings", "training")
    artists = get_artists(artists_path)

    # Get the data we need
    train_paintings, train_paintings_artists, test_paintings, test_paintings_artists = get_train_test(artists)

    # Resize and array-tize images
    train_paintings_resized = get_resized_arrays(train_paintings, resizedim[0], resizedim[1])
    test_paintings_resized = get_resized_arrays(test_paintings, resizedim[0], resizedim[1])

    # Make ML names for what we have
    trainX = train_paintings_resized
    trainY = train_paintings_artists

    testX = test_paintings_resized
    testY = test_paintings_artists

    # Binarize labels
    lb = LabelBinarizer()
    trainY = lb.fit_transform(trainY)
    testY = lb.fit_transform(testY)

    # Initialize label names for CIFAR-10 dataset
    labelNames = artists # Here we know that the order is the same in "artists", so we know how to map the binarized labels onto the string names

    if cnn == "ShallowNet":
        architecture = "INPUT => CONV => ReLU => FC"
        # initialise model
        model = Sequential()

        # define CONV => RELU layer
        model.add(Conv2D(32, (3, 3),
                         padding="same", 
                         input_shape=(32, 32, 3)))
        model.add(Activation("relu"))

        # softmax classifier
        model.add(Flatten())
        model.add(Dense(10))
        model.add(Activation("softmax"))

    elif cnn == "LeNet":
        architecture = "INPUT => CONV => ReLU => MAXPOOL => CONV => ReLU => MAXPOOL => FC => ReLU => FC"
        # define model
        model = Sequential()

        # first set of CONV => RELU => POOL
        model.add(Conv2D(32, (3, 3), 
                         padding="same", 
                         input_shape=(32, 32, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), 
                               strides=(2, 2)))

        # second set of CONV => RELU => POOL
        model.add(Conv2D(50, (5, 5), 
                         padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), 
                               strides=(2, 2)))

        # FC => RELU
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        # softmax classifier
        model.add(Dense(10))
        model.add(Activation("softmax"))

    # Define step size for the gradient descent, as well as define the loss function
    opt = SGD(lr =.01)
    model.compile(loss="categorical_crossentropy",
                  optimizer=opt,
                  metrics=["accuracy"])

    # Fit the model
    H = model.fit(trainX, trainY, 
                  validation_data=(testX, testY), 
                  batch_size = batch_size,
                  epochs = 40,
                  verbose = 1)
    print(f"[INFO] Training of the model has been completed using batchsize: \n{batch_size} \n and the CNN architecture from {cnn}: \n {architecture}\n")

    # Get predictions:
    predictions = model.predict(testX, batch_size=batch_size)

    # Get classification report from predictions
    classif_report = pd.DataFrame(classification_report(testY.argmax(axis=1),
                                predictions.argmax(axis=1),
                                target_names=labelNames, output_dict = True))

    #Printing classification report
    print(classif_report)

    # Save and print classif_report
    classif_report.to_csv(os.path.join("out", 'classification_report.csv'), sep=',', index = True)
    print("A classification report has been saved succesfully: \"out/classification_report.csv\"")

    # Show plot and save it
    plot_history(H, 40)
    print("A plot history report has been saved succesfully: \"out/training_history.png\"")

    # Show plot and save it
    plot_model(model, to_file = os.path.join("out", 'model_plot.png'), show_shapes=True, show_layer_names=True)
    print("A model of the CNN has been saved succesfully: \"out/model_plot.png\"")


# Define behaviour when called from command line
if __name__=="__main__":
    # Initialise ArgumentParser class
    parser = argparse.ArgumentParser(description = "")
    
    # Add inpath argument
    parser.add_argument(
        "-c",
        "--cnn", 
        type = str,
        default = "ShallowNet",
        required = False,
        help= "")
    
    # Add outpath argument
    parser.add_argument(
        "-r",
        "--resizedim",
        type = list, 
        default = [32, 32],
        required = False,
        help = "")
        
    # Add outpath argument
    parser.add_argument(
        "-b",
        "--batch_size",
        type = int, 
        default = 200,
        required = False,
        help = "")
    
    # Taking all the arguments we added to the parser and input into "args"
    args = parser.parse_args()

    main(args.cnn, args.resizedim, args.batch_size)