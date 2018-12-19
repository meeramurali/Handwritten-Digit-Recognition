"""

Author: Meera Murali
Date: 5/6/2018
Course: CS545
Programming Assignment #1
A two-layered neural network (one hidden layer) to perform handwritten digit recognition using MNIST dataset

Experiment 3/3:   Varying no. of training examples

"""

import gzip
import numpy as np
import struct
from matplotlib import pyplot as plt
import time

# Parameters
LEARNING_RATE = 0.1
ALPHA = 0.9
n = 100

NUM_EPOCHS = 50

# MNIST data file paths to load from
TRAINING_IMAGES_FILE = 'samples//train-images-idx3-ubyte.gz'
TRAINING_LABELS_FILE = 'samples//train-labels-idx1-ubyte.gz'
TEST_IMAGES_FILE = 'samples//t10k-images-idx3-ubyte.gz'
TEST_LABELS_FILE = 'samples//t10k-labels-idx1-ubyte.gz'

MNIST_IMAGE = 2051
MNIST_LABEL = 2049

PLOT_NUM = 1


# Loads MNIST images data from ubyte file into a numpy array
# Arguments: filename
# Returns: A numpy array of dimensions (<no. of images> x 784)
def load_images(filename):
    with gzip.open(filename, 'rb') as f:
        magic, number, rows, cols = struct.unpack('>iiii', f.read(16))
        if magic != MNIST_IMAGE:
            raise ValueError("Error reading images")
        images_array = np.frombuffer(f.read(), dtype='uint8')
        images_array = images_array.reshape((number, rows * cols))
        images_array = images_array.astype(float)
        # Preprocessing: Scale the data values to be between 0 and 1 by dividing by 255
        images_array /= 255.
    return images_array


# Loads MNIST labels data from ubyte file into a numpy array
# Arguments: filename
# Returns: A numpy array of dimensions (<no. of images> x 784)
def load_labels(filename):
    """
    Read mnist labels from the ubyte file format
    :param filename: the path to read image labels
    :return: An array with labels as integers for each of the individual images
    """
    with gzip.open(filename, 'rb') as f:
        magic, _ = struct.unpack('>ii', f.read(8))
        if magic != MNIST_LABEL:
            raise ValueError("Error reading labels")
        array = np.frombuffer(f.read(), dtype='uint8')
    array = array.reshape(array.size, 1)
    return array


# Sigmoid activation function
# S(x) = 1 / 1 + e ^ (-x)
def sigmoid(x):
    return (1 / (1 + np.exp(-x)))


# Computes activation at each hidden neuron using the sigmoid activation function
# Arguments: array of input neurons (785, 1), array of weights between input and hidden layers (n, 785)
# Returns array of hidden neurons' activation values of dimension (n, 1)
def compute_hidden_neurons_activation(inputNeurons, hiddenWeights):
    # w.x
    w_dot_x = np.dot(hiddenWeights, inputNeurons)
    # activation = sigmoid (w.x)
    hiddenNeuronsActivation = sigmoid(w_dot_x)
    hiddenNeuronsActivation = hiddenNeuronsActivation.reshape(n,1)
    return hiddenNeuronsActivation


# Computes activation at each output neuron using the sigmoid activation function
# Arguments: array of hidden neurons incl. bias (n+1, 1), array of weights between hidden and output layers (10, n+1)
# Returns array of output neurons' activation values of dimension (10, 1) and predicted value
def compute_output_neurons_activation(hiddenNeurons, outputWeights):
    # w.h
    w_dot_h = np.dot(outputWeights, hiddenNeurons)
    # activation = sigmoid (w.h)
    outputNeuronsActivation = sigmoid(w_dot_h)
    # Predicted value = index of output neuron with maximum activation
    predictedData = np.argmax(outputNeuronsActivation, axis=0)
    return outputNeuronsActivation,predictedData


# Creates target array for a given label, i.e. sets index at label to 0.9, and rest to 0.1
def create_target_array(label):
    # initialize target array of size (10, 1) to 0.1 at each index
    target = np.full((10,1),0.1)
    # set index at label to 0.9
    target[label,0] = 0.9
    return target


# Compute error terms at output and hidden neurons - delta_k and delta_j
def calculateErrorTerms(hidden, output, target, weights):

    # compute delta_k - error term at each output neuron
    ones_k = np.ones((10,1))                                            # array of ones - same size as output
    one_minus_output = np.subtract(ones_k, output)                      # (1 - o)
    target_minus_output = np.subtract(target, output)                   # (t - o)
    delta_k = np.multiply(one_minus_output,target_minus_output)         # (1 - o)(t - o)
    delta_k = np.multiply(output, delta_k)                              # delta_k = o (1 - o)(t - o)

    # compute delta_j - error term at each hidden neuron
    ones_j = np.ones(((n + 1), 1))                                          # array of ones - same size as hidden
    one_minus_hidden = np.subtract(ones_j, hidden)                          # (1 - h)
    h_times_one_minus_hidden = np.multiply(hidden, one_minus_hidden)        # h (1 - h)
    sum_weight_dot_deltak = np.dot(np.transpose(weights),delta_k)           # Summation of (w.delta_k)
    delta_j = np.multiply(h_times_one_minus_hidden,sum_weight_dot_deltak)   # delta_j = h (1 - h) (Summation of (w.delta_k))
    delta_j = np.delete(delta_j, (n), axis=0)                               # Drop error term for hidden bias unit

    return delta_k,delta_j


# Update weights using error terms and momentum
def update_weights(delta_j, delta_k, weights_kj, weights_ji, hidden, input, delWeights_kj, delWeights_ji):

    momentumTerm_kj = ALPHA * delWeights_kj                             # momentum term = alpha * previous weight change
    deltak_dot_hidden = np.dot(delta_k, np.transpose(hidden))           # (delta_k.hidden)
    delWeights_kj = np.multiply(LEARNING_RATE, deltak_dot_hidden)       # (learning rate * (delta_k.hidden))
    delWeights_kj = np.add(delWeights_kj, momentumTerm_kj)              # delWeights_kj = (learning rate * (delta_k.hidden)) + momentum term
    weights_kj = np.add(weights_kj, delWeights_kj)                      # weights_kj = weights_kj + delWeights_kj

    momentumTerm_ji = ALPHA * delWeights_ji                             # momentum term = alpha * previous weight change
    deltaj_dot_input = np.dot(delta_j, np.transpose(input))             # (delta_j.input)
    delWeights_ji = np.multiply(LEARNING_RATE, deltaj_dot_input)        # (learning rate * (delta_j.input))
    delWeights_ji = np.add(delWeights_ji, momentumTerm_ji)              # delWeights_ji = (learning rate * (delta_j.input)) + momentum term
    weights_ji = np.add(weights_ji, delWeights_ji)                      # weights_ji = weights_ji + delWeights_ji

    return weights_kj, weights_ji, delWeights_kj, delWeights_ji         # return updated weights and weight changes


# Plot accuracies for training and test data sets on the same graph
def plot_accuracy(trainingAccuracy, testingAccuracy, num):
    global PLOT_NUM
    PLOT_NUM = PLOT_NUM + 1
    plt.figure(PLOT_NUM)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(trainingAccuracy, color = "red", label = 'training set accuracy')
    plt.plot(testingAccuracy, color = "green", label = 'testing set accuracy')
    plt.ylim(0, 100)
    plt.xlim(0, 50)
    plt.title('Experiment 3 - No. of training examples: ' + str(int(num)))
    lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1),ncol=2)
    plt.savefig('accuracy_E3_n_'+str(num)+'.png',bbox_extra_artists=(lgd,), bbox_inches='tight')






if __name__ == '__main__':
    starttime = time.time()

    # Load input images and labels for training and test datasets
    trainingImages = load_images(TRAINING_IMAGES_FILE)
    testImages = load_images(TEST_IMAGES_FILE)
    trainingLabels = load_labels(TRAINING_LABELS_FILE)
    testLabels = load_labels(TEST_LABELS_FILE)

    testImages = np.concatenate((testImages, np.ones((10000, 1))), axis=1)

    quarter_trainingSet = len(trainingImages)/4
    half_trainingSet = len(trainingImages)/2

    NUM_TRAINING_EGS = [half_trainingSet, quarter_trainingSet]

    # Perform experiment for no. of training examples = quarter of training egs, half of training egs
    for num in NUM_TRAINING_EGS:
        print("")
        print("_________________________________________________________")
        print("No. of training egs: " + str(num))
        print("_________________________________________________________")

        # Build balanced input dataset of specified size...
        # Initialize an array to track no. of images for each class in input array so far
        counter = [0] * 10
        original_size = len(trainingImages)
        max_each_class = num/10
        # Initialize and build image and label arrays for new size
        selected_trainingImages = np.empty((int(num), 784))
        selected_trainingLabels = np.empty((int(num), 1))
        i = 0
        for x in range (0, original_size):
            if counter[trainingLabels[x,0]] < max_each_class:
                imageToAdd = trainingImages[x].reshape((trainingImages[x].shape[0], 1))
                imageToAdd = np.transpose(imageToAdd)
                selected_trainingImages[i] = imageToAdd
                selected_trainingLabels[i] = trainingLabels[x]
                i += 1
                counter[trainingLabels[x,0]] += 1
            if np.sum(counter) == max_each_class * 10:
                break

        # Adding bias unit with value 1 to each image
        biasUnits = np.ones((int(max_each_class*10), 1))
        selected_trainingImages = np.concatenate((selected_trainingImages, biasUnits), axis=1)

        selected_trainingLabels = selected_trainingLabels.astype(int)

        # Initialize accuracy arrays
        training_Accuracy_Array = []
        test_Accuracy_Array = []

        # Initialize weights to small random values between -0.05 to 0.05
        weights_kj = np.random.uniform(-0.05, 0.05, size=(10, (n + 1)))
        weights_ji = np.random.uniform(-0.05, 0.05, size=(n, 785))

        # Initialize weights changes to zeros
        delWeights_kj = np.zeros((10, (n + 1)))
        delWeights_ji = np.zeros((n, 785))

        # Train for specified number of epochs
        for epoch in range(0, NUM_EPOCHS + 1):

            print("Epoch " + str(epoch) + ":")



            # Process training dataset...
            trainingHits = 0

            # Repeat for each image in training data set
            for x in range(0, len(selected_trainingImages)):

                # Set input as current image in testImages array
                training_Input= selected_trainingImages[x].reshape((selected_trainingImages[x].shape[0], 1))

                # Compute activation at each hidden neuron using sigmoid function
                training_Hidden = compute_hidden_neurons_activation(training_Input, weights_ji)

                # Add a bias unit with value 1 to the hidden layer
                training_Hidden = np.concatenate((training_Hidden, np.ones((1, 1))), axis=0)

                # Compute activation at each output neuron using sigmoid function using the hidden layer as input
                training_Output, training_Predicted = compute_output_neurons_activation(training_Hidden, weights_kj)

                # Create target array (10, 1) for current label, i.e. set index at label to 0.9, and rest to 0.1
                training_Target = create_target_array(selected_trainingLabels[x])

                # Perform back propagation from epoch 1 onwards
                if(epoch > 0):
                    # Calculate error terms at each hidden and output neuron
                    delta_kj, delta_ji = calculateErrorTerms(training_Hidden, training_Output, training_Target, weights_kj)

                    # Update weights using error terms and momentum
                    weights_kj, weights_ji, delWeights_kj, delWeights_ji = update_weights(delta_ji, delta_kj, weights_kj, weights_ji,
                                                                                          training_Hidden, training_Input,
                                                                                          delWeights_kj, delWeights_ji)
                # If predicted value matches target, increment no. of hits
                if(training_Target[training_Predicted] == 0.9):
                    trainingHits+=1

            # Compute accuracy on dataset as fraction of correct classification
            training_Accuracy = (trainingHits / len(selected_trainingImages)) * 100
            training_Accuracy_Array.append(training_Accuracy)
            print("Training Accuracy: ", training_Accuracy)



            # Process test dataset...
            testHits = 0

            # Reset confusion matrix
            confusionMatrix = np.zeros((10, 10))
            confusionMatrix = confusionMatrix.astype(int)

            # Repeat for each image in test data set
            for x in range(0, len(testImages)):

                # Set input as current image in testImages array
                test_Input = testImages[x].reshape((testImages[x].shape[0], 1))

                # Compute activation at each hidden neuron using sigmoid function
                test_Hidden = compute_hidden_neurons_activation(test_Input, weights_ji)

                # Add a bias unit with value 1 to the hidden layer
                test_Hidden = np.concatenate((test_Hidden, np.ones((1, 1))), axis=0)

                # Compute activation at each output neuron using sigmoid function using the hidden layer as input
                test_Output, test_Predicted = compute_output_neurons_activation(test_Hidden, weights_kj)

                # Create target array (10, 1) for current label, i.e. set index at label to 0.9, and rest to 0.1
                test_Target = create_target_array(testLabels[x])

                # If predicted value matches target, increment no. of hits
                if (test_Target[test_Predicted] == 0.9):
                    testHits += 1

                # Record prediction in confusion matrix
                confusionMatrix[testLabels[x], test_Predicted] += 1

            # Compute accuracy on dataset as fraction of correct classification
            test_Accuracy = (testHits / len(testImages)) * 100
            test_Accuracy_Array.append(test_Accuracy)
            print("Test Accuracy: ", test_Accuracy)
            print("")


        # Plot accuracies and confusion matrix
        plot_accuracy(training_Accuracy_Array, test_Accuracy_Array, num)
        print("Confusion matrix on test data after training: ")
        print(confusionMatrix)

    print("Run time: " + str((time.time() - starttime)/60) + " min")
