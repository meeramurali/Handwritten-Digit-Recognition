# Handwritten-Digit-Recognition

This project involved the implementation of a Multilayer Perceptron (MLP) to recognize handwritten digits in the MNIST dataset (http://yann.lecun.com/exdb/mnist/). 
The neural network was trained on a dataset of 60,000 images for 50 epochs using backpropagation with stochastic gradient.
The experiment was repeated for varying number of hidden neurons, momentum value (alpha), and number of training examples.
Accuracies between 93-98% were obtained on the test set.

The network has 784 inputs (one for each pixel of the 28x28 images), one hidden layer with n hidden units, and 10 output units (one for each class of digits 0-9).
The sigmoid function is used to determine activation of each hidden /output neuron. 
The network is fully connected such that, every input unit connects to every hidden unit, and every hidden unit connects to every output unit. The input and hidden layers each have a bias unit, whose value is set to 1.

The data values are scaled to be between 0 and 1 by dividing by 255. And both sets of weights (between input and hidden, and between hidden and output) are initialized to small random positive and negative values between -0.05 to 0.05. The target value of the output unit is set to 0.9 if the input class is the corresponding index of the output neuron, and 0.1 otherwise. 
Back propagation with stochastic gradient is used to train the network, and the momentum term is included in the weight updates.
