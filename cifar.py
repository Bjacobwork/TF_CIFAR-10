#
# Convolutional Neural Network for the CIFAR-10 dataset
#
# File:
#   cifar.py
# Version:
#   1.0
# Date:
#   June 3, 2017
# Authors:
#   Jacob Boober
#   Hvass Laboratories
#   Alex Krizhevsky
#

# Note:
#   The network uses the leaky_relu activation function. You can change this in either the build_conv and build_fc
# functions or you can change this in the construction of the network.

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import pickle


def unpickle(file):
    # Modification of unpickle function from:
    #   Alex Krizhevsky
    #   https://www.cs.toronto.edu/~kriz/cifar.html
    # Purpose:
    #   Retrieving and unplickling cifar-10 dataset
    # Precondition:
    #   data_batch_# located at "cifar-10-batches-py/"
    #   "file" is a number between one and five
    # Postcondition:
    #   dictionary filled with data from file
    with open('cifar-10-batches-py/data_batch_{0}'.format(file), 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict



# Variables for batch management for
#   get_batch function
hold_images = []
hold_labels = []
hold_left = -1
hold_file = -1
def get_train_batch(quantity):
    # Purpose:
    #   Obtain a subset of the data from the dataset
    # Precondition:
    #   global variables hold_images, hold_labels, hold_left, hold_file
    #   "quantity" is the number of entries requested for subset
    # Postcondition:
    #   Two numpy arrays are returned holding "quantity" entries
    #       "images" holding cifar-10 images
    #       "labels" holding the target classes for the images
    global hold_images,hold_labels, hold_left, hold_file
    images = []
    labels = []
    while(quantity >= 0):
        if(hold_left < 0):
            hold_file += 1
            data_dict = unpickle((hold_file%4)+1)
            hold_images = data_dict["data"]
            hold_labels = data_dict['labels']
            hold_images = hold_images.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
            hold_left = len(hold_images)-1
        images.append(hold_images[hold_left])
        labels.append(hold_labels[hold_left])
        hold_left -= 1
        quantity -= 1
    images = np.array(images)
    labels = np.array(labels)
    return images,labels


test_images = []
test_labels = []
test_left = -1
def get_test_batch(quantity):
    # Purpose:
    #   Obtain a subset of the data from the dataset
    # Precondition:
    #   global variables hold_images, hold_labels, hold_left, hold_file
    #   "quantity" is the number of entries requested for subset
    # Postcondition:
    #   Two numpy arrays are returned holding "quantity" entries
    #       "images" holding cifar-10 images
    #       "labels" holding the target classes for the images
    global test_images,test_labels, test_left
    images = []
    labels = []
    while(quantity >= 0):
        if(test_left < 0):
            data_dict = unpickle(5)
            test_images = data_dict["data"]
            test_labels = data_dict['labels']
            test_images = test_images.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
            test_left = len(test_images)-1
        images.append(test_images[test_left])
        labels.append(test_labels[test_left])
        test_left -= 1
        quantity -= 1
    images = np.array(images)
    labels = np.array(labels)
    return images,labels

def plot_images(images, cls_true, cls_pred=None):
    # Copied and modified from Hvass Laboratories' Github for MNIST dataset
    #   https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/02_Convolutional_Neural_Network.ipynb
    # Purpose:
    #   Display 9 images and their labels
    # Precondition:
    #   "images" is a list/array of length 9 and contains image content
    #   "cls_true" is a list/array of length 9 and contains the target class index for the images
    #   "cls_pred" is a list/array of length 9 and contains the predicted class index for the images
    # Postcondition:
    #   images and their respective labels are displayed using matplotlib
    assert len(images) == len(cls_true) == 9

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.6, wspace=0.6)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i], cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(classes[cls_true[i]])
        else:
            xlabel = "True: {0}, Pred: {1}".format(classes[cls_true[i]], classes[cls_pred[i]])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

def new_weights(shape):
    # Copied from Hvass Laboratories' Github for MNIST dataset
    #   https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/02_Convolutional_Neural_Network.ipynb
    # Purpose:
    #   To create a new set of weights fitting a specified shape
    # Precondition:
    #   "shape" is the shape of the requested weights
    # Postcondition:
    #   Tensorflow variable of given "shape" with random variables for weights
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05), name="Weights")

def new_biases(length):
    # Copied from Hvass Laboratories' Github for MNIST dataset
    #   https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/02_Convolutional_Neural_Network.ipynb
    # Purpose:
    #   To create a new set of weights of size "length"
    # Precondition:
    #   "length" is the length of the requested biases
    # Postcondition:
    #   Tesnsorflow variable of given "length" with a constant of 0.05
    return tf.Variable(tf.constant(0.05, shape=[length]),name="Biases")

def leaky_relu(layer):
    # Purpose:
    #   Leaky Rectified Linear Unit for activation function
    # Precondtion:
    #   "layer" is the tensorflow layer to be passed throught the activation funtion
    # Postcondition:
    #   The layer post leaky relu is returned with aplha of 0.1
    return tf.maximum(0.1*layer,layer)

def build_conv(input,num_input_channels, filter_size, num_filters, use_pooling=True, leaky= True, name=None):
    # Copied and modified from Hvass Laboratories' Github for MNIST dataset
    #   https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/02_Convolutional_Neural_Network.ipynb
    # Purpose:
    #   Building a convolutional layer
    # Precondition:
    #   "input" is the input layer
    #   "num_imput_channels" is the number of channels in the input layer
    #   "filter_size" is the size of the filters to pass the layer through
    #   "num_filters" is the number of filters to use on the input layer
    #   "use_pooling" is true if using 2x2 pooling
    #   "leaky" is true if the activation function is leaky relu
    # Postcondition:
    #   Returns the output layer from the convolution and it's weights
    tf.name_scope("Convolution")
    shape = [filter_size,filter_size, num_input_channels,num_filters]
    weights = new_weights(shape=shape)
    biases = new_biases(num_filters)

    if name is None:
        layer = tf.nn.conv2d(input= input, filter= weights, strides=[1,1,1,1],
                             padding='SAME')
    else:
        layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1],
                             padding='SAME', name=name)

    layer += biases

    if use_pooling:
        if name is None:
            layer = tf.nn.max_pool(value=layer, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')
        else:
            layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool_{0}".format(name))

    if leaky:
        layer = leaky_relu(layer)
    else:
        layer = tf.nn.relu(layer)

    return layer, weights

def flatten_layer(layer):
    # Copied from Hvass Laboratories' Github for MNIST dataset
    #   https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/02_Convolutional_Neural_Network.ipynb
    # Purpose:
    #   To flatten a layer into a single vector
    # Precondition:
    #   "layer" is the layer to be flattened
    # Postcondition:
    #   The layer is returned as a single vector as well as the number of features
    tf.name_scope("Flatten Layer")
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features

def build_fc(input, num_inputs, num_outputs, use_relu= True, leaky= True, name=""):
    # Copied and modified from Hvass Laboratories' Github for MNIST dataset
    #   https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/02_Convolutional_Neural_Network.ipynb
    # Purpose:
    #   Building a fully connected onto the input layer
    # Precondition:
    #   "input" is the layer being fed into the fully connected layer
    #   "num_inputs" is the number of elements from the input layer
    #   "num_outputs" is the number of elements from the fully connected layer
    #   "use_relu" is true if using relu activation function
    #   "leaky" is true if the activation function is leaky relu
    tf.name_scope("Fully Connected")
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)
    layer = tf.matmul(input, weights, name=name) + biases

    if use_relu:
        if leaky:
            layer = leaky_relu(layer)
        else:
            layer = tf.nn.relu(layer)

    return layer

# Testing and training batch sizes
train_batch_size = 64
test_batch_size = 256

# Variable for tracking the total itterations
total_itterations = 0
def optimize(num_iterations):
    # Copied and modified from Hvass Laboratories' Github for MNIST dataset
    #   https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/02_Convolutional_Neural_Network.ipynb
    # Purpose:
    #   Optimizing the convolutional neural network
    # Precondition:
    #   "num_iterations" is the number of iterations used for training
    # Postcondition:
    #   The neural network trains for "num_iterations" iterations
    global total_itterations
    start_time = time.time()

    for i in range(total_itterations, total_itterations+num_iterations):
        x_batch, y_true_batch = get_train_batch(train_batch_size)

        y_true_one_hot = np.eye(num_classes)[y_true_batch]

        feed_dict_train = {x: x_batch,
                           y_true: y_true_one_hot}
        summary, _ = session.run([merged, optimizer], feed_dict=feed_dict_train)

        if i % 100 == 0:
            acc = session.run(accuracy, feed_dict=feed_dict_train)
            msg = "Optimization Iteration: {0:>6}, Training Accuracy {1:>6.1%}"
            print(msg.format(i+1, acc))

    total_itterations += num_iterations
    end_time = time.time()
    time_dif = end_time-start_time
    print("Time useage: ",str(timedelta(seconds=int(round(time_dif)))))
    return summary

def plot_example_errors(cls_pred, correct, image_set, label_set):
    # Copied and modified from Hvass Laboratories' Github for MNIST dataset
    #   https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/02_Convolutional_Neural_Network.ipynb
    # Purpose:
    #   Plotting example errors from testing data
    # Precondition:
    #   "cls_pred" is the predicted class indexes for the images
    #   "correct" is a boolean vector of correct predictions
    #   "image_set" is the set of images used
    #   "label_set" is the set of corresponding class targets
    # Postcondition:
    #   plot_images is called to plot example errors from test data
    incorrect = (correct == False)
    images = np.array(image_set)[incorrect]
    cls_pred = cls_pred[incorrect]
    cls_true = label_set[incorrect]
    plot_images(images= images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])

def print_test_accuracy(show_example_errors=False):
    # Copied and modified from Hvass Laboratories' Github for MNIST dataset
    #   https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/02_Convolutional_Neural_Network.ipynb
    # Purpose:
    #   Printing the accuracy of test batch
    # Precondition:
    #   "show_example_errors" is true if displaying example errors
    # Postcondition:
    #   Prints the accuracy of a test. If "show_example_errors" is true, display example errors from test.
    images,labels = get_test_batch(test_batch_size)

    one_hot = np.eye(num_classes)[labels]

    feed_dict = {x: images,y_true: one_hot}
    cls_pred = session.run(y_pred_cls, feed_dict=feed_dict)
    correct = (labels == cls_pred)
    correct_sum = correct.sum()
    acc = float(correct_sum)/train_batch_size
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"

    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred,
                            correct= correct,
                            image_set=images,
                            label_set=labels)

# Class labels and number of classes
classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
num_classes = 10
# Input image details
image_channels = 3
image_width = 32
image_height = 32
image_size_flat = image_width*image_height*image_channels
image_shape = (image_width,image_height)


# Placeholder variables
x = tf.placeholder(tf.float32, shape=[None, image_height, image_width, image_channels], name='x')
y_true = tf.placeholder(tf.float32,shape=[None,10],name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

# Filter and layer details
filter_size = 5
filter_count1 = 64
filter_count2 = 64
fc_size1 = 256
fc_size2 = 128

# Layer construction
layer_conv1, weights_conv1 = build_conv(input=x,
                                        num_input_channels=image_channels,
                                        filter_size=filter_size,
                                        num_filters=filter_count1,
                                        use_pooling=True, name="Conv_1")
layer_conv2, weights_conv2 = build_conv(input=layer_conv1,
                                        num_input_channels=filter_count1,
                                        filter_size=filter_size,
                                        num_filters=filter_count2,
                                        use_pooling=True, name="Conv_2")
layer_flat, num_features = flatten_layer(layer_conv2)
layer_fc1 = build_fc(input=layer_flat,
                     num_inputs=num_features,
                     num_outputs=fc_size1,
                     use_relu=True, name="FCL_1")
layer_fc2 = build_fc(input=layer_fc1,
                     num_inputs=fc_size1,
                     num_outputs=fc_size2,
                     use_relu=True, name="FCL_2")
layer_fc3 = build_fc(input=layer_fc2,
                     num_inputs=fc_size2,
                     num_outputs=num_classes,
                     use_relu=False, name="FCL_3")
y_pred = tf.nn.softmax(layer_fc3)
y_pred_cls = tf.argmax(y_pred, dimension=1)

# Obtaining error and initializing optimizer
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc3, labels=y_true)
cost = tf.reduce_mean(cross_entropy)
tf.summary.scalar("Cost", cost)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

# Obtaining accuracy
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar("Accuracy", accuracy)

# Session variables
session = tf.Session()
session.run(tf.global_variables_initializer())

merged = tf.summary.merge_all()

def save_network(path):
    # Purpose:
    #   Saving the neural network
    # Precondition:
    #   Network obtains variables
    #   "path" is an accessible path for writing the file
    # Postcondition:
    #   Saves network in "path"
    saver = tf.train.Saver()
    save_path = saver.save(session, path+".ckpt")
    with open(path+".pkl", 'wb') as fl:
        pickle.dump(total_itterations, fl)



def load_network(path):
    # Purpose:
    #   Saving the neural network
    # Precondition:
    #   Network obtains variables
    #   "path" is an accessible path that contains the file
    # Postcondition:
    #   Loads variables from file
    saver = tf.train.Saver()
    saver.restore(session, path+".ckpt")
    with open(path+".pkl", 'rb') as fl:
        total_itterations = pickle.load(fl)

def run(path, save= True, load=False):
    # Purpose:
    #   Running the neural network
    # Precondition:
    #   "path" is a path for the save file
    #   "save" is true if saving to path
    #   "load" is true if initilizing network from file
    # Postcondition:
    #   Session loads runs and saves, depending on settings
    if load:
        load_network(path)

    writer = tf.summary.FileWriter(path)
    writer.add_graph(session.graph)

    for i in range(2000):
        summary = optimize(10)
        writer.add_summary(summary, total_itterations)
        if i % 1000 == 0:
            print_test_accuracy(show_example_errors=True)

        else:
            print_test_accuracy(show_example_errors=False)
        if save and i%10 == 0:
            save_network(path)
            print("Network saved at",path)

run("data/leaky_relu_0")