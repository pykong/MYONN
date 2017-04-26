#!/usr/bin/env python3

import csv
import datetime
import os
import shelve
from time import time

import matplotlib.pyplot as plt
import numpy as np
import wget
from scipy.special import expit as activation_function  # sigmoid function
from scipy.special import logit as inverse_activation_function

# CONFIG
DATA_DIR = "data"
WEIGHTS_FILE = "weights.db"

# DEBUG = True will use smaller dataset, single run and show pictures
# DEBUG = False will train network with a range of different parameters
DEBUG = True
INPUT_NODES = 28**2
OUTPUT_NODES = 10

DATA_DIR = "test_data/"
TRAIN_FILE = os.path.join(DATA_DIR, "mnist_train.csv")
TEST_FILE = os.path.join(DATA_DIR, "mnist_test.csv")

TRAINFILE_URL = "http://www.pjreddie.com/media/files/mnist_train.csv"
TESTFILE_URL = "http://www.pjreddie.com/media/files/mnist_test.csv"


def reversed_enumerate(sequence):
    return zip(
        reversed(range(len(sequence))),
        reversed(sequence),
    )


class NeuralNetwork:
    def __init__(self, hidden_layers, input_nodes, hidden_nodes, output_nodes, learning_rate):
        """ set number of nodes in each input, hidden, output layer """

        self.lr = learning_rate

        # defining weights
        # input to hidden layer
        l_ih = self.__init_weights(input_nodes, hidden_nodes)
        # hidden to hidden layer
        l_hh = self.__init_weights(hidden_nodes, hidden_nodes)
        # hidden to output layer
        l_ho = self.__init_weights(hidden_nodes, output_nodes)

        self.layer_weights = [l_ih]
        for i in range(0, hidden_layers - 1):
            self.layer_weights.append(l_hh)

        self.layer_weights.append(l_ho)  # closing with output layer

    def __init_weights(self, this_layer, next_layer):
        return np.random.normal(0.0, pow(next_layer, -0.5), (next_layer, this_layer))

    def __calculate_outputs(self, weights, inputs):
        # Step 1 - calculate signals into hidden layer
        weighted_inputs = np.dot(weights, inputs)
        # calculate the signals emerging from hidden layer
        outputs = activation_function(weighted_inputs)
        return outputs

    def __update_weights(self, this_errors, this_outputs, previous_outputs):
        """ """
        # TODO: ValueError: operands could not be broadcast together with shapes (10,1) (200,1)
        x = this_errors * this_outputs * (1.0 - this_outputs)
        updated_weights = self.lr * np.dot(x, np.transpose(previous_outputs))
        return updated_weights

    def __propagate(self, inputs_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        # initialize with inputs, as "outputs from picture"
        outputs_list = [inputs]  # TODO: indexing does not work this way
        for i, weights in enumerate(self.layer_weights):
            this_output = self.__calculate_outputs(weights, outputs_list[i])
            outputs_list.append(this_output)
        return outputs_list

    def train(self, inputs_list, targets_list):
        outputs_list = self.__propagate(inputs_list)
        # transpose the inputs list to a vertical array
        targets = np.array(targets_list, ndmin=2).T

        # backpropagate
        new_weights_rev = []

        this_errors = targets - outputs_list[-1]  # starting with output_errors

        for i, weights in reversed_enumerate(self.layer_weights):
            # print(i)
            # for weights, outputs in reversed(list(zip(self.layer_weights, outputs_list))):
            w_delta = self.__update_weights(this_errors, outputs_list[i + 1], outputs_list[i])
            new_weight = weights + w_delta
            new_weights_rev.append(new_weight)

            this_errors = np.dot(weights.T, this_errors)

        # update wieghts
        self.layer_weights = new_weights_rev[::-1]  # must be reversed

    def query(self, inputs_list):
        output_list = self.__propagate(inputs_list)
        return output_list[-1]  # return final outputs

    # backquery the neural network
    # we'll use the same termnimology to each item,
    # eg target are the values at the right of the network, albeit used as input
    # eg hidden_output is the signal to the right of the middle nodes
    def backquery(self, targets_list):

        # transpose the targets list to a vertical array
        outputs = np.array(targets_list, ndmin=2).T
        for weights in reversed(self.layer_weights):

            # calculate the signal into the final output layer
            inputs = inverse_activation_function(outputs)

            # calculate the signal out of the hidden layer
            outputs = np.dot(weights.T, inputs)
            # scale them back to 0.01 to .99
            outputs -= np.min(outputs)
            outputs /= np.max(outputs)
            outputs *= 0.98
            outputs += 0.01

        return outputs

# HELPER FUNCTIONS -----------------------------------


def ensure_files():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    if not os.path.exists(TRAIN_FILE):
        print("Downloading: {}".format(TRAIN_FILE))
        wget.download(TRAINFILE_URL, out=DATA_DIR)

    if not os.path.exists(TEST_FILE):
        print("Downloading: {}".format(TEST_FILE))
        wget.download(TESTFILE_URL, out=DATA_DIR)


def scale_matrix(matrix):
    """ nn works best with values between 0.01 and 1 """
    return matrix / 255 * 0.99 + 0.01


# TODO: transform to generator
def import_data(file_name):

    with open(file_name, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            label = int(row.pop(0))
            matrix = scale_matrix(np.asfarray(row))  # .reshape((28, 28)))
            yield label, matrix


def gen_target_array(onodes):
    """ generates list of lists, like:
    [[0.01, 0.01, 0.01, 0.01, 0.01, 0.99, 0.01, 0.01, 0.01, 0.01], ... ] """
    o_list = []
    for index in range(onodes):
        targets = np.zeros(onodes) + 0.01
        targets[index] = 0.99
        o_list.append(targets)
    return o_list


def plot_matrix(matrix):
    # what the hell is this doing?
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect('equal')

    plt.imshow(
        matrix, interpolation='nearest',
        cmap=plt.cm.Greys)  # use diffetent cmap?
    plt.colorbar()
    plt.show()


def calc_accuracy(scorecard):
    scorecard_array = np.asarray(scorecard)
    accuracy = scorecard_array.sum() / scorecard_array.size
    return accuracy


def get_sample_size(file_name):
    if DEBUG:
        return 1000
    sample_size = 0
    with open(file_name, "r") as f:
        reader = csv.reader(f)
        sample_size = len(list(reader))
    return sample_size


def run_experiment(hidden_layers, hidden_nodes, learning_rate, epochs):
    # init network
    nn = NeuralNetwork(
        input_nodes=INPUT_NODES,
        hidden_layers=hidden_layers,
        hidden_nodes=hidden_nodes,
        output_nodes=OUTPUT_NODES,
        learning_rate=learning_rate
    )

    target_ar = gen_target_array(OUTPUT_NODES)

    # train network
    training_data = import_data(TRAIN_FILE)
    if DEBUG:  # use less training data for debug
        training_data = list(training_data)[1000:]
    t0 = time()
    for e in range(epochs):
        for record in training_data:
            label, matrix = record
            nn.train(inputs_list=matrix, targets_list=target_ar[label])
    t1 = time() - t0

    # test network
    scorecard = []  # scorecard for how well the network performs
    test_data = import_data(TEST_FILE)
    if DEBUG:  # use less test data for debug
        test_data = list(test_data)[100:]
    for record in test_data:
        correct_label, matrix = record
        outputs = nn.query(matrix)
        label = np.argmax(
            outputs)  # the index of the highest value corresponds to the label

        if (label == correct_label):
            # network's answer matches correct answer, add 1 to scorecard
            scorecard.append(1)
        else:
            # network's answer doesn't match correct answer, add 0 to scorecard
            scorecard.append(0)

    accuracy = calc_accuracy(scorecard)

    # print some info regarding training run
    print("train and test finished\n\n")
    print("accuracy: {0:.1f}%".format(calc_accuracy(scorecard) * 100))
    print("training time: {}".format(str(datetime.timedelta(seconds=t1))))
    print("-" * 15)
    print("sample size: {}".format(get_sample_size(TRAIN_FILE)))
    print("epochs: {}".format(epochs))
    print("hidden layers: {}".format(hidden_layers))
    print("hidden nodes: {}".format(hidden_nodes))
    print("learning rate: {}".format(learning_rate))

    if DEBUG:
        print("\n\nBackquerying the network")
        for i, t in enumerate(target_ar):
            image_data = nn.backquery(t)
            plt.imshow(image_data.reshape(28, 28),
                       cmap='Greys', interpolation='None')
            plt.title("Target: " + str(i))
            plt.show()
    else:
        # save current config to db if better accuracy
        d = shelve.open(WEIGHTS_FILE)
        try:
            data = d["data"]
            data_accuracy = data["accuracy"]
        except:
            d["data"] = {"accuracy": accuracy, "weights": nn.layer_weights}
        else:
            # save only if accuracy higher than current best
            if accuracy > data_accuracy:  # uncomment to save all
                d["data"] = {"accuracy": accuracy, "weights": nn.layer_weights}
        finally:
            d.close()


if __name__ == '__main__':
    ensure_files()  # download training and test data if not present
    if DEBUG:
        print("Running in DEBUG mode.\n\n")
        run_experiment(hidden_layers=0, hidden_nodes=50, learning_rate=0.2, epochs=1)
    else:
        print("Running combinatorial query over parameters.\n\n")
        for hidden_layers in range(2):
            for hidden_nodes in [50, 80, 150, 200, 300]:
                for learning_rate in [0.05, 0.1, 0.2, 0.3, 0.5]:
                    for epochs in [2, 3, 5]:
                        run_experiment(hidden_layers, hidden_nodes, learning_rate, epochs)
