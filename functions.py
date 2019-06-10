from PIL import Image
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops

config = tf.ConfigProto(
        device_count={'GPU': 1}
    )
config.gpu_options.allow_growth = True


class PrepareData:

    def __init__(self, img_fldr):
        # internals
        self.all_files = 0
        self.X = []
        self.Y: np.array
        self.datasets = {}
        self.classes = {}
        self.img_shapes = {}
        self.train_f = []
        self.test_f = []
        self.nmb_of_classes = int
        # properties
        self.img_fldr = img_fldr


    def run(self):
        self.fill_x_y()
        x_shape = self.X.shape
        self.img_shapes['img_width'] = x_shape[2]
        self.img_shapes['img_height'] = x_shape[1]

        self.define_shapes()
        self.datasets['X_train'] = self.X
        self.datasets['Y_train'] = self.Y
        print('X train shape: ' + str(self.X.shape))
        print('Y train shape: ' + str(self.Y.shape))

        self.all_files = 0
        self.X = []

        self.fill_x_y(train=False)
        self.define_shapes()
        self.datasets['X_test'] = self.X
        self.datasets['Y_test'] = self.Y
        print('X test shape: ' + str(self.X.shape))
        print('Y test shape: ' + str(self.Y.shape))

        self.datasets['classes'] = self.classes

        return self.datasets

    def define_shapes(self):
        x_shape = self.X.shape
        if len(x_shape) == 3:
            self.X = self.X.reshape(x_shape[0], x_shape[1], x_shape[2], 1)
            self.img_shapes['channels'] = 1
        else:
            self.img_shapes['channels'] = x_shape[1]
        self.datasets['img_shapes'] = self.img_shapes

    def count_files(self, train):
        self.train_f = os.listdir(self.img_fldr + 'train/')
        self.test_f = os.listdir(self.img_fldr + 'test/')
        try:
            assert self.train_f == self.test_f, "Classes mismatch"
        except AssertionError as e:
            print(e)
        else:
            self.nmb_of_classes = len(self.train_f)

        for i in range(0, self.nmb_of_classes):
            if train:
                cur_folder = self.img_fldr + 'train/' + self.train_f[i] + '/'
                self.classes[i] = self.train_f[i]
            else:
                cur_folder = self.img_fldr + 'test/' + self.test_f[i] + '/'
            files = os.listdir(cur_folder)
            self.all_files += len(files)

        self.Y = np.zeros([self.all_files, self.nmb_of_classes])

    def fill_x_y(self, train=True):
        self.count_files(train)
        cur_file = 0
        for i in range(0, self.nmb_of_classes):
            if train:
                cur_folder = self.img_fldr + 'train/' + self.train_f[i] + '/'
            else:
                cur_folder = self.img_fldr + 'test/' + self.test_f[i] + '/'
            files = os.listdir(cur_folder)
            for ii in range(len(files)):
                file = files[ii]
                im = Image.open(cur_folder + str(file))
                np_im = np.array(im)
                self.X.append(np_im)
                self.Y[cur_file, i] = 1
                cur_file += 1

        self.X = np.asarray(self.X)


def create_placeholders(n_H0, n_W0, n_C0, n_y):

    X = tf.placeholder(tf.float32, shape=(None, n_H0, n_W0, n_C0))
    Y = tf.placeholder(tf.float32, shape=(None, n_y))

    return X, Y


def initialize_parameters(model_params):
    parameters = {}
    for i in range(model_params['layers']):
        l_param = model_params['layers_params']['layer' + str(i + 1)]
        W = 'W'+str(i+1)
        filter = l_param['filter']
        parameters[W] = tf.get_variable(W, filter, initializer=tf.contrib.layers.xavier_initializer())

    return parameters


def forward_propagation(X, parameters, model_params, n_y):
    act_dic = dict()
    l = model_params['layers']
    for i in range(l):
        l_param = model_params['layers_params']['layer' + str(i + 1)]
        if i == 0:
            W = parameters['W'+str(i+1)]
            Z = tf.nn.conv2d(X, W, strides=l_param['strides'], padding=l_param['padding'])
            A = tf.nn.relu(Z)
            act_dic['P'+str(i+1)] = tf.nn.max_pool(A, ksize=l_param['mp_ksize'],
                                                   strides=l_param['mp_strides'], padding=l_param['mp_padding'])
        else:
            W = parameters['W' + str(i + 1)]
            Z = tf.nn.conv2d(act_dic['P'+str(i)], W, strides=l_param['strides'], padding=l_param['padding'])
            A = tf.nn.relu(Z)
            act_dic['P' + str(i + 1)] = tf.nn.max_pool(A, ksize=l_param['mp_ksize'],
                                                       strides=l_param['mp_strides'], padding=l_param['mp_padding'])

    LAST_fl = tf.contrib.layers.flatten(act_dic['P' + str(l)])
    Z3 = tf.contrib.layers.fully_connected(LAST_fl, n_y, activation_fn=None)

    return Z3


def compute_cost(Z3, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))
    # cost = tf.nn.sigmoid(Z3)
    return cost

def random_mini_batches(X, Y, model_params):

    mini_batch_size = model_params['minibatch_size']
    m = X.shape[0]  # number of training examples
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = X[permutation[k * mini_batch_size: (k + 1) * mini_batch_size]]
        mini_batch_Y = Y[permutation[k * mini_batch_size: (k + 1) * mini_batch_size]]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = X[permutation[num_complete_minibatches * mini_batch_size:]]
        mini_batch_Y = Y[permutation[num_complete_minibatches * mini_batch_size:]]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

def model(datasets, model_params):
    X_train = datasets['X_train']
    Y_train = datasets['Y_train']
    X_test = datasets['X_test']
    Y_test = datasets['Y_test']

    learning_rate = model_params['learning_rate']
    num_epochs = model_params['num_epochs']
    minibatch_size = model_params['minibatch_size']
    print_cost = model_params['print_cost']

    ops.reset_default_graph()
    tf.set_random_seed(1)
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []

    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
    parameters = initialize_parameters(model_params)
    Z3 = forward_propagation(X, parameters, model_params, n_y)
    cost = compute_cost(Z3, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    init = tf.global_variables_initializer()
    with tf.Session(config=config) as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
            minibatches = random_mini_batches(X_train, Y_train, model_params)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _, temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                minibatch_cost += temp_cost / num_minibatches

            if print_cost == True and epoch % 5 == 0:
                print("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)
        if print_cost == True:
            plt.plot(np.squeeze(costs[model_params['skip_steps']:]))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()

        # Calculate the correct predictions
        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        images = X_train.shape[0]
        train_accuracies = []
        ts_btch = model_params['accuracy_batch']
        for loop in range(1, int(images/ts_btch)):
            if loop != int(images/ts_btch)-1:
                batch_X = X_train[(loop - 1)*ts_btch:loop*ts_btch, :, :, :]
                batch_Y = Y_train[(loop - 1) * ts_btch:loop * ts_btch, :]
                train_accuracy = accuracy.eval({X: batch_X, Y: batch_Y})
                train_accuracies.append(train_accuracy)
            else:
                batch_X = X_train[(loop - 1) * ts_btch:, :, :, :]
                batch_Y = Y_train[(loop - 1) * ts_btch:, :]
                train_accuracy = accuracy.eval({X: batch_X, Y: batch_Y})
                train_accuracies.append(train_accuracy)

        images = X_test.shape[0]
        test_accuracies = []
        for loop in range(1, int(images / ts_btch)):
            if loop != int(images / ts_btch) - 1:
                batch_X = X_test[(loop - 1) * ts_btch:loop * ts_btch, :, :, :]
                batch_Y = Y_test[(loop - 1) * ts_btch:loop * ts_btch, :]
                test_accuracy = accuracy.eval({X: batch_X, Y: batch_Y})
                test_accuracies.append(test_accuracy)
            else:
                batch_X = X_test[(loop - 1) * ts_btch:, :, :, :]
                batch_Y = Y_test[(loop - 1) * ts_btch:, :]
                test_accuracy = accuracy.eval({X: batch_X, Y: batch_Y})
                test_accuracies.append(test_accuracy)

    return train_accuracies, test_accuracies, parameters

