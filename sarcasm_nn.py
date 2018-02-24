import math
import numpy as np
import h5py
import string
import re
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from gensim.models import Word2Vec
from nltk.corpus import brown
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

sarcSentences = []
notSarcSentences = []
punctuation = ['.', ',', ';', ':', '!', '?', '\"', '\'']
counter = 0
trainTestSplit = 0
numInvalid = dict()
numInvalid['invalid'] = 0

'''
CSV format of this file is weird, so had to do some manual extraction to get word embeddings
embedding = turns word to a vector
'''
def initializeEmbeddings():
    maxWordsAppeared = 0
    indexOfMax = -1
    counter = 1
    with open("train-balanced-sarcasm.csv") as infile:
        for line in infile:
            if line[0] == 'l':
                continue
            currLabel = int(line[0])
            dateReg = r'([12]\d{3}-(0[1-9]|1[0-2]),[12]\d{3}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01]))'
            start = re.search(dateReg, line).start()
            sentence = line[2:start-1]
            for i in range(5):
                indexOfComma = sentence.rfind(',')
                sentence = sentence[:indexOfComma]
            words = sentence.split()
            for i in range(len(words)):
                while len(words[i]) != 0 and words[i][len(words[i])-1] in punctuation:
                    words[i] = words[i][:len(words[i])-1]
                while len(words[i]) != 0 and words[i][0] in punctuation:
                    words[i] = words[i][1:]
            if currLabel == 0:
                notSarcSentences.append(words)
            else:
                sarcSentences.append(words)
            if len(words) > maxWordsAppeared:
                maxWordsAppeared = len(words)
                indexOfMax = counter
            counter += 1

    embeddings = Word2Vec(sarcSentences + notSarcSentences + brown.sents(), min_count=1)
    print("Max words appeared in a sentence: " + str(maxWordsAppeared))
    print("The example number is " + str(indexOfMax))
    return embeddings

'''
Now given a set of words, find the sentence represented in a vector format
To get sentence vector, average word vectors in embeddings
'''
def formulateWordVector(words, embeddings):
    finalVec = np.zeros((100,))
    vocabFound = 0
    for word in words:
        if word not in embeddings.wv:
            continue
        vocabFound += 1
        tempVec = np.array(embeddings.wv[word])
        finalVec = finalVec + tempVec

    if vocabFound == 0:
        numInvalid['invalid'] += 1
        return finalVec, False
    finalVec = finalVec/(1.0*vocabFound)

    return finalVec, True

'''
Split train/test (approx a 90/10 split)
'''
def formTrainTestSplits(indexOfSplit, embeddings):
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []

    # There is some invalid data, so before adding to train/test sets, check
    # if the sentence is valid (i.e. no words in the sentence)
    for i in range(indexOfSplit):
        sarcVec, isValid = formulateWordVector(sarcSentences[i], embeddings)
        if isValid:
            X_train.append(sarcVec)
            Y_train.append([1,0])

        notSarcVec, isValid = formulateWordVector(notSarcSentences[i], embeddings)
        if isValid:
            X_train.append(notSarcVec)
            Y_train.append([0,1])
    for i in range(indexOfSplit, len(sarcSentences)):
        sarcVec, isValid = formulateWordVector(sarcSentences[i], embeddings)
        if isValid:
            X_test.append(sarcVec)
            Y_test.append([1,0])

        notSarcVec, isValid = formulateWordVector(notSarcSentences[i], embeddings)
        if isValid:
            X_test.append(notSarcVec)
            Y_test.append([0,1])

    # Use format to match 230's tensorflow assignment
    X_train = np.array(X_train).T
    Y_train = np.array(Y_train).T
    X_test = np.array(X_test).T
    Y_test = np.array(Y_test).T
    return X_train, Y_train, X_test, Y_test

def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
    n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
    n_y -- scalar, number of classes (from 0 to 5, so -> 6)

    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"

    Tips:
    - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
      In fact, the number of examples during test/train is different.
    """

    X = tf.placeholder(tf.float32, shape=(n_x,None), name = 'X')
    Y = tf.placeholder(tf.float32, shape=(n_y,None), name = 'Y')

    return X, Y

def initialize_parameters():
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [25, 100]
                        b1 : [25, 1]
                        W2 : [12, 25]
                        b2 : [12, 1]
                        W3 : [2, 12]
                        b3 : [2, 1]

    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """

    tf.set_random_seed(1)                   # so that your "random" numbers match ours

    '''
    The way this is constructed right now is there is 2 hidden layers, so experiment
    with the unit numbers and layers (more layers = more W/b's)
    '''
    W1 = tf.get_variable("W1", [25,100], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1", [25,1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [12,25], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2", [12,1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [2,12], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable("b3", [2,1], initializer = tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return parameters


def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """

    # Retrieve the parameters from the dictionary "parameters"
    '''
    Number of W/b parameters depends on hidden layer number
    '''
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    '''
    This would also change, (for 3 hidden layers there would be up to Z4, etc)
    '''
    Z1 = tf.add(tf.matmul(W1, X), b1)                                            # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                              # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)                                              # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                              # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)                                              # Z3 = np.dot(W3,Z2) + b3

    return Z3

def compute_cost(Z3, Y):
    """
    Computes the cost

    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3

    Returns:
    cost - Tensor of the cost function
    """

    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))

    return cost

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
          num_epochs = 1500, minibatch_size = 32, print_cost = True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.

    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost

    # Create Placeholders of shape (n_x, n_y)
    X, Y = create_placeholders(n_x, n_y)

    # Initialize parameters
    parameters = initialize_parameters()

    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3 = forward_propagation(X, parameters)

    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z3, Y)

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:

        # Run the initialization
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch

                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

        return parameters

embeddings = initializeEmbeddings()

'''
Change the 500000 to half of the number you want to use
e.g. I wanted to try one mill, so I used 500k sarc/500k not sarc.
'''
numExamples = int(sys.argv[1])/2
sarcSentences = sarcSentences[:numExamples]
notSarcSentences = notSarcSentences[:numExamples]
trainTestSplit = 9*(len(sarcSentences))/10


# At this point we have our test and train data, just need to apply the tensorflow assignment code to it
X_train, Y_train, X_test, Y_test = formTrainTestSplits(trainTestSplit, embeddings)
parameters = model(X_train, Y_train, X_test, Y_test)
