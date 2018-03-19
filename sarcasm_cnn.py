import math
import numpy as np
import h5py
import string
import re
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import fnmatch
import scipy
from PIL import Image
from scipy import ndimage
from tensorflow.python.framework import ops
from gensim.models import Word2Vec
from nltk.corpus import brown
from tf_utils import load_dataset, random_mini_batches, random_mini_batches_cnn, convert_to_one_hot, predict

sarcSentences = []
notSarcSentences = []
fullSentencesForTFIDF = []
idfValues = dict()
punctuation = ['.', ',', ';', ':', '!', '?', '\"', '\'']
counter = 0
trainTestSplit = 0
maxWords = 30
numInvalid = dict()
numInvalid['invalid'] = 0



def jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection)/len(union)

def term_frequency(term, tokenized_document):
    return tokenized_document.count(term)

def sublinear_term_frequency(term, tokenized_document):
    count = tokenized_document.count(term)
    if count == 0:
        return 0
    return 1 + math.log(count)

def augmented_term_frequency(term, tokenized_document):
    max_count = max([term_frequency(t, tokenized_document) for t in tokenized_document])
    return (0.5 + ((0.5 * term_frequency(term, tokenized_document))/max_count))

def inverse_document_frequencies(tokenized_documents):
    idf_values = {}
    all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])
    for tkn in all_tokens_set:
        contains_token = map(lambda doc: tkn in doc, tokenized_documents)
        idf_values[tkn] = 1 + math.log(len(tokenized_documents)/(sum(contains_token)))
    return idf_values

'''
CSV format of this file is weird, so had to do some manual extraction to get word embeddings
embedding = turns word to a vector
'''
def initializeEmbeddings(numTrainExamples):
    
    for file in os.listdir("."):
        counter = 1
        if not fnmatch.fnmatch(file, "*train-balanced-sarcasm_*"):
            continue
        with open(file) as infile:
            for line in infile:
                if counter == 1:
                    counter += 1
                    continue
                if len(sarcSentences) >= numExamples and len(notSarcSentences) >= numExamples:
                    break
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
                #fullSentencesForTFIDF.append(" ".join(words))
    embeddings = Word2Vec(sarcSentences[:numTrainExamples] + notSarcSentences[:numTrainExamples] + brown.sents(), min_count=1)
    someIdfVals = None#inverse_document_frequencies(sarcSentences[:numTrainExamples] + notSarcSentences[:numTrainExamples])
    return someIdfVals, embeddings


def formulateWordVector(words, embeddings):
    imageW2V = np.zeros((maxWords, 100, 1))
    vocabFound = 0
    for word in words:
        if vocabFound >= maxWords:
            break
        if word not in embeddings.wv:
            continue
        multiplier = 1#sublinear_term_frequency(word, words) * idfValues[word]
        if multiplier == 0:
            continue

        imageW2V[vocabFound, :, 0] = np.array(embeddings.wv[word])
        vocabFound += 1

    return imageW2V, True


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
    numSarcTrain = 0
    numNotSarcTrain = 0
    numSarcTest = 0
    numSarcTrain = 0
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
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)


    return X_train, Y_train, X_test, Y_test


def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_H0 -- scalar, height of an input image
    n_W0 -- scalar, width of an input image
    n_C0 -- scalar, number of channels of the input
    n_y -- scalar, number of classes
        
    Returns:
    X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
    """

    X = tf.placeholder(dtype="float", shape=(None, n_H0, n_W0, n_C0), name='X')
    Y = tf.placeholder(dtype="float", shape=(None,n_y), name='Y')
    ### END CODE HERE ###
    
    return X, Y

def initialize_parameters():
    """
    Initializes weight parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [4, 4, 3, 8]
                        W2 : [2, 2, 8, 16]
    Returns:
    parameters -- a dictionary of tensors containing W1, W2
    """
    
    tf.set_random_seed(1)                              # so that your "random" numbers match ours
    parameters = {}
    ### START CODE HERE ### (approx. 2 lines of code)
    for i in range(1, 6):#maxWords+1):
        # seed in xavier
        parameters["W1_" + str(i)] = tf.get_variable("W1_" + str(i), [i, 100, 1, 2], initializer=tf.contrib.layers.xavier_initializer(seed = 0))
    ### END CODE HERE ###
    
    return parameters

# GRADED FUNCTION: forward_propagation

def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "W2"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """

    
    ### START CODE HERE ###
    # CONV2D: stride of 1, padding 'SAME'
    univarVec = None
    for param in parameters:
        someW = parameters[param]
        Z1 = tf.nn.conv2d(X, someW, strides = [1,1,1,1], padding = 'VALID')
        A1 = tf.nn.relu(Z1)
        numRows = np.shape(Z1)[1]
        P1 = tf.nn.max_pool(A1, ksize = [1,numRows,1,1], strides = [1,1,1,1], padding = 'VALID')
        thinVec = tf.contrib.layers.flatten(P1)
        if univarVec == None:
            univarVec = tf.contrib.layers.flatten(P1)
        else:
            univarVec = tf.concat([univarVec,thinVec], 1)

    Z3 = tf.contrib.layers.fully_connected(univarVec,2, activation_fn=None)

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
    
    ### START CODE HERE ### (1 line of code)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels=Y))
    ### END CODE HERE ###
    
    return cost


# GRADED FUNCTION: model

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.009,
          num_epochs = 100, minibatch_size = 64, print_cost = True):
    """
    Implements a three-layer ConvNet in Tensorflow:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    
    Arguments:
    X_train -- training set, of shape (None, 64, 64, 3)
    Y_train -- test set, of shape (None, n_y = 6)
    X_test -- training set, of shape (None, 64, 64, 3)
    Y_test -- test set, of shape (None, n_y = 6)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    train_accuracy -- real number, accuracy on the train set (X_train)
    test_accuracy -- real number, testing accuracy on the test set (X_test)
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep results consistent (tensorflow seed)
    seed = 3                                          # to keep results consistent (numpy seed)
    (m, n_H0, n_W0, n_C0) = X_train.shape             
    n_y = Y_train.shape[1]                            
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of the correct shape
    ### START CODE HERE ### (1 line)
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
    ### END CODE HERE ###

    # Initialize parameters
    ### START CODE HERE ### (1 line)
    parameters = initialize_parameters()
    ### END CODE HERE ###
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    ### START CODE HERE ### (1 line)
    Z3 = forward_propagation(X, parameters)
    ### END CODE HERE ###
    
    # Cost function: Add cost function to tensorflow graph
    ### START CODE HERE ### (1 line)
    cost = compute_cost(Z3, Y)
    ### END CODE HERE ###
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    ### START CODE HERE ### (1 line)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    ### END CODE HERE ###
    
    # Initialize all the variables globally
    init = tf.global_variables_initializer()
     
    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):
            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches_cnn(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
                ### START CODE HERE ### (1 line)
                _ , temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                ### END CODE HERE ###
                
                minibatch_cost += temp_cost / num_minibatches
                

            # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)
        
        
        # plot the cost
        plt.figure()
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.savefig('cnn_plot_' + str(numExamples) + ".png")

        # Calculate the correct predictions
        oneArr = tf.ones(tf.shape(tf.argmax(Z3, 1)), dtype=tf.int64)
        zeroArr = tf.zeros(tf.shape(tf.argmax(Z3, 1)),  dtype=tf.int64)
        twosArr = tf.fill(tf.shape(tf.argmax(Z3, 1)), 2.0)

        # Number of false positives
        false_pos = tf.reduce_sum(tf.cast(tf.less(tf.argmax(Z3, 1), tf.argmax(Y, 1)), "float"))
        true_pos = tf.reduce_sum(tf.cast(tf.equal(tf.add(tf.argmax(Z3, 1), tf.argmax(Y, 1)), zeroArr), "float"))
        true_neg = tf.reduce_sum(tf.cast(tf.equal(tf.add(tf.argmax(Z3, 1), tf.argmax(Y, 1)), tf.cast(twosArr, tf.int64)), "float"))
        false_neg = tf.reduce_sum(tf.cast(tf.greater(tf.argmax(Z3, 1), tf.argmax(Y, 1)), "float"))

        # All false and true positives
        false_and_true_pos = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(Z3, 1), zeroArr), "float"))
        false_and_true_neg = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(Z3, 1), oneArr), "float"))


        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3, 1), tf.argmax(Y, 1))
        
        precisionCounter = tf.divide(1.0*true_pos, 1.0*(true_pos+false_pos))
        recallCounter = tf.divide(true_pos*1.0,1.0*(true_pos+false_neg))
        otherSide = tf.divide(true_neg*1.0, 1.0*(true_neg+false_pos))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

        precision = precisionCounter.eval({X: X_test, Y: Y_test})
        recall = recallCounter.eval({X: X_test, Y: Y_test})
        otherCalc = otherSide.eval({X: X_test, Y: Y_test})
        confusionMatrix = [[recall, 1-recall], [1-otherCalc, otherCalc]]
        fScore = 2*(precision*recall)/(precision+recall)
        print ("Precision: ", precision)
        print ("Recall: ", recall)
        print ("F1 Score: ", fScore)
        print(confusionMatrix)
        return train_accuracy, test_accuracy, parameters


numExamples = int(sys.argv[1])/2
idfValues, embeddings = initializeEmbeddings(numExamples)
del fullSentencesForTFIDF
'''
Change the 500000 to half of the number you want to use
e.g. I wanted to try one mill, so I used 500k sarc/500k not sarc.
'''

print("For " + str(2*numExamples) + " examples")
sarcSentences = sarcSentences[:numExamples]
notSarcSentences = notSarcSentences[:numExamples]
trainTestSplit = 9*(len(sarcSentences))/10

    
# At this point we have our test and train data, just need to apply the tensorflow assignment code to it
X_train, Y_train, X_test, Y_test = formTrainTestSplits(trainTestSplit, embeddings)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
_, _, parameters = model(X_train, Y_train, X_test, Y_test)