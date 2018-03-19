import math
import numpy as np
import h5py
import string
import re
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from nltk.corpus import brown

np.random.seed(0)
import keras as K
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
np.random.seed(1)

sarcSentences = []
notSarcSentences = []
punctuation = ['.', ',', ';', ':', '!', '?', '\"', '\'']
counter = 0
trainTestSplit = 0

max_features = 69659
embeddingsDim = 200
n_a = 64 #This is size of bidirectional lstm
n_s = 128#This is size of post-attention lstm
Tx = 50 #Max sentence length
Ty = 2 #We are outputting length 2 vector

numInvalid = dict()
numInvalid['invalid'] = 0

# Defined shared layers as global variables
repeator = RepeatVector(Tx)
concatenator = Concatenate(axis=-1)
densor1 = Dense(10, activation = "relu")
densor2 = Dense(1, activation = "relu")
activator = Activation(activation="softmax", name='attention_weights') # We are using a custom softmax(axis = 1) loaded in this notebook
dotor = Dot(axes = 1)

'''
CSV format of this file is weird, so had to do some manual extraction to get word embeddings
embedding = turns word to a vector
'''
def initializeEmbeddings():

    with open("train-balanced-sarcasm_1.csv") as infile:
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
            words = [x.lower() for x in sentence.split()] #already splitting here
            for i in range(len(words)):
                while len(words[i]) != 0 and words[i][len(words[i])-1] in punctuation:
                    words[i] = words[i][:len(words[i])-1]
                while len(words[i]) != 0 and words[i][0] in punctuation:
                    words[i] = words[i][1:]
            if currLabel == 0:
                notSarcSentences.append(words)
            else:
                sarcSentences.append(words)
    embeddings = Word2Vec(sarcSentences, size = embeddingsDim, min_count=1)

    return embeddings.wv.syn0 #These are the weights

    return model


def one_step_attention(a, s_prev):
    """
    Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
    "alphas" and the hidden states "a" of the Bi-LSTM.

    Arguments:
    a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
    s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)

    Returns:
    context -- context vector, input of the next (post-attetion) LSTM cell
    """

    ### START CODE HERE ###
    s_prev = repeator(s_prev)
    concat = concatenator([a, s_prev])
    e = densor1(concat)
    energies = densor2(e)
    alphas = activator(energies)
    context = dotor([alphas, a])
    ### END CODE HERE ###

    return context


post_activation_LSTM_cell = LSTM(128, return_state = True)
output_layer = Dense(1, activation="softmax")


# GRADED FUNCTION: model

def model(Tx, Ty, n_a, n_s, embedding_weights):
    """
    Arguments:
    Tx -- length of the input sequence
    Ty -- length of the output sequence
    n_a -- hidden state size of the Bi-LSTM
    n_s -- hidden state size of the post-attention LSTM
    human_vocab_size -- size of the python dictionary "human_vocab"
    machine_vocab_size -- size of the python dictionary "machine_vocab"

    Returns:
    model -- Keras model instance
    """

    # Define the inputs of your model with a shape (Tx,)
    # Define s0 and c0, initial hidden state for the decoder LSTM of shape (n_s,)
    X = Input(shape=(Tx, ))
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0

    # Initialize empty list of outputs
    outputs = []

    ### START CODE HERE ###
    embeddings = Embedding(max_features, embeddingsDim, input_length=Tx, weights=[embedding_weights])(X)
    a = Bidirectional(LSTM(n_a, return_sequences=True))(embeddings)

    outputs = []
    # Step 2: Iterate for n_s(originally Ty) steps
    for t in range(Ty):
        context = one_step_attention(a, s)

        s, _, c = post_activation_LSTM_cell(context, initial_state=[s, c])

        out = output_layer(s)

        outputs.append(out)

    model = Model(inputs=[X, s0, c0], outputs=outputs)

    ### END CODE HERE ###

    return model

weights = initializeEmbeddings() #These are the weights.
print("done init")
trainTestSplit = 9*(len(sarcSentences))/10 #Do train-test split here
numExamples = 150
sarcSentences = sarcSentences[:numExamples]
notSarcSentences = notSarcSentences[:numExamples]

#TODO add test
xTrain = [] #These are the data
yTrain = [] #These are the labels
for i, x in enumerate(sarcSentences):
    xTrain.append(x)
    yTrain.append([1, 0])
for i, y in enumerate(notSarcSentences):
    xTrain.append(y)
    yTrain.append([0, 1])

tokenizer = Tokenizer(num_words=max_features, lower=True, split=" ")
tokenizer.fit_on_texts(xTrain)
train_sequences = sequence.pad_sequences(tokenizer.texts_to_sequences(xTrain) , maxlen = Tx) #padding sentences to max length.
#test_sequences = sequence.pad_sequences( tokenizer.texts_to_sequences(yTrain) , maxlen = maxSentenceLength)
train_matrix = tokenizer.texts_to_matrix(xTrain) #converting texts to matrices
#est_matrix = tokenizer.texts_to_matrix( test_texts )
print("done tokenizing")

model = model(Tx, Ty, n_a, n_s, weights)
s0 = np.zeros((numExamples * 2, n_s))
c0 = np.zeros((numExamples * 2, n_s))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit([train_sequences, s0, c0], list(np.array(yTrain).swapaxes(0, 1)), epochs = 50, batch_size = 32, shuffle=True)
print("done fitting")