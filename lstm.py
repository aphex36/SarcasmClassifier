import math
import numpy as np
import h5py
import string
import re
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from nltk.corpus import brown
import sys
import sklearn as sk

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

sarcSentences = []
notSarcSentences = []
punctuation = ['.', ',', ';', ':', '!', '?', '\"', '\'']
counter = 0
trainTestSplit = 0
numExamples = int(sys.argv[1])/2
maxSentenceLength = 50
embeddingsDim = 200

numInvalid = dict()
numInvalid['invalid'] = 0

'''
CSV format of this file is weird, so had to do some manual extraction to get word embeddings
embedding = turns word to a vector
'''
def initializeEmbeddings():

    with open("train-balanced-sarcasm_1.csv") as infile:
        for line in infile:
            if line[0] == 'l':
                continue
            if len(sarcSentences) >= numExamples and len(notSarcSentences) >= numExamples:
                x = 3
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
    embeddings = Word2Vec(sarcSentences + brown.sents(), size = embeddingsDim, min_count=1)
    return embeddings.wv.syn0 #These are the weights

def sarcasmModel(inputShape, max_features, embeddings_dim, embedding_weights):
    # Propagate sentence_indices through your embedding layer, you get back the embeddings
    sentence = Input(name='input', shape=(maxSentenceLength,))
    #embed sentence
    embeddings = Embedding(max_features, embeddings_dim, input_length=maxSentenceLength, mask_zero=True, weights=[embedding_weights])(sentence)

    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a batch of sequences.
    X = LSTM(128, return_sequences=True)(embeddings)
    # Add dropout with a probability of 0.5
    X = Dropout(.9)(X)
    # Propagate X trough another LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a single hidden state, not a batch of sequences.
    X = LSTM(128)(X)
    # Add dropout with a probability of 0.5
    X = Dropout(.9)(X)
    # Propagate X through a Dense layer with softmax activation to get back a batch of 5-dimensional vectors.
    X = Dense(2)(X)
    # Add a softmax activation
    X = Activation('softmax')(X)

    # Create Model instance which converts sentence_indices into X.
    model = Model(sentence, X)

    return model

weights = initializeEmbeddings() #These are the weights.
max_features = np.shape(weights)[0]
print("done init")
trainTestSplit = 9*(numExamples)/10 #Do train-test split here
np.random.shuffle(sarcSentences)
np.random.shuffle(notSarcSentences)
sarcSentences = sarcSentences[:numExamples]
notSarcSentences = notSarcSentences[:numExamples]

#TODO add test
xTrain = [] #These are the data
yTrain = [] #These are the labels
xTest = []
yTest = []
for i, x in enumerate(sarcSentences):
    if (i < trainTestSplit):
        xTrain.append(x)
        yTrain.append([1, 0])
    else:
        xTest.append(x)
        yTest.append([1,0])
for i, y in enumerate(notSarcSentences):
    if (i < trainTestSplit):
        xTrain.append(x)
        yTrain.append([0, 1])
    else:
        xTest.append(x)
        yTest.append([0, 1])

p =  np.random.permutation(len(xTrain))
xTrain = np.array(xTrain)[p]
yTrain = np.array(yTrain)[p]
p =  np.random.permutation(len(xTest))
xTest = np.array(xTest)[p]
yTest = np.array(yTest)[p]

tokenizer = Tokenizer(num_words=max_features, lower=True, split=" ")
tokenizer.fit_on_texts(xTrain)
train_sequences = sequence.pad_sequences(tokenizer.texts_to_sequences(xTrain) , maxlen = maxSentenceLength) #padding sentences to max length.
test_sequences = sequence.pad_sequences( tokenizer.texts_to_sequences(xTest) , maxlen = maxSentenceLength)
#train_matrix = tokenizer.texts_to_matrix(xTrain) #converting texts to matrices
#test_matrix = tokenizer.texts_to_matrix( test_texts )
print("done tokenizing")

model = sarcasmModel(maxSentenceLength, max_features, embeddingsDim, weights)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_sequences, np.array(yTrain), epochs = 2, batch_size = 32, shuffle=True)
predictions = np.round(np.array(model.predict(test_sequences, batch_size=32)))

actualPredictions = []
for i in predictions:
    if (int(i[0]) == 1):
        actualPredictions.append(1)
    else:
        actualPredictions.append(0)
actualLabels = []
for i in yTest:
    if (int(i[0]) == 1):
        actualLabels.append(1)
    else:
        actualLabels.append(0)

confusionMatrix = sk.metrics.confusion_matrix(actualLabels, actualPredictions)
tn, fp, fn, tp = confusionMatrix.ravel()
precision = (1.0*tp)/(tp+fp)
recall = (1.0*tp)/(tp+fn)
confusionMatrix = confusionMatrix.T
temp = confusionMatrix[0][0]
confusionMatrix[0][0] = confusionMatrix[1][1]
confusionMatrix[1][1] = temp
confusionMatrix = (1.0*confusionMatrix)/confusionMatrix.sum(axis=1)[:,None]
print(confusionMatrix)
print("true negatives: " + str(tn))
print("false positives: " + str(fp))
print("false negatives: " + str(fn))
print("true positives: " + str(tp))
print("precision: " + str(precision))
print("recall: " + str(recall))