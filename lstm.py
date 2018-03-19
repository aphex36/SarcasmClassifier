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
import os
import fnmatch
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
from keras.callbacks import EarlyStopping

sarcSentences = []
notSarcSentences = []
punctuation = ['.', ',', ';', ':', '!', '?', '\"', '\'']
counter = 0
trainTestSplit = 0
numExamples = int(sys.argv[1])/2
maxSentenceLength = 30
embeddingsDim = 200

numInvalid = dict()
numInvalid['invalid'] = 0

'''
CSV format of this file is weird, so had to do some manual extraction to get word embeddings
embedding = turns word to a vector
'''
def initializeEmbeddings():

    for file in os.listdir("."):
        counter = 1
        if not fnmatch.fnmatch(file, "*train-balanced-sarcasm_*"):
            continue
        with open(file) as infile:
            for line in infile:
                if line[0] == 'l':
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
        infile.close()
    embeddings = Word2Vec(sarcSentences, size = embeddingsDim, min_count=1)
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
    X = Dropout(0.25)(X)
    # Propagate X trough another LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a single hidden state, not a batch of sequences.
    X = LSTM(8)(X)
    # Add dropout with a probability of 0.5
    X = Dropout(0.25)(X)
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
'''
np.random.shuffle(sarcSentences)
np.random.shuffle(notSarcSentences)
sarcSentences = sarcSentences[:numExamples]
notSarcSentences = notSarcSentences[:numExamples]
'''
#TODO add test
xTrain = [] #These are the data
yTrain = [] #These are the labels
xTest = []
yTest = []
#print(notSarcSentences)
for i, x in enumerate(sarcSentences):
    if (i < trainTestSplit):
        #print("train sarc")
        #print(x)
        xTrain.append(x)
        yTrain.append([1, 0])
    else:
        #print("test sarc")
        #print(x)
        xTest.append(x)
        yTest.append([1,0])


for i, y in enumerate(notSarcSentences):
    if (i < trainTestSplit):
        xTrain.append(y)
        yTrain.append([0, 1])
    else:
        xTest.append(y)
        yTest.append([0, 1])
del sarcSentences
del notSarcSentences

'''
p =  np.random.permutation(len(xTrain))
xTrain = np.array(xTrain)[p]
yTrain = np.array(yTrain)[p]
p =  np.random.permutation(len(xTest))
xTest = np.array(xTest)[p]
yTest = np.array(yTest)[p]
'''

tokenizer = Tokenizer(num_words=max_features, lower=True, split=" ")
tokenizer.fit_on_texts(xTrain)
train_sequences = sequence.pad_sequences(tokenizer.texts_to_sequences(xTrain) , maxlen = maxSentenceLength) #padding sentences to max length.
test_sequences = sequence.pad_sequences( tokenizer.texts_to_sequences(xTest) , maxlen = maxSentenceLength)

#train_matrix = tokenizer.texts_to_matrix(xTrain) #converting texts to matrices
#test_matrix = tokenizer.texts_to_matrix( test_texts )
print("done tokenizing")
model = sarcasmModel(maxSentenceLength, max_features, embeddingsDim, weights)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
earlystop = EarlyStopping(monitor='acc', min_delta=0.1, patience=0, \
                          verbose=0, mode='auto')

model.fit(train_sequences, np.array(yTrain), callbacks=[earlystop], epochs = 20, batch_size = 32, shuffle=True)
print(model.metrics_names)
print(model.evaluate(test_sequences, np.array(yTest)))
predictions = np.round(np.array(model.predict(test_sequences, batch_size=32)))

actualPredictions = []
actualLabels = []
for i in predictions:
    if (int(i[0]) == 1):
        actualPredictions.append(1)
    else:
        actualPredictions.append(0)
for i in yTest:
    if (int(i[0]) == 1):
        actualLabels.append(1)
    else:
        actualLabels.append(0)
tn = 0
fp = 0
fn = 0
tp = 0
numCorrect = 0
for i in range(len(actualPredictions)):
    if actualPredictions[i] == actualLabels[i]:
        if actualPredictions[i] == 0:
            tp += 1
        else:
            tn += 1
        numCorrect += 1
    else:
        if actualPredictions[i] == 0:
            fn += 1
        else:
            fp += 1
print("tp " + str(tp))
print("tn " + str(tn))
print("fp " + str(fp))
print("fn " + str(fn))

print("acc: " + str((1.0*numCorrect)/(len(actualPredictions))))
#tn, fp, fn, tp = confusionMatrix.ravel()
precision = (1.0*tp)/(tp+fp)
recall = (1.0*tp)/(tp+fn)
confusionMatrix = np.zeros((2,2))
temp = confusionMatrix[0][0]
confusionMatrix[0][0] = recall
confusionMatrix[0][1] = 1-recall
confusionMatrix[1][0] = float(tn)/(tn+fp)
confusionMatrix[1][1] = 1-confusionMatrix[1][0]
print(confusionMatrix)
print("precision: " + str(precision))
print("recall: " + str(recall))
