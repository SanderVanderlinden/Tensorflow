from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow.keras.layers as layers
from tensorflow_core.python.keras.layers.convolutional import Conv2D
from tensorflow_core.python.keras.layers.core import Flatten, Dense
from tensorflow_core.python.keras.models import Sequential

vector_file = "C:/Users/Sander/Documents/KUL/Toegepaste Informatica/2019 - 2020/Afstudeerproject/NLCOW data/glove50_1000.txt"
sentence_file = "C:/Users/Sander/Documents/KUL/Toegepaste Informatica/2019 - 2020/Afstudeerproject/NLCOW data/output_file_12_25000.txt"

print('\nIndexing token vectors.')
with open(vector_file, 'r', errors='ignore') as input_file:
    vector_per_token = dict()
    token_indices = dict()
    i = 0
    for line in input_file:
        line_array = line.split()
        try:
            vector_per_token[line_array[0]] = [[float(x)] for x in line_array[1:]]
        except ValueError:
            print("VALUE-ERROR!!!!!")
        token_indices[line_array[0]] = i
        i += 1

#print(vector_per_token)

amount_of_token_vectors = len(vector_per_token)
token_vector_dimension = len(vector_per_token[next(iter(vector_per_token))])
print('Found {0} token vectors, each with dimension of {1}.'.format(amount_of_token_vectors, token_vector_dimension))

print('\nProcessing text dataset')
with open(sentence_file, 'r', errors='ignore') as f:
    sentences = []  # list of text samples
    for line in f:
        sentences.append(line.strip("\n").split(" "))

amount_of_sentences = len(sentences)
tokens_per_sentence = len(sentences[0])
print('Found {0} sentences, each with a length of {1} tokens.'.format(amount_of_sentences, tokens_per_sentence))

gap_index = 5
y_values = []
x_train = []
y_train = []
x_test = []
y_test = []

for sentence in sentences:
    for i in range(tokens_per_sentence):
        if i == gap_index:
            try:
                index = (token_indices[sentence[gap_index]])
            except KeyError:
                index = -1
            output_vector = [0] * amount_of_token_vectors
            if index != -1:
                output_vector[index] = 1
            y_values.append(output_vector)

        try:
            sentence[i] = vector_per_token[sentence[i]]
        except KeyError:
            sentence[i] = [[0.0]] * token_vector_dimension

training_ratio = 0.83
aantal_training = (int(len(sentences) * training_ratio))

'''for i in range(len(sentences)):
    for j in range(len(sentences[i])):
        for k in range(len(sentences[i][j])):
            sentences[i][j][k]= float(sentences[i][j][k])'''

for i in range(len(sentences)):
    if i < aantal_training:
        x_train.append(sentences[i][:gap_index] + sentences[i][(gap_index + 1):])
        y_train.append(y_values[i])

    else:
        x_test.append(sentences[i][:gap_index] + sentences[i][(gap_index + 1):])
        y_test.append(y_values[i])


x_train_np = np.asarray(x_train)
x_test_np = np.asarray(x_test)
y_train_np = np.asarray(y_train)
y_test_np = np.asarray(y_test)

#########################################################################################################################

input_shape = (tokens_per_sentence - 1, token_vector_dimension, 1)

model = Sequential()
model.add(Conv2D(int(amount_of_sentences ** 0.4),
                 kernel_size = (5, token_vector_dimension),
                 strides = 1,
                 activation = 'relu',
                 input_shape = input_shape))
'''model.add(Conv2D(int(amount_of_sentences ** 0.4),
                 kernel_size = (1, token_vector_dimension),
                 strides = 1,
                 activation = 'relu',
                 input_shape = input_shape))'''
model.add(Flatten())
model.add(Dense(int(amount_of_token_vectors ** 0.4),
                activation='relu'))
model.add(Dense(int(amount_of_token_vectors ** 0.7),
                activation='relu'))
model.add(Dense(amount_of_token_vectors,
                activation='softmax'))
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])
model.summary()
import time
time.sleep(1)

#####################

print("x_train_np.shape: " + str(x_train_np.shape))
print("y_train_np.shape: " + str(y_train_np.shape))
print(x_train_np[0])
model.fit(x = x_train_np,
          y = y_train_np,
          epochs = 5,
          batch_size = amount_of_sentences)

test_scores = model.evaluate(x_test, y_test, verbose=2)
print('Test loss:', test_scores[0])
print('Test accuracy:', test_scores[1])

model.save('model.h5')