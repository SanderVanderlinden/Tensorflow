from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow_core.python.keras.layers.convolutional import Conv1D
from tensorflow_core.python.keras.layers.core import Flatten, Dense
from tensorflow_core.python.keras.layers.embeddings import Embedding
from tensorflow_core.python.keras.layers.pooling import MaxPooling1D
from tensorflow_core.python.keras.models import Sequential

vector_file = "C:/Users/Sander/Documents/KUL/Toegepaste Informatica/2019 - 2020/Afstudeerproject/NLCOW data/glove50_100.txt"
sentence_file = "C:/Users/Sander/Documents/KUL/Toegepaste Informatica/2019 - 2020/Afstudeerproject/NLCOW data/parsed12.txt"

print('\nIndexing word vectors.')
with open(vector_file, 'r', errors='ignore') as input_file:
    vector_per_token = dict()
    for line in input_file:
        line_array = line.split()
        vector_per_token[line_array[0]] = line_array[1:]

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

for sentence in sentences:
    for i in range(tokens_per_sentence):
        try:
            sentence[i] = vector_per_token[sentence[i]]
        except KeyError:
            sentence[i] = [0.0] * token_vector_dimension


training_ratio = 0.83
aantal_training = (int(len(sentences) * training_ratio))
gap_index = 5

x_train = []
y_train = []
x_test = []
y_test = []

for i in range(len(sentences)):
    if i < aantal_training:
        x_train.append(sentences[i][:gap_index] + sentences[i][(gap_index + 1):])
        y_train.append(sentences[i][gap_index])

    else:
        x_test.append(sentences[i][:gap_index] + sentences[i][(gap_index + 1):])
        y_test.append(sentences[i][gap_index])


x_train_np = np.asarray(x_train)
x_test_np = np.asarray(x_test)
y_train_np = np.asarray(y_train)
y_test_np = np.asarray(y_test)

#########################################################################################################################

input_shape = (tokens_per_sentence - 1, token_vector_dimension)

model = Sequential()
model.add(Conv1D(int((amount_of_sentences)**0.4), kernel_size = 7, strides = 1,
                 activation = 'relu',
                 input_shape = input_shape))
model.add(MaxPooling1D(pool_size = 2, strides = 2))
model.add(Dense(int((amount_of_sentences)**0.7), activation='relu'))
model.add(Dense(amount_of_token_vectors, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

#####################

model.fit(x = x_train_np,
          y = y_train_np,
          epochs = 5,
          batch_size = amount_of_sentences)

test_scores = model.evaluate(x_test, y_test, verbose=2)
print('Test loss:', test_scores[0])
print('Test accuracy:', test_scores[1])











'''
inputs = keras.Input(shape=(1, (tokens_per_sentence - 1) * token_vector_dimension), name='img')
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(10)(x)

model = keras.Model(inputs=inputs, outputs=outputs, name='mnist_model')

model.summary()

x_train_np = x_train_np.reshape(len(x_train_np), (tokens_per_sentence - 1) * token_vector_dimension).astype('float32')
x_test_np = x_test_np.reshape(len(x_test_np), (tokens_per_sentence - 1) * token_vector_dimension).astype('float32')

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy'])


history = model.fit(x_train_np, y_train_np,
                    epochs=2,
                    batch_size=128)
test_scores = model.evaluate(x_test, y_test, verbose=2)
print('Test loss:', test_scores[0])
print('Test accuracy:', test_scores[1])
'''