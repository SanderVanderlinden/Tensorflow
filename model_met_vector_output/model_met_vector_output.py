'''
dit script traint een model en slaat het op
gebruik dit script als volgt:
python vector_file.txt sentence_file.txt output_model.h5
'''

from __future__ import print_function

import os
import sys
import numpy as np
from tensorflow_core.python.keras.layers.convolutional import Conv2D
from tensorflow_core.python.keras.layers.core import Flatten, Dense
from tensorflow_core.python.keras.models import Sequential

def make_token_vector_dictionary(v_file):
    print('\nmaking token to vector dictionary...')
    with open(v_file, 'r', errors='ignore') as file:
        vec_per_token = dict()
        for line in file:
            line_array = line.split()
            try:
                vec_per_token[line_array[0]] = [[float(x)] for x in line_array[1:]]
                #vec_per_token[line_array[0]] = [float(x) for x in line_array[1:]]
            except ValueError:
                print("VALUE-ERROR!!!!!")
    tok_vector_dimension = len(vec_per_token[next(iter(vec_per_token))])
    vec_per_token["<UNK>"] = [0.0] * tok_vector_dimension
    am_of_token_vectors = len(vec_per_token)
    print(
        'Found {0} token vectors, each with dimension of {1}.'.format(am_of_token_vectors, tok_vector_dimension))
    return vec_per_token, am_of_token_vectors, tok_vector_dimension

def sentences_to_vector_arrays(s_file):
    print('\ncreating vector arrays of the sentences...')
    with open(s_file, 'r', errors='ignore') as file:
        sentences = []  # list of text samples
        for line in file:
            sentences.append(line.strip("\n").split(" "))
    am_of_sentences = len(sentences)
    toks_per_sentence = len(sentences[0])
    print('Found {0} sentences, each with a length of {1} tokens.'.format(am_of_sentences, toks_per_sentence))

    for sentence in sentences:
        for i in range(toks_per_sentence):
            try:
                sentence[i] = vector_per_token[sentence[i]]
            except KeyError:
                sentence[i] = [[0.0]] * token_vector_dimension
                #sentence[i] = [0.0] * token_vector_dimension
    return sentences, am_of_sentences, toks_per_sentence

def make_train_test_data(gap_index, training_ratio):
    aantal_training = (int(amount_of_sentences * training_ratio))
    x_tn = []
    y_tn = []
    x_tt = []
    y_tt = []

    for i in range(len(sentences)):
        if i < aantal_training:
            x_tn.append(sentences[i][:gap_index] + sentences[i][(gap_index + 1):])
            y_tn.append(sentences[i][gap_index])

        else:
            x_tt.append(sentences[i][:gap_index] + sentences[i][(gap_index + 1):])
            y_tt.append(sentences[i][gap_index])

    x_train_np = np.asarray(x_tn)
    y_train_np = np.asarray(y_tn)
    x_test_np = np.asarray(x_tt)
    y_test_np = np.asarray(y_tt)

    print("Shapes of x_train:", end = "")
    print(x_train_np.shape)
    print("Shapes of y_train:", end = "")
    print(y_train_np.shape)
    print("Shapes of x_test:", end = "")
    print(x_test_np.shape)
    print("Shapes of y_test:", end = "")
    print(y_test_np.shape)
    return x_train_np, y_train_np, x_test_np, y_test_np

dirpath = os.getcwd()
vector_file = "C:/Users/Sander/Documents/KUL/Toegepaste Informatica/2019 - 2020/Afstudeerproject/NLCOW data/glove_f50_25000.txt"
#vector_file = dirpath + "/" + sys.argv[1]
sentence_file = "C:/Users/Sander/Documents/KUL/Toegepaste Informatica/2019 - 2020/Afstudeerproject/NLCOW data/output_file_12_250000.txt"
#sentence_file = dirpath + "/" + sys.argv[2]
vector_per_token, amount_of_token_vectors, token_vector_dimension = make_token_vector_dictionary(vector_file)
sentences, amount_of_sentences, tokens_per_sentence = sentences_to_vector_arrays(sentence_file)
gap_index = 5
training_ratio = 0.83
x_train, y_train, x_test, y_test = make_train_test_data(gap_index, training_ratio)

#########################################################################################################################


input_shape = (tokens_per_sentence - 1, token_vector_dimension, 1)

model = Sequential()
model.add(Conv2D(int(amount_of_sentences ** 0.4),
                 kernel_size = (1, token_vector_dimension),
                 strides = 1,
                 activation = 'relu',
                 input_shape = input_shape))
model.add(Flatten())
model.add(Dense(int(amount_of_token_vectors ** 0.4),
                activation='relu'))
model.add(Dense(int(amount_of_token_vectors ** 0.7),
                activation='relu'))
model.add(Dense(token_vector_dimension,
                activation='relu'))
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])
model.summary()
import time
time.sleep(1)

epochs = 150
batch_size = int(2 * amount_of_sentences / epochs)

model.fit(x = x_train,
          y = y_train,
          epochs = epochs,
          batch_size = batch_size)

test_scores = model.evaluate(x_test, y_test, verbose=2)
print('Test loss:', test_scores[0])
print('Test accuracy:', test_scores[1])

model.save('model_met_vector_output.h5')