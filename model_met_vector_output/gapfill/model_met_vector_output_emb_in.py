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
from tensorflow_core.python.keras.layers.embeddings import Embedding
from tensorflow_core.python.keras.layers.recurrent import LSTM, GRU
from tensorflow_core.python.keras.models import Sequential

def make_token_vector_dictionary(v_file):
    print('\nmaking token to vector dictionary...')
    with open(v_file, 'r', errors='ignore') as file:
        index_per_token = dict()
        token_per_index = []
        index_per_token["<PAD>"] = 0
        token_per_index.append("<PAD>")
        index_per_token["<START>"] = 1
        token_per_index.append("<START>")
        index_per_token["<UNK>"] = 2
        token_per_index.append("<UNK>")
        index_per_token["<UNUSED>"] = 3
        token_per_index.append("<UNUSED>")
        i = 4
        vec_per_token = dict()
        for line in file:
            line_array = line.split()
            try:
                index_per_token[line_array[0]] = i
                token_per_index.append(line_array[0])
                i += 1
                vec_per_token[line_array[0]] =  np.asarray(line_array[1:], dtype='float32')
            except ValueError:
                print("VALUE-ERROR!!!!!")
    tok_vector_dimension = len(vec_per_token[next(iter(vec_per_token))])
    vec_per_token["<UNK>"] = [0.0] * tok_vector_dimension
    am_of_token_vectors = len(vec_per_token)
    print(
        'Found {0} token vectors, each with dimension of {1}.'.format(am_of_token_vectors, tok_vector_dimension))
    return vec_per_token, index_per_token, token_per_index, am_of_token_vectors, tok_vector_dimension

def sentences_to_vector_arrays(s_file):
    print('\ncreating vector arrays of the sentences...')
    with open(s_file, 'r', errors='ignore') as file:
        sentences = []  # list of text samples
        indices = []
        for line in file:
            sentences.append(line.strip("\n").split(" "))
            indices.append(line.strip("\n").split(" "))
    am_of_sentences = len(sentences)
    toks_per_sentence = len(sentences[0])
    print('Found {0} sentences, each with a length of {1} tokens.'.format(am_of_sentences, toks_per_sentence))

    for i in range(len(sentences)):
        for j in range(toks_per_sentence):
            try:
                indices[i][j] = index_per_token[sentences[i][j]]
                sentences[i][j] = vector_per_token[sentences[i][j]]
            except KeyError:
                indices[i][j] = 2
                sentences[i][j] = [0.0] * token_vector_dimension
    return sentences, indices, am_of_sentences, toks_per_sentence

def make_train_test_data(gap_index, training_ratio):
    aantal_training = (int(amount_of_sentences * training_ratio))
    x_tn = []
    y_tn = []
    x_tt = []
    y_tt = []

    for i in range(len(sentences)):
        if i < aantal_training:
            x_tn.append(indices[i][:gap_index] + indices[i][(gap_index + 1):])
            y_tn.append(sentences[i][gap_index])

        else:
            x_tt.append(indices[i][:gap_index] + indices[i][(gap_index + 1):])
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

def train_model():
    input_shape = (tokens_per_sentence - 1, token_vector_dimension, 1)

    model = Sequential()
    model.add(Conv2D(tokens_per_sentence * 3,
                     kernel_size=(8, token_vector_dimension),
                     strides=1,
                     activation='relu',
                     input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(tokens_per_sentence * 2,
                    activation='relu'))
    model.add(Dense(int(tokens_per_sentence * 1.5),
                    activation='relu'))
    model.add(Dense(token_vector_dimension,
                    activation='relu'))
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    model.summary()

    epochs = 150
    batch_size = int(5 * amount_of_sentences / epochs)

    model.fit(x=x_train,
              y=y_train,
              epochs=epochs,
              batch_size=batch_size)

    test_scores = model.evaluate(x_test, y_test, verbose=2)
    print('Test loss:', test_scores[0])
    print('Test accuracy:', test_scores[1])
    return model

def train_model_embeddings(vocab_size, embedding_dim, seq_length):
    model = Sequential()
    model.add(Embedding(vocab_size,
                        embedding_dim,
                        weights =[embedding_matrix],
                        input_length = seq_length,
                        trainable = False))
    model.add(LSTM(embedding_dim,
                   return_sequences=True))
    model.add(LSTM(embedding_dim))
    model.add(Dense(token_vector_dimension,
                    activation='relu'))
    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    model.summary()

    epochs = 100
    batch_size = int(5 * amount_of_sentences / epochs)

    model.fit(x=x_train,
              y=y_train,
              epochs=epochs,
              batch_size=batch_size)

    test_scores = model.evaluate(x_test, y_test, verbose=2)
    print('Test loss:', test_scores[0])
    print('Test accuracy:', test_scores[1])
    return model

dirpath = os.getcwd()
#vector_file = "C:/Users/Sander/Documents/KUL/Toegepaste Informatica/2019 - 2020/Afstudeerproject/NLCOW data/glove_f50_1000.txt"
vector_file = dirpath + "/" + sys.argv[1]
#sentence_file = "C:/Users/Sander/Documents/KUL/Toegepaste Informatica/2019 - 2020/Afstudeerproject/NLCOW data/output_file_12_1000.txt"
sentence_file = dirpath + "/" + sys.argv[2]
vector_per_token, index_per_token, token_per_index, amount_of_token_vectors, token_vector_dimension = make_token_vector_dictionary(vector_file)
sentences, indices, amount_of_sentences, tokens_per_sentence = sentences_to_vector_arrays(sentence_file)
gap_index = 4
training_ratio = 0.83
x_train, y_train, x_test, y_test = make_train_test_data(gap_index, training_ratio)

rnn_units = 1024
batch_size = 128

embedding_matrix = np.zeros((len(index_per_token), token_vector_dimension))
for word, i in index_per_token.items():
    embedding_vector = vector_per_token.get(word)
    if embedding_vector is not None:
        #print(embedding_vector)
        embedding_matrix[i] = embedding_vector
seq_length = x_train.shape[1]
model = train_model_embeddings((len(index_per_token)), token_vector_dimension, seq_length)

#########################################################################################################################

model.save('lekkerpik.h5')