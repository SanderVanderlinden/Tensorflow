import os
import copy
import tensorflow as tf
import numpy as np


def make_input_output(sentence_file, word_length_input, word_length_output):
    input_words = []
    output_words = []

    print('\nSplitting the sentences')
    with open(sentence_file, 'r', errors='ignore') as input_file:
        for line in input_file:
            words = line.strip().split(" ")
            for i in range(len(words) - word_length_input - word_length_output + 1):
                input_words.append(words[i:i + word_length_input])
                output_words.append(words[i + word_length_input: i + word_length_input + word_length_output])
                #input_output_words.append([words[i:i + word_length_input], )
    return input_words, output_words


def make_indices(vector_file):
    print('\nIndexing token vectors.')
    with open(vector_file, 'r', errors='ignore') as input_file:
        token_indices = dict()
        token_list = []
        token_indices["<PAD>"] = 0
        token_list.append("<PAD>")
        token_indices["<START>"] = 1
        token_list.append("<START>")
        token_indices["<UNK>"] = 2
        token_list.append("<UNK>")
        token_indices["<UNUSED>"] = 3
        token_list.append("<UNUSED>")
        i = 4
        for line in input_file:
            line_array = line.split()
            token_indices[line_array[0]] = i
            token_list.append(line_array[0])
            i += 1
    return token_indices, token_list


def swap_words_with_indices(words):
    word_indices = copy.deepcopy(words)
    print('\nswapping words with indices')
    for i in range(len(words)):
        for j in range(len(words[i])):
            try:
                word_indices[i][j] = token_indices[words[i][j]]
            except KeyError:
                word_indices[i][j] = 2
    return word_indices


sentence_file = "C:/Users/Sander/Documents/KUL/Toegepaste Informatica/2019 - 2020/Afstudeerproject/NLCOW data/parsed12.txt"
word_length_input = 3
word_length_output = 3

input_words, output_words = make_input_output(sentence_file, word_length_input, word_length_output)
print(input_words)
print(output_words)

vector_file = "C:/Users/Sander/Documents/KUL/Toegepaste Informatica/2019 - 2020/Afstudeerproject/NLCOW data/glove50_1000.txt"

token_indices, token_list = make_indices(vector_file)
print(token_indices)
print(token_list)

input_words_indices, output_words_indices = swap_words_with_indices(input_words), swap_words_with_indices(output_words)
print(input_words_indices)
print(output_words_indices)

################################################

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

word_list_length = len(token_list)
with open(vector_file, 'r', errors='ignore') as input_file:
    token_vector_dimension = len(input_file.readline().split(" ")) - 1

rnn_units = 1024
batch_size = 128

amount_of_batches = int(len(input_words_indices)/batch_size)

input_words_indices = input_words_indices[:amount_of_batches * batch_size]
output_words_indices = output_words_indices[:amount_of_batches * batch_size]

input_words_indices_np, output_words_indices_np = np.asarray(input_words_indices), np.asarray(output_words_indices)

print(input_words_indices_np.shape)
print(output_words_indices_np.shape)

def loss(input, target):
    return tf.keras.losses.sparse_categorical_crossentropy(input, target, from_logits=True)


model = build_model(word_list_length, token_vector_dimension, rnn_units, batch_size)

model.compile(optimizer = "adam",
              loss = loss)
model.summary()

checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only = True)

model.fit(input_words_indices_np,
          output_words_indices_np,
          epochs = 20,
          callbacks = [checkpoint_callback])
model.save('modelBoys.h5')
model_json = model.to_json()

with open("model.json", "w") as json_file:
    json_file.write(model_json)

score = model.evaluate(input_words_indices_np, output_words_indices_np, verbose = 0)
print(score)

model.save_weights("modelBoysWeights.h5")
