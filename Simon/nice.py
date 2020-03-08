

# Commented out IPython magic to ensure Python compatibility.
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras.layers import *

import string
import numpy as np
import os
import time
import re


uploaded = open('/scratch/leuven/334/vsc33423/zinnen_met_n_tokens/output_file_12.txt', 'r')

"""###Upload our Glove Model"""


def clean_doc(doc):
	# replace '--' with a space ' '
	doc = doc.replace('--', ' ')
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', string.punctuation)
	tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
	# tokens = [word for word in tokens if word.isalpha()]  <==== dan gaan <num> etc weg
	# make lower case
	tokens = [word.lower() for word in tokens]
	return tokens

tokens = clean_doc(uploaded.read())


length = 5 + 1  #Bigram???
sequences = list()
for i in range(length, len(tokens)):
	# select sequence of tokens
	seq = tokens[i-length:i]
	# convert into a line
	line = ' '.join(seq)
	# store
	sequences.append(line)
print('Total Sequences: %d' % len(sequences))

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(sequences)
sequences = tokenizer.texts_to_sequences(sequences)

vocab_size = len(tokenizer.word_index) + 1


sequences = np.array(tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=6, padding='pre'))

X, y = sequences[:,:-1], sequences[:,-1]
y = tf.keras.utils.to_categorical(y, num_classes=vocab_size)
seq_length = X.shape[1]




def build_model(vocab_size, embedding_dim, rnn_units):
    model = tf.keras.Sequential()
    model.add(
      tf.keras.layers.Embedding(vocab_size,
                            embedding_dim,
                            input_length=5)
      )
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(256))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))
  
    return model

model = build_model(
  vocab_size = vocab_size,
  embedding_dim=150,
  rnn_units=512)


model.summary()


def loss(labels, logits):
  return tf.keras.losses.categorical_crossentropy(labels, logits, from_logits=True)

"""Configure the training procedure using the `tf.keras.Model.compile` method. We'll use `tf.keras.optimizers.Adam` with default arguments and the loss function."""

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')

"""### Configure checkpoints

Use a `tf.keras.callbacks.ModelCheckpoint` to ensure that checkpoints are saved during training:
"""

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints_nlcow'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

"""### Execute the training

To keep training time reasonable, use 10 epochs to train the model. In Colab, set the runtime to GPU for faster training.
"""

EPOCHS=100

history = model.fit(X,y,batch_size=32, epochs=EPOCHS, callbacks=[checkpoint_callback])

from pickle import dump
model.save('modelnl6_2_all.h5')
# save the tokenizer
dump(tokenizer, open('tokenizernl6_2_all.pkl', 'wb'))

# generate a sequence from a language model
def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
	result = list()
	in_text = seed_text
	# generate a fixed number of words
	for _ in range(n_words):
		# encode the text as integer
		encoded = tokenizer.texts_to_sequences([in_text])[0]
		# truncate sequences to a fixed length
		encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
		# predict probabilities for each word
		yhat = model.predict_classes(encoded, verbose=0)
		# map predicted word index to word
		out_word = ''
		for word, index in tokenizer.word_index.items():
			if index == yhat:
				out_word = word
				break
		# append to input
		in_text += ' ' + out_word
		result.append(out_word)
	return ' '.join(result)

from random import randint

seed_text = sequences[randint(0,len(sequences))]

print(seed_text)

# generate new text
from tensorflow.keras.preprocessing.sequence import pad_sequences

generated = generate_seq(model, tokenizer, seq_length, "ik leer over de", 1)
print(generated)

