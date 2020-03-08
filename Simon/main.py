import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pickle import load
import sys


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


if __name__ == "__main__":


    m,t = sys.argv[1:3]


    print(m)
    print(t)

    model = tf.keras.models.load_model(m)
    tokenizer = load(open(t, 'rb'))

    print(generate_seq(model, tokenizer, 5, sys.argv[3], int(sys.argv[4])))

	


