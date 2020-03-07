import numpy as np
from sklearn.cluster import k_means
import os
import sys
from flask import Flask, render_template, redirect, url_for,request
from flask import make_response
from flask_cors import CORS
import warnings 
import fasttext
import os
import tensorflow as tf
from tensorflow import keras
from scipy.spatial import distance
from scipy import spatial
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pickle import load
#import gensim.models.wrappers.fasttext
warnings.filterwarnings(action = 'ignore')
app = Flask(__name__) #Dit zorgt ervoor dat deze file een flask server wordt
CORS(app)

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
                    print(index_per_token[line_array[0]])
                    print(i)
                except KeyError:
                    try:
                        index_per_token[line_array[0]] = i
                        token_per_index.append(line_array[0])
                        i += 1
                        vec_per_token[line_array[0]] = np.asarray(line_array[1:], dtype='float32')
                    except ValueError:
                        print("VALUE-ERROR!!!!!")
        tok_vector_dimension = len(vec_per_token[next(iter(vec_per_token))])
        vec_per_token["<UNK>"] = [0.0] * tok_vector_dimension
        am_of_token_vectors = len(vec_per_token)
        print('Found {0} token vectors, each with dimension of {1}.'.format(am_of_token_vectors, tok_vector_dimension))
        return vec_per_token, index_per_token, token_per_index, am_of_token_vectors, tok_vector_dimension



def sentence_to_index_array(sentence, gap_index):
    vector_file = "glove_f150.txt"
    vector_per_token, index_per_token, token_per_index, amount_of_token_vectors, token_vector_dimension = make_token_vector_dictionary(vector_file)
    sentence = sentence.strip("\n").split(" ")
    tokens_in_sentence = len(sentence)

    for i in range(tokens_in_sentence):
        try:
            sentence[i] = index_per_token[sentence[i]]
        except KeyError:
            sentence[i] = index_per_token["<UNK>"]

    return [sentence[:gap_index] + sentence[gap_index + 1:]]



def gap_fill_index(zin, gap_index):
    zin_index = sentence_to_index_array(zin, gap_index)
    zin_index_np = np.asarray(zin_index)
    model = tf.keras.models.load_model("testModel3.h5", compile=False)
    return model.predict(zin_index_np).tolist()

def consine_distance(vector1, vector2):
    return distance.cosine(vector1, vector2)

def most_similar_word(vector1, top):
    print('Calculating most similar words to missing vector...')
    word_to_cos_dist = dict()
    for token, vector in vector_per_token.items():
        try:
            vector = vector.tolist()
        except:
            None
        word_to_cos_dist[token] = consine_distance(vector1, vector)
    sorted_dict = sorted(word_to_cos_dist.items(), key=lambda x: x[1])
    return sorted_dict[:top] + sorted_dict[-top:]


def beste_matches_index(zin, gap_index, aantal_matches):
    print("\n\n" + " ".join(zin.split(" ")[:gap_index] + ["<GAP>"] + zin.split(" ")[gap_index + 1:]))
    verwachte_token_vector = (gap_fill_index(zin, gap_index))
    verwachte_token_vector = verwachte_token_vector[0]
    print("Shapes of output_vector: ", end="")
    print(verwachte_token_vector)
    return most_similar_word(verwachte_token_vector, aantal_matches)


def print_beste_match_index(zin, woord1, woord2, woord3):
    gap_index = 4 #Het model is getraind voor index 5
    beste_aantal_matches = 10
    vector_file = "glove_f150.txt"
    vector_per_token, index_per_token, token_per_index, amount_of_token_vectors, token_vector_dimension = make_token_vector_dictionary(vector_file)
    verwachte_token_vector = (gap_fill_index(zin, gap_index))
    
    verwachte_token_vector = verwachte_token_vector[0]
    
    print("\n\n" + " ".join(zin.split(" ")[:gap_index] + ["<GAP>"] + zin.split(" ")[gap_index + 1:]))
    print("mogelijke woorden: {0} | {1} | {2}".format(woord1, woord2, woord3))
    mapping = dict()
    try:
        mapping[woord1] = consine_distance(verwachte_token_vector, vector_per_token[woord1])
    except:
        print("'{0}' is geen gekende token.".format(woord1))
        mapping[woord1] = 2
    try:
        mapping[woord2] = consine_distance(verwachte_token_vector, vector_per_token[woord2])
    except:
        print("'{0}' is geen gekende token.".format(woord2))
        mapping[woord2] = 2
    try:
        mapping[woord3] = consine_distance(verwachte_token_vector, vector_per_token[woord3])
    except:
        print("'{0}' is geen gekende token.".format(woord3))
        mapping[woord3] = 2
    print(mapping)
    oplossing = ""
    if mapping[woord1] < mapping[woord2] and mapping[woord1] < mapping[woord3]:
        oplossing = woord1
    elif mapping[woord2] < mapping[woord3]:
        oplossing = woord2
    else:
        oplossing = woord3
    
    return "de beste match is: {0}!".format(oplossing)

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

class Glove():
    """
    Class for training, using and evaluating glove described in https://code.google.com/p/word2vec/
    The model has three methods:
    1.consine_distance: Calculating the two different word vectors' consine distance
    2.MostSimilarWord: Finding the topN closest words of the given word
    3.clustering: Using sklearn's kmeans to clustering
    """
   
    def __init__(self, fUrl):
        """
        load model(no-binary model)
        """
        with open(fUrl, errors='ignore') as f:
            self.word_dic = {line.split()[0]:np.asarray(line.split()[1:], dtype='float') for line in f}

    def consine_distance(self, word1, word2):
        return np.dot(self.word_dic[word1],self.word_dic[word2])\
        /(np.linalg.norm(self.word_dic[word1])* np.linalg.norm(self.word_dic[word2]))

    def most_similar_word(self, word,top):
        return sorted({word2:self.consine_distance(word, word2) for word2 in self.word_dic.keys()}.items(), key=lambda x:x[1], reverse=True)[1:top+1]

    def clustering(self, cluster_size):
        X = np.array(list(self.word_dic.items()))
        return k_means(X, n_clusters= cluster_size, init= "k-means++")

    



@app.route('/glove')
def func():
    #Haal de parameters word, amount, dimension uit het get request
    word = request.args.get('word')
    amount = request.args.get('amount')
    dimension = request.args.get('dimension')
    int_amount = int(amount)
    #Gebruik de glove file met de meegegeven dimensie
    file = "glove_f{0}.txt".format(dimension)
    print(file)
    #Zet de file om naar een Glove model
    model = Glove(file)
    #Zoek het x-aantal woorden die het meest lijken op het meegegeven woord
    result = model.most_similar_word(word, int_amount)
    return str(result)

@app.route('/w2v')
def w2v():
    #Haal de parameters word, amount, dimension uit het get request
    word = request.args.get('word')
    amount = request.args.get('amount')
    int_amount = int(amount)
    dimension = request.args.get('dimension')
    #Gebruik de fasttext file met de meegegeven dimensie
    file = "ft{0}.bin".format(dimension)
    print(file)
    #Zet de file op naar een fasttext model
    model = fasttext.load_model(file)
    result = model.get_nearest_neighbors(word, k=int_amount)

    return str(result)

@app.route('/similarityGlove')
def relation():
    #Haal de 2 woorden op uit de GET request
    word1 = request.args.get('w1')
    word2 = request.args.get('w2')

    #Zet file om naar model
    model = Glove("glove_f150.txt")

    #Bereken cosinus afstand tussen de 2 meegegeven woorden
    return (str(model.consine_distance(word1, word2)))

@app.route('/similarityW2V')
def rel():
    #Haal de 2 woorden op uit de GET request
    word1 = request.args.get('w1')
    word2 = request.args.get('w2')

    #Zet file om naar model
    model = fasttext.load_model("ft150.bin")
    #Zet woord om naar zijn woordvector om er de cosinusafstand voor te kunnen berekenen
    vec1 = model[word1]
    vec2 = model[word2]

    #Bereken cosinus afstand tussen de 2 meegegeven woorden
    result = distance.cosine(vec1, vec2)
    r = 1 - result

    return (str(r))

@app.route('/predictNWords')
def predict():
    #Maak model van file
    model = tf.keras.models.load_model('modelnl6_2.h5', compile=False)
    #Gebruik tokenizer file om de tokens van elk woord te vinden
    tokenizer = load(open('tokenizernl6_2.pkl', 'rb'))
    #Haal parameters uit GET request
    tekst = request.args.get('tekst')
    aantal = request.args.get('aantal')
    #Stuur resultaat terug van beste aantal woorden
    return str(generate_seq(model, tokenizer, 5, tekst, int(aantal)))



@app.route('/gap')
def gap():
    '''
    dit script test een getraind model
    gebruik dit script als volgt:
    waarbij glove_150.txt dezelfde file is als de vectorfile waarmee het model getraind is.
    '''
    model = tf.keras.models.load_model("testModel3.h5", compile=False)

    dirpath = os.getcwd()
    vector_file = "glove_f150.txt"
    
    gap_index = 4 #Het model is getraind voor index 5 in te vullen
    beste_aantal_matches = 10


    zin = request.args.get('zin')
    woord1 = request.args.get('w1')
    woord2 = request.args.get('w2')
    woord3 = request.args.get('w3')
    #Geef het woord terug die de beste match is om in de zin te plaatsen, kies woord tussen de 3 meegegeven woorden
    result = print_beste_match_index(zin, woord1, woord2, woord3)
    
    return str(result)




if __name__ == "__main__":
    #starttime = datetime.datetime.now()
    #model = Glove("vectors.txt") #load model
    #print(model.most_similar_word(sys.argv[1]))
    app.run(threaded=False, processes=4)
    

    
    #endtime = datetime.datetime.now()
    #print 'Time:', (endtime - starttime).seconds,'s'