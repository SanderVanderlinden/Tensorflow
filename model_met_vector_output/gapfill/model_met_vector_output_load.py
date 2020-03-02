import numpy as np
import tensorflow as tf
from tensorflow import keras
from scipy import spatial

'''
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
    vec_per_token["<UNK>"] = [[0.000001]] * tok_vector_dimension
    am_of_token_vectors = len(vec_per_token)
    print(
        'Found {0} token vectors, each with dimension of {1}.'.format(am_of_token_vectors, tok_vector_dimension))
    return vec_per_token, am_of_token_vectors, tok_vector_dimension
'''

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

def sentence_to_vector_array(sentence, gap_index):
    sentence = sentence.strip("\n").split(" ")
    tokens_in_sentence = len(sentence)

    for i in range(tokens_in_sentence):
        try:
            sentence[i] = vector_per_token[sentence[i]]
        except KeyError:
            sentence[i] = vector_per_token["<UNK>"]

    return [sentence[:gap_index] + sentence[gap_index + 1:]]

def sentence_to_index_array(sentence, gap_index):
    sentence = sentence.strip("\n").split(" ")
    tokens_in_sentence = len(sentence)

    for i in range(tokens_in_sentence):
        try:
            sentence[i] = index_per_token[sentence[i]]
        except KeyError:
            sentence[i] = index_per_token["<UNK>"]

    return [sentence[:gap_index] + sentence[gap_index + 1:]]


def gap_fill_vector(zin, gap_index):
    zin_vector = sentence_to_vector_array(zin, gap_index)
    zin_vector_np = np.asarray(zin_vector)
    return model.predict(zin_vector_np).tolist()

def gap_fill_index(zin, gap_index):
    zin_index = sentence_to_index_array(zin, gap_index)
    zin_index_np = np.asarray(zin_index)
    return model.predict(zin_index_np).tolist()

def consine_distance(vector1, vector2):
    return spatial.distance.cosine(vector1, vector2)

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

def beste_matches(zin, gap_index, aantal_matches):
    print("\n\n" + " ".join(zin.split(" ")[:gap_index] + ["<GAP>"] + zin.split(" ")[gap_index + 1:]))
    verwachte_token_vector = (gap_fill_vector(zin, gap_index))
    verwachte_token_vector = verwachte_token_vector[0]
    # print("Shapes of output_vector: ", end="")
    # print(len(verwachte_token_vector))
    return most_similar_word(verwachte_token_vector, aantal_matches)

def beste_matches_index(zin, gap_index, aantal_matches):
    print("\n\n" + " ".join(zin.split(" ")[:gap_index] + ["<GAP>"] + zin.split(" ")[gap_index + 1:]))
    verwachte_token_vector = (gap_fill_index(zin, gap_index))
    verwachte_token_vector = verwachte_token_vector[0]
    print("Shapes of output_vector: ", end="")
    print(verwachte_token_vector)
    return most_similar_word(verwachte_token_vector, aantal_matches)

def print_beste_match(zin, woord1, woord2, woord3):
    verwachte_token_vector = (gap_fill_vector(zin, gap_index))
    verwachte_token_vector = verwachte_token_vector[0]
    print("\n\n" + " ".join(zin.split(" ")[:gap_index] + ["<GAP>"] + zin.split(" ")[gap_index + 1:]))
    print("mogelijke woorden: {0} | {1} | {2}".format(woord1, woord2, woord3))
    mapping = dict()
    mapping[woord1] = consine_distance(verwachte_token_vector, vector_per_token[woord1])
    mapping[woord2] = consine_distance(verwachte_token_vector, vector_per_token[woord2])
    mapping[woord3] = consine_distance(verwachte_token_vector, vector_per_token[woord3])
    print(mapping)
    oplossing = ""
    if mapping[woord1] < mapping[woord2] and mapping[woord1] < mapping[woord3]:
        oplossing = woord1
    elif mapping[woord2] < mapping[woord3]:
        oplossing = woord2
    else:
        oplossing = woord3
    print("de beste match is: {0}!".format(oplossing))

def print_beste_match_index(zin, woord1, woord2, woord3):
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
    print("de beste match is: {0}!".format(oplossing))

model = keras.models.load_model('lekkerpik.h5')

vector_file = "C:/Users/Sander/Documents/KUL/Toegepaste Informatica/2019 - 2020/Afstudeerproject/NLCOW data/glove_f50_1000.txt"
vector_per_token, index_per_token, token_per_index, amount_of_token_vectors, token_vector_dimension = make_token_vector_dictionary(vector_file)
gap_index = 4 #Het model is getraind voor index 5
beste_aantal_matches = 10
zin = "de heuvelachtige omgeving kenmerkt zich <GAP> kleine landerijen afgewisseld met"
print(beste_matches_index(zin, gap_index, beste_aantal_matches))
zin = "zo kun je je horloge altijd aanpassen aan je eigen"
print(beste_matches_index(zin, gap_index, beste_aantal_matches))
zin = "we rijden verder door een maanlandschap en stijgen verder er"
print(beste_matches_index(zin, gap_index, beste_aantal_matches))
zin = "vanavond gaan lopen en morgen heb ik hopelijk terug goed"
print(beste_matches_index(zin, gap_index, beste_aantal_matches))
zin = "uiteindelijk begon er toch wat orde in de chaos te"
print(beste_matches_index(zin, gap_index, beste_aantal_matches))
zin = "investeer het uitgespaarde budget liever in grotere ramen in de"
print(beste_matches_index(zin, gap_index, beste_aantal_matches))
zin = "groene groene groene groene groene groene groene groene groene groene"
print(beste_matches_index(zin, gap_index, beste_aantal_matches))
zin = "waarom waarom waarom waarom waarom waarom waarom waarom waarom waarom"
print(beste_matches_index(zin, gap_index, beste_aantal_matches))

zin = "Jan eet vandaag een dikke <GAP> en hij vond hem"
woord1 = "appel"
woord2 = "spaghetti"
woord3 = "brood"
print_beste_match_index(zin, woord1, woord2, woord3)

zin = "investeer morgen onmiddellijk het uitgespaarde budget liever in grotere ramen"
woord1 = "budget"
woord2 = "appel"
woord3 = "schieten"
print_beste_match_index(zin, woord1, woord2, woord3)

zin = "Jan eet vandaag een dikke <GAP> en hij vond hem"
woord1 = "appel"
woord2 = "terwijl"
woord3 = "lopen"
print_beste_match_index(zin, woord1, woord2, woord3)

zin = "waarom waarom waarom waarom waarom waarom waarom waarom waarom waarom"
woord1 = "waarom"
woord2 = "terwijl"
woord3 = "lopen"
print_beste_match_index(zin, woord1, woord2, woord3)