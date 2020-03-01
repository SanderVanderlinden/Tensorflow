import numpy as np
import tensorflow as tf
from tensorflow import keras
from scipy import spatial
from sklearn.cluster import k_means


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

def sentence_to_vector_array(sentence, gap_index):
    sentence = sentence.strip("\n").split(" ")
    tokens_in_sentence = len(sentence)

    for i in range(tokens_in_sentence):
        try:
            sentence[i] = vector_per_token[sentence[i]]
        except KeyError:
            sentence[i] = vector_per_token["<UNK>"]

    return [sentence[:gap_index] + sentence[gap_index + 1:]]


def gap_fill_vector(zin, gap_index):
    zin_vector = sentence_to_vector_array(zin, gap_index)
    zin_vector_np = np.asarray(zin_vector)
    print("Shapes of zin_vector: ", end="")
    print(zin_vector_np.shape)

    return model.predict(zin_vector_np).tolist()

def consine_distance(vector1, vector2):
    return spatial.distance.cosine(vector1, vector2)

def most_similar_word(vector1, top):
    print('\nCalculating most similar words to missing vector...')
    word_to_cos_dist = dict()
    for token, vector in vector_per_token.items():
        one_dim_list = [item for sublist in vector for item in sublist]
        word_to_cos_dist[token] = consine_distance(vector1, one_dim_list)
    sorted_dict = sorted(word_to_cos_dist.items(), key=lambda x: x[1])
    return sorted_dict[:top]

def beste_matches(zin, gap_index, aantal_matches):
    print(" ".join(zin.split(" ")[:gap_index] + ["<GAP>"] + zin.split(" ")[gap_index + 1:]))
    verwachte_token_vector = (gap_fill_vector(zin, gap_index))
    verwachte_token_vector = verwachte_token_vector[0]
    # print("Shapes of output_vector: ", end="")
    # print(len(verwachte_token_vector))
    return (most_similar_word(verwachte_token_vector, aantal_matches))

def print_beste_match(zin, woord1, woord2, woord3):
    verwachte_token_vector = (gap_fill_vector(zin, gap_index))
    verwachte_token_vector = verwachte_token_vector[0]
    print(" ".join(zin.split(" ")[:gap_index] + ["<GAP>"] + zin.split(" ")[gap_index + 1:]))
    print("mogelijke woorden: {0} | {1} | {2}".format(woord1, woord2, woord3))
    print("de beste match is: {0}!".format(max(consine_distance(verwachte_token_vector, vector_per_token[woord1]),
                                               consine_distance(verwachte_token_vector, vector_per_token[woord2]),
                                               consine_distance(verwachte_token_vector, vector_per_token[woord3]))))

model = keras.models.load_model("model_met_vector_output.h5")

vector_file = "C:/Users/Sander/Documents/KUL/Toegepaste Informatica/2019 - 2020/Afstudeerproject/NLCOW data/glove_f50_12500.txt"
vector_per_token, amount_of_token_vectors, token_vector_dimension = make_token_vector_dictionary(vector_file)
gap_index = 5 #Het model is getraind voor index 5
beste_aantal_matches = 10
zin = "de heuvelachtige omgeving kenmerkt zich <GAP> kleine landerijen afgewisseld met bos ."
print(beste_matches(zin, gap_index, beste_aantal_matches))
zin = "zo kun je je horloge altijd aanpassen aan je eigen stijl ."
print(beste_matches(zin, gap_index, beste_aantal_matches))
zin = "we rijden verder door een maanlandschap en stijgen verder er verder ."
print(beste_matches(zin, gap_index, beste_aantal_matches))
zin = "vanavond gaan lopen en morgen heb ik hopelijk terug goed nieuws !"
print(beste_matches(zin, gap_index, beste_aantal_matches))
zin = "uiteindelijk begon er toch wat orde in de chaos te komen ."
print(beste_matches(zin, gap_index, beste_aantal_matches))

zin = "Niels eet vandaag een dikke <GAP> en hij vond hem lekker."
woord1 = "appel"
woord2 = "terwijl"
woord3 = "lopen"
print_beste_match(zin, woord1, woord2, woord3)
