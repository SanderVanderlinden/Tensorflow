import tensorflow as tf
from tensorflow import keras

#def loss(input, target):
#    return tf.keras.losses.sparse_categorical_crossentropy(input, target, from_logits=True)

#model = keras.models.load_model("modelBoys.h5", compile = False)
model = keras.models.load_model("modelBoysMetGlove.h5", compile = False)

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

def generate_next_word(model, string):
    string_splitted = string.split(" ")
    string_indices = []
    for index in range(len(string_splitted)- 2):
        a = []
        a.append(word2indx[string_splitted[index]])
        a.append(word2indx[string_splitted[index + 1]])
        a.append(word2indx[string_splitted[index + 2]])
        string_indices.append(a)
    print(string_splitted)
    print(string_indices)

    yhat = model.predict_classes(string_indices, verbose = 0)
    out_indices = []
    for i in yhat:
        out_indices.append(i)
    print(out_indices)
    out_text = []
    for i in out_indices:
        out_text.append(idx2word[i])
    return " ".join(out_text)

def generate_text(model, string, n_words):
    while len(string.split(" ")) < 50:
        string_splitted = string.split(" ")
        string_indices = []
        for index in range(len(string_splitted)):
            string_indices.append(word2indx[string_splitted[index]])
        yhat = model.predict_classes(string_indices, verbose=0)
        out_indices = []
        for i in yhat:
            out_indices.append(i)
        out_text = []
        for i in out_indices:
            out_text.append(idx2word[i])
        string = (" ".join(string_splitted) + " " + out_text[-1])
    return string


vector_file = "C:/Users/Sander/Documents/KUL/Toegepaste Informatica/2019 - 2020/Afstudeerproject/NLCOW data/glove_f50_12500.txt"
word2indx, idx2word = make_indices(vector_file)
n_words = 50

print(generate_next_word(model, 'de tijd is er weer bijna om'))
print(generate_next_word(model, 'ik wil gaan lopen met mijn vrienden'))
tekst = generate_text(model, 'ik wil gaan lopen met mijn vrienden', 50)
print(tekst)
print("aantal woorden in bovenstaande tekst is {0}".format(len(tekst.split(" "))))