import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model("model.h5")

vector_file = "C:/Users/Sander/Documents/KUL/Toegepaste Informatica/2019 - 2020/Afstudeerproject/NLCOW data/glove50_1000.txt"
print('\nIndexing token vectors.')
with open(vector_file, 'r', errors='ignore') as input_file:
    vector_per_token = dict()
    token_indices = dict()
    i = 0
    for line in input_file:
        line_array = line.split()
        try:
            vector_per_token[line_array[0]] = [[float(x)] for x in line_array[1:]]
        except ValueError:
            print("VALUE-ERROR!!!!!")
        token_indices[line_array[0]] = i
        i += 1

zin =("we worden dan nog van het kastje naar de muur gestuurd .")

gap_index = 5
y_value = []
woorden = zin.split(" ")
print(woorden)

for woord in woorden:
    for i in range(len(woorden)):
        if i == gap_index:
            try:
                print(woorden[gap_index])
                print(woorden)
                index = token_indices[woorden[gap_index]]
            except KeyError:
                index = -1
            output_vector = [0] * len(woorden)
            if index != -1:
                output_vector[index] = 1
            y_value.append(output_vector)

        try:
            woorden[i] = vector_per_token[zin[i]]
        except KeyError:
            woorden[i] = [[0.0]] * len(woorden)

print(woorden)
print(y_value)