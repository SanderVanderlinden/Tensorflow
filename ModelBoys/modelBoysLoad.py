import tensorflow as tf
from tensorflow import keras

def loss(input, target):
    return tf.keras.losses.sparse_categorical_crossentropy(input, target, from_logits=True)

pipikaka = keras.models.load_model("modelBoys.h5", compile = False)

pipikaka.compile(optimizer = "adam",
              loss = loss)

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


def generate_text(model, start_string):
    # Evaluation step (generating text using the learned model)
    # Number of characters to generate
    num_generate = 1
    # Converting our start string to numbers (vectorizing)
    string_die = start_string.split(" ")
    input_eval = [0] * len(string_die)
    for i in range(len(string_die)):
        try:
            input_eval[i] = word2indx[string_die[i]]
        except KeyError:
            input_eval[i] = 2
    print(input_eval)
    input_eval = tf.expand_dims(input_eval, 0)
    print(input_eval)
    # Empty string to store our results
    text_generated = []
    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = 1.0
    # Here batch size == 1
    model.reset_states()
    print("ttttttttttttttttttttt" + str(len(input_eval)))
    for i in range(num_generate):
        predictions = model(input_eval)
        #predictions = model.predict(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)
        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        # We pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2word[predicted_id])
    return start_string + ' ' + ' '.join(text_generated)


vector_file = "C:/Users/Sander/Documents/KUL/Toegepaste Informatica/2019 - 2020/Afstudeerproject/NLCOW data/glove50_1000.txt"
word2indx, idx2word = make_indices(vector_file)

print(generate_text(pipikaka, start_string='de tijd is er weer bijna om ggggggggggggggggggg'))