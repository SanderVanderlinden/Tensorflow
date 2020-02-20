import numpy as np
import tensorflow as tf
 
import chakin
import keras
import json
import os
from collections import defaultdict

tf.compat.v1.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
SUBFOLDER_NAME = "glove1"

#verander de volgende naa het juiste
NUMBER_OF_DIMENSIONS = 50
DATA_FOLDER = "PAD WAAR GLOVE ZIT"
GLOVE_FILENAME = 'Data/vectors.txt' #verctor.txt van glove

#https://github.com/guillaume-chevalier/GloVe-as-a-TensorFlow-Embedding-Layer

def load_embedding_from_disks(glove_filename, with_indexes=True):
    """
    Read a GloVe txt file. If `with_indexes=True`, we return a tuple of two dictionnaries
    `(word_to_index_dict, index_to_embedding_array)`, otherwise we return only a direct 
    `word_to_embedding_dict` dictionnary mapping from a string to a numpy array.
    """
    if with_indexes:
        word_to_index_dict = dict()
        index_to_embedding_array = []
    else:
        word_to_embedding_dict = dict()

    
    with open(glove_filename, 'r',  encoding="utf8") as glove_file:
        for (i, line) in enumerate(glove_file):
            
            split = line.split(' ')
            
            word = split[0]
            
            representation = split[1:]
            representation = np.array(
                [float(val) for val in representation]
            )
            
            if with_indexes:
                word_to_index_dict[word] = i
                index_to_embedding_array.append(representation)
            else:
                word_to_embedding_dict[word] = representation

    _WORD_NOT_FOUND = [0.0]* len(representation)  # Empty representation for unknown words.
    if with_indexes:
        _LAST_INDEX = i + 1
        word_to_index_dict = defaultdict(lambda: _LAST_INDEX, word_to_index_dict)
        index_to_embedding_array = np.array(index_to_embedding_array + [_WORD_NOT_FOUND])
        return word_to_index_dict, index_to_embedding_array
    else:
        word_to_embedding_dict = defaultdict(lambda: _WORD_NOT_FOUND)
        return word_to_embedding_dict


print("Loading embedding from disks...")
word_to_index, index_to_embedding = load_embedding_from_disks(GLOVE_FILENAME, with_indexes=True)
print("Embedding loaded from disks.")


#nu gaan we glove omzetten naar TF 

batch_size = None  # Any size is accepted

tf.compat.v1.reset_default_graph()
sess = tf.compat.v1.InteractiveSession()  # sess = tf.Session()

# Define the variable that will hold the embedding:
tf_embedding = tf.Variable(
    tf.constant(0.0, shape=index_to_embedding.shape),
    trainable=False,
    name="Embedding"
)

tf_word_ids = tf.compat.v1.placeholder(tf.int32, shape=[batch_size])

tf_word_representation_layer = tf.nn.embedding_lookup(
    params=tf_embedding,
    ids=tf_word_ids
)

tf_embedding_placeholder = tf.compat.v1.placeholder(tf.float32, shape=index_to_embedding.shape)
tf_embedding_init = tf_embedding.assign(tf_embedding_placeholder)
_ = sess.run(
    tf_embedding_init, 
    feed_dict={
        tf_embedding_placeholder: index_to_embedding
    }
)

print("Embedding now stored in TensorFlow. Can delete numpy array to clear some CPU RAM.")
del index_to_embedding

prefix = SUBFOLDER_NAME + "." + str(NUMBER_OF_DIMENSIONS) + "d"
TF_EMBEDDINGS_FILE_NAME = os.path.join(DATA_FOLDER, prefix + ".ckpt")
DICT_WORD_TO_INDEX_FILE_NAME = os.path.join(DATA_FOLDER, prefix + ".json")

variables_to_save = [tf_embedding]
embedding_saver = tf.compat.v1.train.Saver(variables_to_save)
embedding_saver.save(sess, save_path=TF_EMBEDDINGS_FILE_NAME)
print("TF embeddings saved to '{}'.".format(TF_EMBEDDINGS_FILE_NAME))
sess.close()

with open(DICT_WORD_TO_INDEX_FILE_NAME, 'w') as f:
    json.dump(word_to_index, f)
print("word_to_index dict saved to '{}'.".format(DICT_WORD_TO_INDEX_FILE_NAME))

