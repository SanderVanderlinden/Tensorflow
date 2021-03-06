import numpy as np
import tensorflow as tf

import os
import json
from string import punctuation
from collections import defaultdict 


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#tf.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()
batch_size = None  # Any size is accepted
word_representations_dimensions = 50  # Embedding of size (vocab_len, nb_dimensions)


DATA_FOLDER = "Data"
SUBFOLDER_NAME = "glove1"
TF_EMBEDDING_FILE_NAME = "{}.ckpt".format(SUBFOLDER_NAME)
SUFFIX = SUBFOLDER_NAME + "." + str(word_representations_dimensions)
TF_EMBEDDINGS_FILE_PATH = os.path.join(DATA_FOLDER, SUFFIX + "d.ckpt")
DICT_WORD_TO_INDEX_FILE_NAME = os.path.join(DATA_FOLDER, SUFFIX + "d.json")


def load_word_to_index(dict_word_to_index_file_name):
    """
    Load a `word_to_index` dict mapping words to their id, with a default value
    of pointing to the last index when not found, which is the unknown word.
    """
    with open(dict_word_to_index_file_name, 'r') as f:
        word_to_index = json.load(f)
    _LAST_INDEX = len(word_to_index) - 2  # Why - 2? Open issue?
    print("word_to_index dict restored from '{}'.".format(dict_word_to_index_file_name))
    word_to_index = defaultdict(lambda: _LAST_INDEX, word_to_index)

    return word_to_index

def load_embedding_tf(word_to_index, tf_embeddings_file_path, nb_dims):
    """
    Define the embedding tf.Variable and load it.
    """
    # 1. Define the variable that will hold the embedding:
    tf_embedding = tf.Variable(
        tf.constant(0.0, shape=[len(word_to_index)-1, nb_dims]),
        trainable=False,
        name="Embedding"
    )

    # 2. Restore the embedding from disks to TensorFlow, GPU (or CPU if GPU unavailable):
    variables_to_restore = [tf_embedding]
    embedding_saver = tf.compat.v1.train.Saver(variables_to_restore)
    embedding_saver.restore(sess, save_path=tf_embeddings_file_path)
    print("TF embeddings restored from '{}'.".format(tf_embeddings_file_path))
    
    return tf_embedding
    
def cosine_similarity_tensorflow(tf_word_representation_A, tf_words_representation_B):
    """
    Returns the `cosine_similarity = cos(angle_between_a_and_b_in_space)` 
    for the two word A to all the words B.
    The first input word must be a 1D Tensors (word_representation).
    The second input words must be 2D Tensors (batch_size, word_representation).
    The result is a tf tensor that must be fetched with `sess.run`.
    """
    a_normalized = tf.nn.l2_normalize(tf_word_representation_A, axis=-1)
    b_normalized = tf.nn.l2_normalize(tf_words_representation_B, axis=-1)
    similarity = tf.reduce_sum(
        tf.multiply(a_normalized, b_normalized), 
        axis=-1
    )
    
    return similarity


# In case you didn't do the "%reset": 
tf.compat.v1.reset_default_graph()
sess = tf.compat.v1.InteractiveSession()  # sess = tf.Session()

# Load the embedding matrix in tf
word_to_index = load_word_to_index(
    DICT_WORD_TO_INDEX_FILE_NAME)
tf_embedding = load_embedding_tf(
    word_to_index,
    TF_EMBEDDINGS_FILE_PATH, 
    word_representations_dimensions)

# Input to the graph where word IDs can be sent in batch. Look at the "shape" args:
tf_word_A_id = tf.compat.v1.placeholder(tf.int32, shape=[1])
tf_words_B_ids = tf.compat.v1.placeholder(tf.int32, shape=[batch_size])

# Conversion of words to a representation
tf_word_representation_A = tf.nn.embedding_lookup(
    params=tf_embedding, ids=tf_word_A_id)
tf_words_representation_B = tf.nn.embedding_lookup(
    params=tf_embedding, ids=tf_words_B_ids)

# The graph output are the "cosine_similarities" which we want to fetch in sess.run(...). 
cosine_similarities = cosine_similarity_tensorflow(
    tf_word_representation_A, 
    tf_words_representation_B)

print("Model created.")

def sentence_to_word_ids(sentence, word_to_index):
    """
    Note: there might be a better way to split sentences for GloVe.
    Please look at the documentation or open an issue to suggest a fix.
    """
    # Separating punctuation from words:
    for punctuation_character in punctuation:
        sentence = sentence.replace(punctuation_character, " {} ".format(punctuation_character))
    # Removing double spaces and lowercasing:
    sentence = sentence.replace("  ", " ").replace("  ", " ").lower().strip()
    # Splitting on every space:
    split_sentence = sentence.split(" ")
    # Converting to IDs:
    ids = [word_to_index[w.strip()] for w in split_sentence]
    print(ids)
    return ids, split_sentence

def predict_cosine_similarities(sess, word_A, words_B):
    """
    Use the model in sess to predict cosine similarities.
    """

    word_A_id, _ = sentence_to_word_ids(word_A, word_to_index)
    words_B_ids, split_sentence = sentence_to_word_ids(words_B, word_to_index)

    evaluated_cos_similarities = sess.run(
        cosine_similarities, 
        feed_dict={
            tf_word_A_id: word_A_id,
            tf_words_B_ids: words_B_ids
        }
    )
    return evaluated_cos_similarities, split_sentence


word_A = "Waarom"
words_B = "hallo beste domme kijkers, veel worden zijn juist behalve dEZezeefgzhaef!"

evaluated_cos_similarities, splitted = predict_cosine_similarities(sess, word_A, words_B)

print("Cosine similarities with \"{}\":".format(word_A))
for word, similarity in zip(splitted, evaluated_cos_similarities):
    print("    {}{}".format((word+":").ljust(15), similarity))

tf.compat.v1.reset_default_graph()


# Transpose word_to_index dict:
index_to_word = dict((val, key) for key, val in word_to_index.items())


# New graph
tf.compat.v1.reset_default_graph()
sess = tf.compat.v1.InteractiveSession()

# Load the embedding matrix in tf
tf_word_to_index = load_word_to_index(
    DICT_WORD_TO_INDEX_FILE_NAME)
tf_embedding = load_embedding_tf(
    tf_word_to_index,
    TF_EMBEDDINGS_FILE_PATH, 
    word_representations_dimensions)

# An input word 
tf_word_id = tf.compat.v1.placeholder(tf.int32, shape=[1])
tf_word_representation = tf.nn.embedding_lookup(
    params=tf_embedding, ids=tf_word_id)

# An input 
tf_nb_similar_words_to_get = tf.compat.v1.placeholder(tf.int32)

# Dot the word to every embedding
tf_all_cosine_similarities = cosine_similarity_tensorflow(
    tf_word_representation, 
    tf_embedding)

# Getting the top cosine similarities. 
tf_top_cosine_similarities, tf_top_word_indices = tf.nn.top_k(
    tf_all_cosine_similarities,
    k=tf_nb_similar_words_to_get+1,
    sorted=True
)

# Discard the first word because it's the input word itself:
tf_top_cosine_similarities = tf_top_cosine_similarities[1:]
tf_top_word_indices = tf_top_word_indices[1:]

# Get the top words' representations by fetching 
# tf_top_words_representation = "tf_embedding[tf_top_word_indices]":
tf_top_words_representation = tf.gather(
    tf_embedding,
    tf_top_word_indices)


nb_similar_words_to_get = 10


word = "bal"
word_id = word_to_index[word]

top_cosine_similarities, top_word_indices, top_words_representation = sess.run(
    [tf_top_cosine_similarities, tf_top_word_indices, tf_top_words_representation],
    feed_dict={
        tf_word_id: [word_id],
        tf_nb_similar_words_to_get: nb_similar_words_to_get
    }
)

print("Top similar words to \"{}\":\n".format(word))
loop = zip(top_cosine_similarities, top_word_indices, top_words_representation)
for cos_sim, word_id, word_repr in loop:
    print(
        (index_to_word[word_id]+ ":").ljust(15),
        (str(cos_sim) + ",").ljust(15),
        np.linalg.norm(word_repr)
    )