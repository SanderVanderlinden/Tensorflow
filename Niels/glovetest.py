import numpy as np
from sklearn.cluster import k_means
import datetime
import os
import sys
from flask import Flask, render_template, redirect, url_for,request
from flask import make_response
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

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
        return np.dot(self.word_dic[word1],self.word_dic[word2])/(np.linalg.norm(self.word_dic[word1])* np.linalg.norm(self.word_dic[word2]))

    def most_similar_word(self, word,top = 30):
        return sorted({word2:self.consine_distance(word, word2) for word2 in self.word_dic.keys()}.items(), key=lambda x:x[1], reverse=True)[1:top+1]

    def clustering(self, cluster_size):
        X = np.array(list(self.word_dic.items()))
        return k_means(X, n_clusters= cluster_size, init= "k-means++")

    def gap_fill(self, words):
        for word
        return best_result


@app.route('/glove', methods=['GET'])
def func():
    word = 'bal'
    model = Glove("vectors.txt")
    return model.most_similar_word(word)

if __name__ == "__main__":
    #starttime = datetime.datetime.now()
    #model = Glove("vectors.txt") #load model
    #print(model.most_similar_word(sys.argv[1]))
    app.run()
    

    
    #endtime = datetime.datetime.now()
    #print 'Time:', (endtime - starttime).seconds,'s'