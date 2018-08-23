import gensim
from gensim.models.word2vec import LineSentence
from gensim.models.fasttext import FastText
import csv

class FastText_Model():

    def __init__(self):
        self.model = None

    def newModel(self, text_file, vector_size, window, min_count=1, iter=10):
        data = LineSentence(text_file)
        self.model = FastText(data, size=vector_size, window = window, min_count = min_count, iter = iter)
        return self.model

