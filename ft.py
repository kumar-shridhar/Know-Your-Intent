import gensim
from gensim.models.word2vec import LineSentence
from gensim.models.fasttext import FastText
import csv

class FastText_Model():

    def __init__(self):
        self.model = None
        self.vector_size = 0

    def create_model(self, text_file, vector_size, window, min_count=1, iter=10):
        data = LineSentence(text_file)
        self.model = FastText(data, size=vector_size, window = window, min_count = min_count, iter = iter)
        self.vector_size = 0
        return self.model

    def get_vector(self, word, defaultToZero):
        if self.model = None:
            raise RunTimeError("No model have been loaded! Use create_model to create a FastText model")
        if word not in self.model.wv.vocab:
            if defaultToZero:
                return np.zeros(self.vector_size)
            return None
        return self.model.wv(word)
