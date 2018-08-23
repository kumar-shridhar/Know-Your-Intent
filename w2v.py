import gensim
import numpy as np

class Model():

    #Creates the Model object. If model is not set to None it calls load_model with that parameter
    def __init__(self, model = None):
        self.model = None
        if model != None:
            load_model(model)

    #Takes a path to a word2vec binary and loads that as the current model
    def load_model(self, model):
        if model == "Google":
            model = "GoogleNews-vectors-negative300.bin",
        model = gensim.models.KeyedVectors.load_word2vec_format(model, binary = True)    
        self.model = model
        return self

    #Returns a numpy array of the word embedding. Will return a zero vector if defaultToZero is True and the word cannot be found
    def get_vector(self, word, defaultToZero = True):
        if self.model = None:
            raise RunTimeError("No model have been loaded! Use load_model(<model_name>) ir __init__(<model_name>) to load a word2vec model")
        if word not in model and defaultToZero:
            return np.zeros(self.model.vector_size)
        return self.model[word]

