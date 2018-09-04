# coding: utf-8

import csv
import os
import json
import spacy
import math
import random
from tqdm import tqdm
import numpy as np

from time import time

import re
import os
import codecs
import spacy
import sklearn
import matplotlib.pyplot as plt
from sklearn import model_selection
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import pickle

nlp_spacy_en = None
nlp_spacy_es = None


def get_spacy_model(lang="en"):
    """Set the spacy model in use.
    
        @param: lang A string to set English or Spanish model (default 'en')
        
        return the selected spacy model en/es"""
    global nlp_spacy_en
    global nlp_spacy_es
    if lang == "en":
        if nlp_spacy_en is None: 
            nlp_spacy_en = spacy.load(lang)
        return nlp_spacy_en
    elif lang == "es":
        if nlp_spacy_es is None: 
            nlp_spacy_es = spacy.load(lang)
        return nlp_spacy_es

class MeraDataset():
    """ Class to find typos based on the keyboard distribution, for QWERTY style keyboards
    
        It's the actual test set as defined in the paper that we comparing against."""

    def __init__(self, dataset_path, n_splits=3, ratio=0.3, augment=False):
        """ Instantiate the object.
            @param: dataset_path The directory which contains the data set.
            @param: n_splits For the stratified_split, not in use here (default 0.3).
            @param: ratio  For the stratified_split, not in use here (default 0.3).
            @param: augment If the data set should be augmented (default False)"""
        self.dataset_path = dataset_path
        self.augment = augment
        self.n_splits = n_splits
        self.ratio = ratio
        self.X_test, self.y_test, self.X_train, self.y_train = self.load()
        self.keyboard_cartesian = {'q': {'x': 0, 'y': 0}, 'w': {'x': 1, 'y': 0}, 'e': {'x': 2, 'y': 0},
                                   'r': {'x': 3, 'y': 0}, 't': {'x': 4, 'y': 0}, 'y': {'x': 5, 'y': 0},
                                   'u': {'x': 6, 'y': 0}, 'i': {'x': 7, 'y': 0}, 'o': {'x': 8, 'y': 0},
                                   'p': {'x': 9, 'y': 0}, 'a': {'x': 0, 'y': 1}, 'z': {'x': 0, 'y': 2},
                                   's': {'x': 1, 'y': 1}, 'x': {'x': 1, 'y': 2}, 'd': {'x': 2, 'y': 1},
                                   'c': {'x': 2, 'y': 2}, 'f': {'x': 3, 'y': 1}, 'b': {'x': 4, 'y': 2},
                                   'm': {'x': 5, 'y': 2}, 'j': {'x': 6, 'y': 1}, 'g': {'x': 4, 'y': 1},
                                   'h': {'x': 5, 'y': 1}, 'j': {'x': 6, 'y': 1}, 'k': {'x': 7, 'y': 1},
                                   'l': {'x': 8, 'y': 1}, 'v': {'x': 3, 'y': 2}, 'n': {'x': 5, 'y': 2}, }
        self.nearest_to_i = self.get_nearest_to_i(self.keyboard_cartesian)
        self.splits = self.stratified_split()


    def get_nearest_to_i(self, keyboard_cartesian):
        """ Get the nearest key to the one read.
            @params: keyboard_cartesian The layout of the QWERTY keyboard for English
            
            return dictionary of eaculidean distances for the characters"""
        nearest_to_i = {}
        for i in keyboard_cartesian.keys():
            nearest_to_i[i] = []
            for j in keyboard_cartesian.keys():
                if self._euclidean_distance(i, j) > 1.2:
                    nearest_to_i[i].append(j)
        return nearest_to_i

    def _shuffle_word(self, word, cutoff=0.9):
        """ Rearange the given characters in a word simulating typos given a probability.
        
            @param: word A single word coming from a sentence
            @param: cutoff The cutoff probability to make a change (default 0.9)
            
            return The word rearranged 
            """
        word = list(word.lower())
        if random.uniform(0, 1.0) > cutoff:
            loc = np.random.randint(0, len(word))
            if word[loc].isalpha():
                word[loc] = random.choice(self.nearest_to_i[word[loc]])
        return ''.join(word)

    def _euclidean_distance(self, a, b):
        """ Calculates the euclidean between 2 points in the keyboard
            @param: a Point one 
            @param: b Point two
            
            return The euclidean distance between the two points"""
        X = (self.keyboard_cartesian[a]['x'] - self.keyboard_cartesian[b]['x']) ** 2
        Y = (self.keyboard_cartesian[a]['y'] - self.keyboard_cartesian[b]['y']) ** 2
        return math.sqrt(X + Y)

    def _augment_sentence(self, sentence, num_samples):
        """ Augment the dataset of with a sentence shuffled
            @param: sentence The sentence from the set
            @param: num_samples The number of sentences to genererate
            
            return A set of augmented sentences"""
        sentences = []
        for _ in range(num_samples):
            sentences.append(' '.join([self._shuffle_word(item) for item in sentence.split(' ')]))
        sentences = list(set(sentences))
        return sentences + [sentence]

    def _augment_split(self, X_train, y_train, num_samples=1000):
        """ Split the aumented train dataset
            @param: X_train The full array of sentences
            @param: y_train The train labels in the train dataset
            @param: num_samples the number of new sentences to create (default 1000)
            
            return Augmented training dataset"""
        Xs, ys = [], []
        for X, y in zip(X_train, y_train):
            tmp_x = self._augment_sentence(X, num_samples)
            _ = [[Xs.append(item), ys.append(y)] for item in tmp_x]
        return Xs, ys

    def load(self):
        """ Load the file for now only the test.csv, train.csv files hardcoded
        
            return The vector separated in test, train and the labels for each one"""
        with open('test.csv') as csvfile:
            readCSV = csv.reader(csvfile, delimiter='\t')
            all_rows = list(readCSV)
            for i in all_rows:
                if i ==  28823:
                    print(all_rows[i])
            X_test = [a[0] for a in all_rows]
            y_test = [a[1] for a in all_rows]

        with open('train.csv', encoding="utf8") as csvfile:
            readCSV = csv.reader(csvfile, delimiter='\t')
            all_rows = list(readCSV)
            X_train = [a[0] for a in all_rows]
            y_train = [a[1] for a in all_rows]
        return X_test, y_test, X_train, y_train

    def process_sentence(self, x):
        """ Clean the tokens from stop words in a sentence.
            @param x Sentence to get rid of stop words.
            
            returns clean string sentence"""
        clean_tokens = []
        doc = get_spacy_model("en")(x)
        for token in doc:
            if not token.is_stop:
                clean_tokens.append(token.lemma_)
        return " ".join(clean_tokens)

    def process_batch(self, X):
        """See the progress as is coming along.
        
            return list[] of clean sentences"""
        return [self.process_sentence(a) for a in tqdm(X)]

    def stratified_split(self):
        """ Split data whole into stratified test and training sets, then remove stop word from sentences
        
            return list of dictionaries with keys train,test and values the x and y for each one"""

        self.X_train, self.y_train = self._augment_split(self.X_train,
                                                         self.y_train)
        self.X_train = self.process_batch(self.X_train)
        self.X_test = self.process_batch(self.X_test)

        splits = [{"train": {"X": self.X_train, "y": self.y_train},
                   "test": {"X": self.X_test, "y": self.y_test}}]
        return splits

    def get_splits(self):
        """ Get the splitted sentences
            
            return splitted list of dictionaries"""
        return self.splits


class Dataset():
    """ Class to stratify and split the data passed into it, simple separarator of text"""

    def __init__(self, dataset_path, n_splits=3, ratio=0.3, augment=False):
        """ Instantiate the object.
            @param: dataset_path The directory which contains the data set.
            @param: n_splits For the stratified_split, not in use here (default 0.3).
            @param: ratio  For the stratified_split, not in use here (default 0.3).
            @param: augment If the data set should be augmented (default False)
            
            return A created object with a stratified split"""
        self.dataset_path = dataset_path
        self.augment = augment
        self.n_splits = n_splits
        self.ratio = ratio
        self.X, self.y = self.load()
        self.splits = self.stratified_split(self.X, self.y, self.n_splits, self.ratio)
    
    def load(self):
        """ Load the file for now only the test.csv, train.csv files hardcoded
        
            return The list of dictionaries separated in test, train in X and the labels for each one in y"""
        with open(self.dataset_path, "r", encoding='utf8') as f:
            dataset = json.load(f)
            X = [sample["text"] for sample in dataset["sentences"]]
            y = [sample["intent"] for sample in dataset["sentences"]]
        return X, y
    
    def stratified_split(self, X, y, n_splits=10, test_size=0.2):
        """ Sklearn function to do stratified splitting of the input
            @param: X List of text vlaues for train/test
            @param: y List of expected labels for train/test
            @param: n_splits Number of splits to return the data in (default 10)
            @param: test_size Size of data to hold for testing purposes (default 0.2)
            
            return List of dictionaries separated by keys train, test
            """
        skf = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)
        skf.get_n_splits(X, y)
        splits = []
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
            y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]
            # add augmentation code here
            splits.append({"train": {"X": X_train, "y": y_train},
                           "test": {"X": X_test, "y": y_test}})
        return splits
    
    def get_splits(self):
        """ Get the data in splitted format
        
            return List of dictionaries separated by keys train, test
            """
        return self.splits

dataset = Dataset("./data/datasets/AskUbuntuCorpus.json")
splits = dataset.get_splits()
for split in splits:
    print("X train", split["train"]["X"][: 2])
    print("y train", split["train"]["y"][:2])
    print("X test", split["test"]["X"][: 2])
    print("y test", split["test"]["y"][:2])

def find_ngrams(input_list, n):
    """Create ngrams for input list
        @param: input_list The list to separate in ngrams
        @param: n Number of ngrams to find"""
    return zip(*[input_list[i:] for i in range(n)])

def semhash_tokenizer(text):
    """ To create the tokens from the ngrams with #token# style
        @param: text String to transform
        
        return final_tokens list of ngrams for words with hashes at the beginning and end"""
    tokens = text.split(" ")
    final_tokens = []
    for unhashed_token in tokens:
        hashed_token = "#{}#".format(unhashed_token)
        final_tokens += [''.join(gram)
                         for gram in list(find_ngrams(list(hashed_token), 3))]
    return final_tokens

class SemhashFeaturizer:
    """Class to construct a featurizer for SemHash model """
    def __init__(self):
        """Instantiate object """
        self.vectorizer = self.get_vectorizer()
    
    def get_vectorizer(self):
        """ Use sklearn method TfidfVectorizer to create a vectorized input removing stop words with the semhash_tokenizer
        
            return An object that converts a collection of raw documents to a matrix of TF-IDF features."""
        return TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False,
                               sublinear_tf=True,
                               tokenizer=semhash_tokenizer, )
    
    def fit(self, X, *args, **kwargs):
        """ Learn vocabulary and idf, return term-document matrix. This is equivalent to fit followed by transform, but more efficiently implemented.
            @param: X an iterable which yields either str, unicode or file objects

            return Tf-idf-weighted document-term matrix
            """
        self.vectorizer.fit(X)
        
    
    def transform(self, X):
        """ Transform documents to document-term matrix, uses the vocabulary and document frequencies (df) learned by fit
            @param: an iterable which yields either str, unicode or file objects
            
            return Tf-idf-weighted document-term matrix as an array"""
        return self.vectorizer.transform(X).toarray()

X, y = ["hello", "I am a boy"], ["A", "B"]

semhash_featurizer = SemhashFeaturizer()
semhash_featurizer.fit(X, y)
X_ = semhash_featurizer.transform(X)

class W2VFeaturizer:
    """ Class to encapsule the object which will get the Word2Vec featurizer """
    def __init__(self, lang):
        """ Instantiate the object with any language downloaded for spacy"""
        self.lang = lang
    
    def fit(self, X, *args, **kwargs):
        """ Not implemented, does nothing"""
        pass
    
    def transform(self, x):
        """ Transforms the a list of strings to a np.array as vectors from a spacy model, needs to have downloaded the spacy model used, in this case English 'en'
            @param: x list of strings 
            
            return np.array from a spacy model as vectors"""
        np_val = np.array([get_spacy_model(self.lang)(s).vector for s in x])
        return np_val

X, y = ["hello", "I am a boy"], ["A", "B"]
glove_path = ""
w2v_featurizer = W2VFeaturizer("en")
w2v_featurizer.fit(X, y)
X_ = w2v_featurizer.transform(X)

def save_file(file_name, results):
    """Save the results of the train models to a file as binary
        @param: file_name the file name to save as
        @param: results Results matrix to put into file"""
    with open(file_name, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_file(name):
    """Load the binary file of the results
        @param: name The file that was saved as binary 
        
        prints the data in the file (Note this one is hardcoded for testing purposes 
        """
    with open('./outlook.txt', 'rb') as handle:
        file_data = pickle.load(handle)
        print(file_data)


class Trainer:
    """Class of ml models to do the tests """
    def __init__(self, splits, featurizer, path="./data/plots", lang="en", name="default"):
        """ Instantiate object
            @param: splits Data formatted and splitted from the raw input
            @param: featurizer Where is specified the type of featurizer needed semhash or w2v
            @param: path The path to save the resulting plots (default ./data/plots)
            @param: lang Spacy model used (default en)
            @param: name Name of the run"""
        self.path = os.path.join(path, name)
        if not os.path.exists(self.path): os.makedirs(self.path)
        self.splits = splits
        self.featurizer = featurizer
        self.lang = lang
        self.results = None
    
    def get_X_andy_from_split(self, split):
        """ To get the X from splitted list of dictionaries
            @param: split A List of dictionaries with keys {'train':{'X':[],'y':[]}, 'test':{'X':[],'y':[] }}
            
            return the splitted X, y for the train, test"""
        train_corpus, y_train = split["train"]["X"], split["train"]["y"]
        test_corpus, y_test = split["test"]["X"], split["test"]["y"]
        self.featurizer.fit(train_corpus)
        self.featurizer.fit(test_corpus)
        X_train = self.featurizer.transform(train_corpus)
        X_test = self.featurizer.transform(test_corpus)
        return X_train, y_train, X_test, y_test
    
    def train(self):
        """Train several models for comparison MLP, RF, KNN, SVC, SGD and plot their results """
        parameters_mlp={'hidden_layer_sizes':[(100,50),(300,100,50),(200,
                                                                     100)],
                        "solver": ["sgd"], "activation": ["relu", "tanh"],
                        "learning_rate": ["adaptive"]}
        parameters_RF={ "n_estimators" : [50,60,70], "min_samples_leaf" : [1, 2]}
        k_range = list(range(1, 11))
        parameters_knn = {'n_neighbors':k_range}
        
        for i_s, split in enumerate(self.splits):
            print("Evaluating Split {}".format(i_s))
            X_train, y_train, X_test, y_test = self.get_X_andy_from_split(split)
            print("Train Size: {}\nTest Size: {}".format(X_train.shape[0], X_test.shape[0]))
            results = []
            knn=KNeighborsClassifier(n_neighbors=5)

            for clf, name in [  
                    #(RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge "
            #                                                    "Classifier"),
                    #(GridSearchCV(knn,parameters_knn, cv=10),"gridsearchknn"),
                    (GridSearchCV(MLPClassifier(verbose=True, max_iter=500),
                                  parameters_mlp, cv=3),"gridsearchmlp"),
                    #(PassiveAggressiveClassifier(n_iter=50),
            # "Passive-Aggressive"),
                    #(GridSearchCV(RandomForestClassifier(n_estimators=10),
            # parameters_RF, cv=10),"gridsearchRF")
            ]:
                #print('=' * 80)
                #print(name)
                results.append(self.benchmark(clf, X_train, y_train, X_test,
                                          y_test))

            for penalty in ["l2", "l1"]:
                print('=' * 80)
                print("%s penalty" % penalty.upper())
                # Train Liblinear model
                #grid=(GridSearchCV(LinearSVC,parameters_Linearsvc, cv=10),"gridsearchSVC")
                #results.append(benchmark(LinearSVC(penalty=penalty), X_train, y_train, X_test, y_test, target_names,
                                        # feature_names=feature_names))
                """
                results.append(self.benchmark(LinearSVC(penalty=penalty,
                                                    dual=False,tol=1e-3),
                                         X_train, y_train, X_test, y_test))
                """
                # Train SGD model
                results.append(self.benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                                       penalty=penalty),
                                         X_train, y_train, X_test, y_test))

            # Train SGD with Elastic Net penalty
            print('=' * 80)
            print("Elastic-Net penalty")
            results.append(self.benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                                   penalty="elasticnet"),
                                     X_train, y_train, X_test, y_test))
            self.results = results
            self.plot_results(results)

#             """
#             # Train NearestCentroid without threshold
#             print('=' * 80)
#             print("NearestCentroid (aka Rocchio classifier)")
#             results.append(self.benchmark(NearestCentroid(),
#                                      X_train, y_train, X_test, y_test))
#             try:
#                 # Train sparse Naive Bayes classifiers
#                 print('=' * 80)
#                 print("Naive Bayes")
#                 results.append(self.benchmark(MultinomialNB(alpha=.01),
#                                          X_train, y_train, X_test, y_test))
#                 results.append(self.benchmark(BernoulliNB(alpha=.01),
#                                          X_train, y_train, X_test, y_test))
#             except:
#                 continue
# 
#             print('=' * 80)
#             print("LinearSVC with L1-based feature selection")
#             # The smaller C, the stronger the regularization.
#             # The more regularization, the more sparsity.
# 
#             
#             results.append(self.benchmark(Pipeline([
#                                           ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False,
#                                                                                           tol=1e-3))),
#                                           ('classification', LinearSVC(penalty="l2"))]),
#                                      X_train, y_train, X_test, y_test))
#            # print(grid.grid_scores_)
#            #KMeans clustering algorithm 
#             print('=' * 80)
#             print("KMeans")
#             #results.append(self.benchmark(KMeans(n_clusters=2,
#             # init='k-means++', max_iter=300,
#           #              verbose=0, random_state=0, tol=1e-4),
#           #                           X_train, y_train, X_test, y_test))
# 
# 
# 
#             print('=' * 80)
#             print("LogisticRegression")
#             #kfold = model_selection.KFold(n_splits=2, random_state=0)
#             #model = LinearDiscriminantAnalysis()
#             results.append(self.benchmark(LogisticRegression(C=1.0,
#                                                          class_weight=None, dual=False,
#                   fit_intercept=True, intercept_scaling=1, max_iter=100,
#                   multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
#                   solver='liblinear', tol=0.0001, verbose=0, warm_start=False),
#                                      X_train, y_train, X_test, y_test))
#             """
    
    
    def benchmark(self, clf, X_train, y_train, X_test, y_test,
              print_report=True, print_top10=False,
              print_cm=True):
        """Display the training process, time taken, accuracy, classification report, confusion matrix and score
            @param: clf Model used for classification
            @param: X_train list of training sentences
            @param: y_train list of labels for the training set
            @param: X_test list of testing sentences
            @param: print_report Whether the report should be printed (default True)
            @param: print_top10 Whether the top 10 classifiers shouls be printed (default False)
            @param: print_cm Wheter the confusion matrix should be printed for each classifier (defatlt True)
            
            return clf_descr, score, train_time, test_time description of classifier and metrics of the runs"""
        print('_' * 80)
        print("Training: ")
        print(clf)
        t0 = time()
        clf.fit(X_train, y_train)
        train_time = time() - t0
        print("train time: %0.3fs" % train_time)

        t0 = time()
        pred = clf.predict(X_test)
        test_time = time() - t0
        print("test time:  %0.3fs" % test_time)

        score = metrics.accuracy_score(y_test, pred)
        print("accuracy:   %0.3f" % score)

        if hasattr(clf, 'coef_'):
            print("dimensionality: %d" % clf.coef_.shape[1])
            print("density: %f" % density(clf.coef_))
            print()

        if print_report:
            print("classification report:")
            print(metrics.classification_report(y_test, pred))

        if print_cm:
            print("confusion matrix:")
            print(metrics.confusion_matrix(y_test, pred))

        print()
        clf_descr = str(clf).split('(')[0]
        return clf_descr, score, train_time, test_time
    
    def plot_results(self, results):
        """ Plot the results of the tests in a horizontal fashion
            @param: results list of lists of results
            """
        indices = np.arange(len(results))

        results = [[x[i] for x in results] for i in range(4)]

        save_file("./outlook.txt", results)
        clf_names, score, training_time, test_time = results
        training_time = np.array(training_time) / np.max(training_time)
        test_time = np.array(test_time) / np.max(test_time)

        plt.figure(figsize=(12, 8))
        plt.title("Score")
        plt.barh(indices, score, .2, label="classifiers", color='navy')
        plt.barh(indices + .3, training_time, .2, label="training time",
                 color='c')
        plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
        plt.yticks(())
        plt.legend(loc='best')
        plt.subplots_adjust(left=.25)
        plt.subplots_adjust(top=.95)
        plt.subplots_adjust(bottom=.05)

        for i, c in zip(indices, clf_names):
            plt.text(-.3, i, c)
        plt.savefig("./plottedResultsHor.png", format="png")
        plt.show()

    def save_to_file():
        """ Integrate the save results to the class, not used as of now"""
        with open("./oulook.txt", "w") as text_file:
            print("test", file=text_file)
        
    def plot_ver_bars(self,results):
        """ Plot the results of the tests in a vertical fashion
            @param: results list of lists of results"""
        indices = np.arange(len(results))

        results = [[x[i] for x in results] for i in range(len(results))]

        save_file("outlook.txt", results)
        clf_names, score, training_time, test_time = results
        print(results, "func", indices, "indices", clf_names, "clf_names" )
        training_time = np.array(training_time) / np.max(training_time)
        test_time = np.array(test_time) / np.max(test_time)

        plt.figure(figsize=(12, 8))
        plt.title("Score")
        plt.bar(indices, score, .2, label="classifiers", color='navy')
        plt.bar(indices + .3, training_time, .2, label="training time",
                 color='c')
        plt.bar(indices + .6, test_time, .2, label="test time", color='darkorange')
        plt.xlabel("Classifiers used")
        plt.ylabel("Score")
        plt.xticks(())
        plt.legend(loc='best')
        plt.subplots_adjust(left=.05)
        plt.subplots_adjust(top=.95)
        plt.subplots_adjust(bottom=.15)

        for i, c in zip(indices, clf_names):
            plt.text(i, -.05, c)
            # print(i,c) #to set the text of the plots i needed to check them
            # plt.savefig("./plots/"+c+".png", format="png")
        plt.savefig("./plottedResultsVert.png", format="png")
        plt.show()
        

save_file("test.txt",{"hello":42})
semhash_featurizer = SemhashFeaturizer()
dataset = MeraDataset("./data/datasets/AskUbuntuCorpus"
                  ".json", ratio=0.2)
splits = dataset.get_splits()

trainer = Trainer(splits, semhash_featurizer, lang="en", path="./data/plots")#
dataset = Dataset("./data/datasets/AskUbuntuCorpus.json", n_splits=2, ratio=0.66, augment=False)
splits = dataset.get_splits()
trainer = Trainer(splits, semhash_featurizer, lang="en", path="./data/plots",
                  name="Ubuntu")

trainer.train()

trainer.plot_ver_bars(trainer.results)
print(trainer.results)
print(len(trainer.results))
load_file("./otulook.txt")
print("********")
print(len(trainer.results), "len", trainer.results, "full", trainer.results[3])
