from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io
import logging
import os

from future.utils import PY3
from sklearn.feature_extraction.text import CountVectorizer

from chai.components import Component
from chai.featurizers import Featurizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)


def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])

def semhash_tokenizer(text):
    tokens = text.split(" ")
    final_tokens = []
    for unhashed_token in tokens:
        hashed_token = "#{}#".format(unhashed_token)
        final_tokens += [''.join(gram)
                         for gram in list(find_ngrams(list(hashed_token), 3))]
    return final_tokens


class SemhashFeaturizer(Featurizer):
    name = "intent_featurizer_semhash"

    provides = ["text_features"]

    requires = []

    def __init__(self):
        self.vectorizer = None
        self.stemmer = None
        self.stop_words = None
        self.language = None
        self.language_mapping = None

    def train(self, training_data, config, **kwargs):
        logger.info("Extracting semhash features")

        self._set_language_mapping(config)
        self._set_language(config.get("language"), training_data.language)
        self._load_stemmer(self.language)
        self._load_stop_words(self.language)

        vectors = self._fit_vectorizer(training_data.training_examples, self.language)
        for training_instance, vec in zip(training_data.training_examples, vectors):
            combined_vec = self._combine_with_existing_text_features(training_instance, vec)
            training_instance.set("text_features", combined_vec)

    def process(self, message, **kwargs):
        clean_corpus = self._preprocess_corpus([message.text], self.language)
        combined_vec = self._combine_with_existing_text_features(message,
                                self.vectorizer.transform(clean_corpus).toarray()[0])
        message.set("text_features", combined_vec)

    def _set_language(self, config_lang, detected_lang):
        if config_lang is None and detected_lang is None:
            raise Exception('language was neither specified nor detected '
                            'while training. Consider setting the language '
                            'parameter while training.')
        self.language = config_lang if config_lang is not None else detected_lang

    def _fit_vectorizer(self, training_examples, language):
        corpus = []
        for training_instance in training_examples:
            corpus.append(training_instance.text)
        clean_corpus = self._preprocess_corpus(corpus, language)

        self.vectorizer = CountVectorizer(tokenizer=semhash_tokenizer)
        vectors = self.vectorizer.fit_transform(clean_corpus).toarray()
        return vectors

    def _load_stemmer(self, language):
        lang_code = self.language_mapping[language]
        self.stemmer = SnowballStemmer(lang_code)

    def _load_stop_words(self, language):
        lang_code = self.language_mapping[language]
        self.stop_words = set(stopwords.words(lang_code))

    def _preprocess_corpus(self, corpus, language):
        lang_code = self.language_mapping[language]
        clean_corpus = []
        for text in corpus:
            tokens = word_tokenize(text, lang_code)
            clean_tokens = self._remove_stopwords(tokens)
            clean_tokens = self._stem_tokens(clean_tokens)
            clean_corpus.append(" ".join(clean_tokens))
        return clean_corpus

    def _remove_stopwords(self, tokens):
        filtered_words = []
        for w in tokens:
            if w not in self.stop_words:
                filtered_words.append(w)
        return filtered_words

    def _stem_tokens(self, tokens):
        filtered_words = []
        for token in tokens:
            filtered_words.append(self.stemmer.stem(token))
        return filtered_words

    def _set_language_mapping(self, config):
        if "language_mapping" in config:
            self.language_mapping = config["language_mapping"]
        else:
            raise Exception("No language mapping provided.")

    @classmethod
    def load(cls, model_dir=None, model_metadata=None, cached_component=None, **kwargs):
        # type: (Text, Metadata, Optional[Component], **Any) -> NGramFeaturizer
        import cloudpickle

        if model_dir and model_metadata.get("intent_featurizer_semhash"):
            classifier_file = os.path.join(model_dir, model_metadata.get("intent_featurizer_semhash"))
            with io.open(classifier_file, 'rb') as f:   # pramga: no cover
                if PY3:
                    return cloudpickle.load(f, encoding="latin-1")
                else:
                    return cloudpickle.load(f)
        else:
            return SemhashFeaturizer()

    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]
        """Persist this model into the passed directory. Returns the metadata necessary to load the model again."""
        import cloudpickle

        classifier_file = os.path.join(model_dir, "intent_featurizer_semhash.pkl")
        with io.open(classifier_file, 'wb') as f:
            cloudpickle.dump(self, f)

        return {
            "intent_featurizer_semhash": "intent_featurizer_semhash.pkl"
        }
