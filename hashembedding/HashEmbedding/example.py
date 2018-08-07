from layers import HashEmbedding, ReduceSum
from keras.layers import Input, Dense, Activation, Embedding
from keras.models import Model
import hashlib
import nltk
import keras
import numpy as np
from keras.callbacks import EarlyStopping
from sklearn.datasets import fetch_20newsgroups

def get_model(embedding, num_classes):
    input_words = Input([None], dtype='int32', name='input_words')

    x = embedding(input_words)
    x = ReduceSum()([x, input_words])

    x = Dense(50, activation='relu')(x)

    x = Dense(num_classes)(x)
    x = Activation('softmax')(x)
    model = Model(input=input_words, output=x)
    return model

def word_encoder(w, max_idx):
    # v = hash(w) #
    v = int(hashlib.sha1(w.encode('utf-8')).hexdigest(), 16)
    return (v % (max_idx-1)) + 1


if __name__ == '__main__':
    use_hash_embeddings = True
    embedding_size = 20
    num_buckets = 5000 # number of buckets in second hashing layer (hash embedding)
    max_words = 10**6  # number of buckets in first hashing layer
    max_epochs = 50
    num_hash_functions = 2

    if use_hash_embeddings:
        embedding = HashEmbedding(max_words, num_buckets, embedding_size, num_hash_functions=num_hash_functions)
    else:
        embedding = Embedding(max_words, embedding_size)

    # create dataset
    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    num_classes = len(categories)
    twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
    data = [nltk.word_tokenize(text)[0:150] for text in twenty_train.data]
    # data = [nltk.word_tokenize(text) for text in twenty_train.data]
    data_encoded = [[word_encoder(w, max_words) for w in text] for text in data]
    max_len = max([len(d) for d in data])

    # pad data
    data_encoded = [d+[0]*(max_len-len(d)) for d in data_encoded]

    idx_test = int(len(data_encoded)*0.5)
    data_encoded = np.vstack(data_encoded)
    targets = np.asarray(twenty_train.target, 'int32').reshape((-1,1))

    model = get_model(embedding, num_classes)
    metrics = ['accuracy']
    loss = 'sparse_categorical_crossentropy'
    model.compile(optimizer=keras.optimizers.Adam(),loss=loss, metrics=['accuracy'])

    print('Num parameters in model: %i' % model.count_params())
    model.fit(data_encoded[0:idx_test,:], targets[0:idx_test], validation_split=0.1,nb_epoch=max_epochs,
              callbacks=[EarlyStopping(patience=5)])

    test_result = model.test_on_batch(data_encoded[idx_test::, :], targets[idx_test::])
    for i, (name, res) in enumerate(zip(model.metrics_names, test_result)):
        print('%s: %1.4f' % (name, res))