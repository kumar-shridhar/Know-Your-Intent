import tensorflow as tf
import numpy as np
import w2v
import csv
import random
import re

def nn(data, neurons_per_layer):
    layers = [data]
    for x in range(len(neurons_per_layer)):
        layers.append(tf.layers.dense(inputs=layers[-1], units=neurons_per_layer[x], activation=tf.nn.relu))
    return layers[-1]

epochs = 1000

window = 3

vector_size = 300

words_training = []
words_valid = []

with open("ask_ubuntu_data.csv") as csv_file:
    reader = csv.reader(csv_file, delimiter = ",")
    for row in reader:
        s = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", row[0])
        if random.random() > 0.1:
            words_training.append(s.split(" "))
        else:
            words_valid.append(s.split(" "))

with open("yahoo_answers_csv/train.csv") as csv_file:
    reader = csv.reader(csv_file, delimiter = ",")
    i = 0
    for row in reader:
        i+=1
        for sentence in row[1:]:
            s = re.sub(r"\\[-()\"#/@;:<>{}`+=~|.!?,]", "", sentence)
            if random.random() > 0.1:
                words_training.append(s.split(" "))
            else:
                words_valid.append(s.split(" "))
        if i > 1000:
            break

data_training = []
data_valid = []

data_training_x = []
data_training_y = []
data_valid_x = []
data_valid_y = []

model = w2v.Model("Google")

#print(words_training)

for sentence in words_training:
    s = []
    for word in sentence:
        v = model.get_vector(word, defaultToZero = False)
        if v is not None:
            s.append(v)
    if len(s) > 1:        
        data_training.append(s)

for sentence in words_valid:
    s = []
    for word in sentence:
        v = model.get_vector(word, defaultToZero = False)
        if v is not None:
            s.append(v)
    if len(s) > 1:
        data_valid.append(s)

model = None

for vectors in data_training:
    for i in range(len(vectors)):
        pre = vectors[max(0,i-window):i]
        post = vectors[i+1:min(len(vectors)+1, i+1+window)]
        prepost = pre+post
        if len(prepost) == 0:
            print(vectors)
        x = np.zeros(vector_size * window* 2)
        prepost = np.concatenate(prepost)
        x[:prepost.size] = prepost
        data_training_x.append(x)
        data_training_y.append(vectors[i])


for vectors in data_valid:
    for i in range(len(vectors)):
        pre = vectors[max(0,i-window):i]
        post = vectors[i+1:min(len(vectors), i+1+window)]
        prepost = pre+post
        x = np.zeros(vector_size * window* 2)
        prepost = np.concatenate(prepost)
        x[:prepost.size] = prepost
        data_valid_x.append(x)
        data_valid_y.append(vectors[i])


neurons_per_layer = [vector_size * window * 2, 1000, 300]

x = tf.placeholder('float', [None, vector_size * window * 2])
y = tf.placeholder('float')

value = nn(x, neurons_per_layer)

cost = tf.reduce_mean(tf.squared_difference(value, y))

optimizer = tf.train.AdamOptimizer(0.0001).minimize(loss = cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        costing = 0
        for i in range(len(data_training_x)):
            inp = [data_training_x[i]]
            out = [data_training_y[i]]
            _, c = sess.run([optimizer, cost], feed_dict = {x: inp, y: out})
            costing += c
        print("Epoch cost = {}".format(costing))
        correctness = 0
        for i in range(len(data_valid_x)):
            inp = [data_valid_x[i]]
            out = [data_valid_y[i]]
            val = sess.run([value], feed_dict = {x: inp})
            dist = np.linalg.norm(out[0]-val[0])
            correctness += dist
        print("Epoch wrongness = {}".format(correctness))
