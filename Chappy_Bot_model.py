import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import pandas as pd
import pickle
import random
import tempfile

import json
with open('intents.json') as json_data:
    intents = json.load(json_data)


words = []
classes = []
documents = []
ignore_words = []

for intent in intents ['intents']:
    for pattern in intent ['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

            words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
            words = sorted(list(set(words)))
            classes = sorted(list(set(classes)))
            print(len(documents), "documents")
            print (len(classes), "classes", classes)
            print (len(words), "unique stemmed words", words)

            pickle.dump(words, open('words.pkl', 'wb'))
            pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]

    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1

        training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])
print ("Training data created")


model = keras.Sequential()
model.add(keras.layers.Dense(128, input_shape=(len (train_x[0]),), activation ='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=50, batch_size=5, verbose=1)

def clean_up_sentence (sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:

                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))


p = bow("How are you today", words)
print (p)
print (classes)

inputvar = pd.DataFrame([p], dtype=float, index=['input'])
print(model.predict(inputvar))

model.save("chappy-bot-model.h5", hist)

pickle.dump({'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}, open ( "chappy-bot-data.pkl", "wb"))