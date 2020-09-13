
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

from flask_cors import CORS, cross_origin
from flask import Flask, jsonify, request
from flask import Flask, render_template, request
from keras.models import load_model
from keras.optimizers import SGD


import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework import ops
import numpy as np
import pandas as pd
import pickle
import json
import random



data = pickle.load(open( "chappy-bot-data.pkl", "rb"))
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data ['train_y']

with open('intents.json') as json_data:
    intents = json.load(json_data)


model = keras.Sequential()
model.add(keras.layers.Dense(128, input_shape=(len (train_x[0]),), activation ='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])



def clean_up_sentence(sentence):
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
                    print("found in bag: %s" % w)
    return (np.array(bag))

def predict_class(sentence, model):
    p = bow (sentence, words,show_details=False)
    res = model.predict(np.array([p])) [0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
        return return_list
                

p = bow("Load bood pessure for patient", words)
print (p)
print (classes)


global graph
graph = ops.get_default_graph()

with open(f'chappy-bot-model.h5', 'rb') as f:
    model = keras.models.load_model('chappy-bot-model.h5')

coughQuestionAsked = False
feverQuestionAsked = False
tirednessQuestionAsked = False
difficultyInBreathingQuestinAsked = False
noneResponse = False

hasCough = False
hasFever = False
hasTiredness = False
hasDifficultyInBreathing = False
hasNone = False

def assessmentCheck(response):
    global coughQuestionAsked, feverQuestionAsked, tirednessQuestionAsked, difficultyInBreathingQuestinAsked, hasCough, hasFever, hasTiredness, hasDifficultyInBreathing, noneResponse, hasNone
    
    feedback = "continue"
    if coughQuestionAsked == False or feverQuestionAsked == False or tirednessQuestionAsked == False or difficultyInBreathingQuestinAsked == False or noneResponse == False:
        if response == 'cough':
            coughQuestionAsked = True
            hasCough = True
            feedback = "Which of these other symptoms do you have: Fever, Tiredness, Difficulty in Breathing"

        if response == 'Fever':
            feverQuestionAsked = True
            hasFever = True
            feedback = "Which of these other symptoms do you have: Cough, Tiredness, Difficulty in Breathing"

        if response == 'Tiredness':
            tirednessQuestionAsked == True
            hasTiredness = True
            feedback = "Which of these other symptoms do you have: Cough, Fever, Difficulty in Breathing"

        if response == 'Difficulty in breathing':
            difficultyInBreathingQuestinAsked == True
            hasDifficultyInBreathing = True
            feedback = "Which of these other symptoms do you have: Cough, Fever, Tiredness"

        if response == 'None':
            noneResponse = True
            hasNone = True
            feedback = "Great"


    if hasCough == True and hasFever == True and hasTiredness == True and hasDifficultyInBreathing == True:
        feedback = "Warning!!! You might have covid. Go for test Immediately - High risk 4"
    if hasCough == True and hasFever == True and hasTiredness == True and hasDifficultyInBreathing ==  False:
        feedback = "Warning!!! You might have covid. Go for test Immediately high risk 4"
    if hasCough == True and hasFever == True and hasTiredness == False and hasDifficultyInBreathing == True:
        feedback = "Warning!!! You might have covid. Go for test Immediately high risk 4"
    if hasCough == True and hasFever == False and hasTiredness == True and hasDifficultyInBreathing == True:
        feedback = "Warning!!! You might have covid. Go for test Immediately high risk 4"
    if hasCough == False and hasFever == True and hasTiredness == True and hasDifficultyInBreathing == True:
        feedback = "Warning!!! You might have covid. Go for test Immediately high risk 4"
    if hasCough == True and hasFever == True and hasTiredness == False and hasDifficultyInBreathing == False:
        feedback = "Cough and Fever are strong symptoms of the COVID-19 disease. I advise you to stay at home and self-isolate"
    if hasCough == True and hasFever == False and hasTiredness == True and hasDifficultyInBreathing == False:
        feedback = "Cough and Tiredness are strong symptoms of the COVID-19 disease. I advise you to stay at home and self-isolate"
    if hasCough == True and hasFever == False and hasTiredness == False and hasDifficultyInBreathing == True:
        feedback = "Cough and Difficulty in Breathing are strong symptoms of the COVID-19 disease. I advise you to stay at home and self-isolate"
    if hasCough == False and hasFever == True and hasTiredness == True and hasDifficultyInBreathing == False:
        feedback = "Fever and Tiredness are strong symptoms of the COVID-19 disease. I advise you to stay at home and self-isolate"
    if hasCough == False and hasFever == True and hasTiredness == False and hasDifficultyInBreathing == True:
        feedback = "Fever and Difficulty in Breathing are strong symptoms of the COVID-19 disease. I advise you to stay at home and self-isolate"
    if hasCough == False and hasFever == False and hasTiredness == True and hasDifficultyInBreathing == True:
        feedback = "Tiredness and Difficulty in Breathing are strong symptoms of the COVID-19 disease. I advise you to stay at home and self-isolate"
    
    if hasCough == True and hasNone == True:
        feedback = "Sometimes Coughs could be a little bother. Wear your Face Mask while in the college at all times and see a Pharmacist."
    if hasFever == True and hasNone == True:
        feedback = "Sorry about the Fever. I advise you stay at home and call a Doctor."
    if hasTiredness == True and hasNone == True:
        feedback ="Sometimes I get binary tired too. I advise you stay at home and call a Doctor."
    if hasDifficultyInBreathing == True and hasNone == True:
        feedback = "Sometimes I also find it hard to breath with all these algorithm processing. But, I advise you wear a Face Mask while in the college at all times and see a Pharmacist."

    return feedback


def getResponse(ints, intents_json):
    tag = ints[0] ['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if (i ['tag'] == tag):
            print(i['tag'])
            print(tag)

            result = random.choice(i['responses'])

            feed = "continue"
            if i['tag']== "cough" or i['tag']== "Fever" or i['tag']=="Tiredness" or i['tag']=="Difficulty in breathing" or i['tag']=="None":
                feed = assessmentCheck(i['tag'])

            if feed != "continue":
                result = feed
           
            print (result)
            break
    return result



def chatbot_response(text):
    ints = predict_class(text, model)
    print(ints)
    res = getResponse (ints, intents)
    return res


app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
# def classify():
    
    userText = request.args.get('msg')

    return str(chatbot_response(userText))
    # return str("Hello")

    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000 )  

