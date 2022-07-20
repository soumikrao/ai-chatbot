import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle

#-------------opening and processing data --------------------------------

with open("intents.json") as file:  # opening the json file
    data = json.load(file)  # loading the data

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    # print(data['intents'])  # printing as a check
    words = []  # all the words
    labels = []  # all the tags
    docs_x = []  # all the patterns
    docs_y = []  # list of tags (may repeat)

    for intent in data['intents']:
        # print(intent)  # prints all the intents
        for pattern in intent['patterns']:
            wrds = nltk.word_tokenize(pattern)  # gives us list with all the words in the intents file
            words.extend(wrds)  # adding all those words
            docs_x.append(wrds)
            docs_y.append(intent['tag'])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])  # we will get all the different tags

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    #------------- end  --------------------------------

    #  neural networks understand only numbers, so we need to convert the words into numbers

    #------------- Training the model  --------------------------------

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]
    for x, doc in enumerate(docs_x):  # enumerate gives the iteration counter
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)  # if the word exists in input, append 1
            else:
                bag.append(0)  # if word does not exist, append 0

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1  # look through label list, see where that tag is and set it to 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)  # taking the lists and storing as arrays
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)


#--------------- The whole AI model ----------------

tensorflow.reset_default_graph()  # resetting, usually done everywhere

# making the neural network

net = tflearn.input_data(shape=[None, len(training[0])])  # shape of the input
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")  # softmax gives a probability to the tag based on the input
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)  # epoch sees the words the specified number of times
    model.save("model.tflearn")

#--------------- End ----------------

#--------------- Classes for chat interface --------------------


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:  # if current word matches the word in our sentence
                bag[i] = 1  # appends 1 if the word exists

    return numpy.array(bag)


def chat():
    print("Start chatting with the bot! (Type quit to stop)")
    while True:
        inp = input("You: ")
        if inp.lower() == 'quit':
            break

        result = model.predict([bag_of_words(inp, words)])[0]  # this will show all the probabilities of different neurons
        result_index = numpy.argmax(result)  # gives the index of the greatest probability
        tag = labels[result_index]

        if result[result_index] > 0.6:
            for tg in data['intents']:
                if tg["tag"] == tag:
                    responses = tg['responses']

            print(random.choice(responses))
        else:
            print("I don't understand")


chat()  # calling the chat functions
