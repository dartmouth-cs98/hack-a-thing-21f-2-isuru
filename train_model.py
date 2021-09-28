import json

words = []
classes = []
documents = []

# loading pre recorded responses stored in the intents.json file
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)

print("Successfully loaded data")

print("Starting preprocessing")

import nltk

print("Tokenizing")
for intent in intents['intents']:
    for pattern in intent['patterns']:

        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])



# Stemming etc.

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import pickle

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

print (len(documents), "documents")
# classes = intents
print (len(classes), "classes", classes)
# words = all words, vocabulary
print (len(words), "unique lemmatized words", words)

# writing .pkl files intro repo
pickle.dump(words, open('words.pkl','wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

print("Finished preprocessing")

# create training data
# the input for the model is a pattern and the output is the 
# class it belongs to.

training_data = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(w.lower()) for w in pattern_words]

    # loops through corpus of words, if that word exists in the pattern words, it gets 1 else 0
    for w in words:
        if w in pattern_words:
            bag.append(1)
        else:
            bag.append(0)
    

    # 0 for all classes and 1 for current class. Class = tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training_data.append([bag, output_row])

import random
import numpy as np

# Suppressing the warning: train_model.py:84: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. 
# If you meant to do this, you must specify 'dtype=object' when creating the ndarray.

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

random.shuffle(training_data)
training = np.array(training_data)

train_x = list(training[:,0])
train_y = list(training[:,1])

print("Training data created")

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))
# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#fitting and saving the model 
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)
print("model created")
