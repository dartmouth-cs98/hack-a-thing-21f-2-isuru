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