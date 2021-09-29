# Hack Technology / Project Attempted


## What you built? 

This project attempted to build a functional chatbot that could respond to medical questions. Requests and responses are pre-recorded in the intents.json file and forms the corpus of this project. The project was inspired by Suds' and Mac's presentation on medical chatbots.

### How it works?

Each object in the intents.json file consists of potential questions that a user may ask and a group of responses to those questions that is grouped under a theme. The following below is a trimmed version of the intents.json file. 

```
{"intents": [
        {"tag": "greeting",
         "patterns": ["Hi there", "How are you", "Is anyone there?","Hey","Hola", "Hello", "Good day"],
         "responses": ["Hello, thanks for asking", "Good to see you again", "Hi there, how can I help?"],
         "context": [""]
        }]

```
The patterns form the corpus for the deep learning model I will touch on in a second. The corpus is written into a pickle file - words.pkl. The tags form the classes for the model and is saved as classes.pkl.

#### The model

The LTSM model is essentially a classification model. It takes tokenized, processed patterns and outputs a certain tag or class of responses. Helper functions in the invoked in chatapp.py uses the intents.json file to find the group of responses that match the class that the model outputs. A response is then picked randomly from this group and is output in the GUI.

The model technically achieved a 100% accuracy after being trained on 200 epochs.

#### The GUI

The GUI uses tkinter - a python library. It is a very simple GUI that logs the entire conversation that the user has with the bot.

Here are a few screenshots of the application:


Include some screenshots.
[How?](https://help.github.com/articles/about-readmes/#relative-links-and-image-paths-in-readme-files)

## Who Did What?

I worked on the project alone.

## What you learned

a) The responses are chosen randomly from the output class. The problem with this approach is that a user could say "Hi, how are you?" and the bot would output "Hello, thanks for asking.". Which doesn't make any sense.
b)

## Authors

I was the author of this project.

## Acknowledgments

This project is based on the following tutorial: https://data-flair.training/blogs/python-chatbot-project/
I had some experience in Python and machine learning so some of the code was done based on my intuition. However, I was guided by the tutorial and the code for the GUI is from the tutorial.
