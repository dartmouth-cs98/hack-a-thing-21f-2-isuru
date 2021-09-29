# Medical chatbot in Python


## What you built? 

This project attempted to build a functional chatbot that could respond to medical questions. Requests and responses are pre-recorded in the intents.json file and forms the corpus of this project. The project was inspired by Suds' and Mac's presentation on medical chatbots.

### How it works?

#### The corpus:

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

Here is some information on the corpus:

![alt text][corpus]

[corpus]: docs/corpus.png

#### The model:

The LTSM model is essentially a classification model. It takes tokenized, processed patterns and outputs a certain tag or class of responses. Helper functions in the invoked in chatapp.py uses the intents.json file to find the group of responses that match the class that the model outputs. A response is then picked randomly from this group and is output in the GUI.

The model technically achieved a 100% accuracy after being trained on 200 epochs.

Here is some information on the model:

![alt text][model]

[model]: docs/model.png

#### The GUI:

The GUI uses tkinter - a python library. It is a very simple GUI that logs the entire conversation that the user has with the bot.

### Running the application

Run the following command in the terminal or on VS Code.

```
bash bot_run.sh
```
This is what the app looks like:

![alt text][app2]

[app2]: docs/app2.png

Here is the performance of the app from a more buggy run:

![alt text][app]

[app]: docs/app.png

## Who Did What?

I worked on the project alone.

## What you learned

I learned a lot about building deep learning models and how to use one in applications once it has been constructed. This program was an effective introduction into keras and tensorflow. I also did not know anything about GUI programming in Python, and the tutorial did a good job of getting me started.

#### What worked?

The model ran well and the computation was relatively fast. It also seemed to be picking up the right tag each time the app was run.

The GUI lagged had a small lag when the first user question was sent but there were no problems after that. Not entirely sure why that keeps happening in every run.

The model also worked within the scope of medicine. For example, the bot would say that the question is out of the mandate of its function if a user inputs "what is the weather today?".

#### What didn't work?

a) The responses are chosen randomly from the output class. The problem with this approach is that a user could say "Hi, how are you?" and the bot would output "Hello, thanks for asking.". Which doesn't make any sense.

b) The GUI was very plain. tkinter does not allow for a lot of customization and it shows in the GUI. Since I learned how to create a deep learning model that could be used in applications, I believe that I could simply have a react / react native frontend. [This tutorial](https://www.tensorflow.org/js/tutorials/conversion/import_keras) shows how a keras model can be incorporated with react using tensorflow.js.

c) The depth of the corpus is incredibly limited. For example, when a user inputs "blood pressure check" the bot simply responds with "switching to blood pressure module". There is no detailed information provided. This is because the responses in intents.json is very short. There's two options here: manually add responses to the file or rely on other corpuses - possibly online. 

## Authors

I was the author of this project.

## Acknowledgments

This project is based on the [following tutorial](https://data-flair.training/blogs/python-chatbot-project/).  I had some experience in Python and machine learning so some of the code was done based on my intuition. However, I was guided by the tutorial and the code for the GUI is from the tutorial.
