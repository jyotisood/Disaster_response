# Disaster_response
This project is a part of Udacity Data Science Nanodegree curriculum where natural language processing is utilized to predict types of disaster messages
#### Overview
The data of this project was provided by FigureEight. The data consists of tweets and texts that were sent during real world disasters and can be labeled into at least one of 36 categories.
#### Libraries Required
The code was written in python 3 and requires the following packages: re, sys, json, plotly, pandas, nltk, flask, sklearn, sqlalchemy and pickle. The following nltk data must be downloaded: punkt, stopwords, wordnet, and averaged_perceptron_tagger.

### Three Main steps of the project:
#### Create ETL pipeline and upload cleansed data to a database
This step involved merging two datasets of messages and categories together, cleaning of data, re encoding and splitting category column to create new columns in the dataset. It resulted in 36 new columns. Further these category values were binaraized for easy calculations.
#### Create Machine Learning Pipeline 
After loading the cleaned, the data was tokenized, target and feature variables defined, split into test and train data. 
Then first machine learning pipeline was created using CountVectorizer, TfidfTransformer and MultiOutputClassifier as the data had multiple target variables instead of binary targets to classify.
The model was trained and tested and the resulting f1 scores of all features were stored in a list.
Then an improved model using grid search cv was fine-tuned upon. F1 scores of both models were compared to see if any difference existed between base model and improved model. The best model was selected and stored in a sav file 
Notes:
The following paths were used in the terminal respectively
To run ETL pipeline: python data/process_data.py data/messages.csv data/categories.csv data/Disasters.db
To run ML pipeline:
python models/train_classifier.py models/Disasters.db models/disaster_message_model.sav
For running the pipeline in terminal, the train_classifer.py included only the model used finally and excluded the other model to optimize run time.
#### Running the app
The following was run in the terminal \app\run.py 
To run the web app from Udacity Terminal
1. Run your app with cd app and python run.py commands
2. Open another terminal and type env|grep WORK this will give you the spaceid it will start with viewXXXXXXX and some characters after that
3. Now open your browser window and type https://viewa7a4999b-3001.udacity-student-workspaces.com, replace the whole viewa7a4999b with your space id you got in the step 2
4. Press enter and the app should now run for you
In the app, if you enter some message into the text bar, you can see what model predicts. An easy one to use is "There is a hurricane moving to the city", which gets classified as Related , Weather related and Storm.
