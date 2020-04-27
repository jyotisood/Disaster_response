import sys
import re
import nltk
import pickle
import pandas as pd
import numpy as np

from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.metrics import precision_recall_fscore_support

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


def load_data(database_filepath):
    """Loads sqlite database into a dataframe then defines X and Y variables
    Args:
        database_filename: string - name of database (ex. 'Messages.db')
    Returns:
        X: dataframe - df containing features (the messages)
        Y: dataframe - df containing labels (the categories)
        category_names: column names of Y df (cateegory names)
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Disasters', engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    """Tokenizes text by performing: case normalization, punctuation removal,
    word tokenization, stop word removal, and stemming.
    Args:
        text: string - messages to be tokenized
    Returns:
        words: list of strings - tokenized words of text
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens




def build_model():
    """
    Performs machine learning pipeline
    Args:
        None
    Returns:
        model: trained model
    """
    # Build ML pipeline using random forest classifier
    model = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(
        n_estimators=100, min_samples_split=2)))
    ])

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates model by calculating f1-score, precision, and recall
    Args:
        model: trained model
        X_test: list - train-test split of inputs
        Y_test: list - train-test split of inputs
        category_names: column names of Y df (cateegory names)
    Prints:
        report: dataframe - accuracy report
    """

    # Predict labels using model
    y_pred1 = model.predict(X_test)

    # Generate accuracy report
    accuracy = [[(y_pred1[:, i] == Y_test.values[:, i]).mean(),
             *precision_recall_fscore_support(
                 Y_test.values[:, i], y_pred1[:, i], average='weighted')]
            for i in range(y_pred1.shape[1])]
    accuracy = np.array(accuracy)[:, :-1]
    accuracy = (accuracy * 10000).astype(int) / 100
    scores1= pd.DataFrame( data=accuracy,  index=list(Y_test),  columns=['Accuracy', 'Precision', 'Recall', 'F-score'])
    print(scores1)
    return scores1
    


def save_model(model, model_filepath):
    """
    Saves model as pickle file
    Args:
        model: trained model
        model_filepath: string - filepath of where model will be saved
    Returns:
        None
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()