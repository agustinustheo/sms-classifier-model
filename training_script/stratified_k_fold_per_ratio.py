import os
import re
import csv
import sys
import nltk
import random
import pickle
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.tokenize import PunktSentenceTokenizer, word_tokenize
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Indonesian SMS Preprocessing
def convert_tacky_text(text):
    words = word_tokenize(text)
    new_string = ''
    for msg in words:
        new_word = ''
        alpha_flag = False
        digit_flag = False
        for c in msg:
            if c.isalpha():
                alpha_flag = True
            elif c.isdigit():
                digit_flag = True
        
        if alpha_flag and digit_flag:
            msg = msg.lower()
            if msg[-4:] != 'ribu' and msg[-3:] != 'rbu' and msg[-2:] != 'rb':
                for c in msg:
                    if c == '1':
                        c = 'i'
                    elif c == '2':
                        c = 's'
                    elif c == '3':
                        c = 'e'
                    elif c == '4':
                        c = 'a'
                    elif c == '5':
                        c = 's'
                    elif c == '6':
                        c = 'g'
                    elif c == '7':
                        c = 't'
                    elif c == '8':
                        c = 'b'
                    elif c == '9':
                        c = 'g'
                    elif c == '0':
                        c = 'o'
                    new_word = new_word + c
        
        if new_word != '':
            new_string = new_string + new_word + ' '
        else:
            new_string = new_string + msg + ' '

    return new_string

def convert_tacky_text_df(df):
    for text in df:
        words = word_tokenize(text)
        new_string = ''
        for msg in words:
            new_word = ''
            alpha_flag = False
            digit_flag = False
            for c in msg:
                if c.isalpha():
                    alpha_flag = True
                elif c.isdigit():
                    digit_flag = True
            
            if alpha_flag and digit_flag:
                msg = msg.lower()
                if msg[-4:] != 'ribu' and msg[-3:] != 'rbu' and msg[-2:] != 'rb':
                    for c in msg:
                        if c == '1':
                            c = 'i'
                        elif c == '2':
                            c = 's'
                        elif c == '3':
                            c = 'e'
                        elif c == '4':
                            c = 'a'
                        elif c == '5':
                            c = 's'
                        elif c == '6':
                            c = 'g'
                        elif c == '7':
                            c = 't'
                        elif c == '8':
                            c = 'b'
                        elif c == '9':
                            c = 'g'
                        elif c == '0':
                            c = 'o'
                        new_word = new_word + c
            
            if new_word != '':
                new_string = new_string + new_word + ' '
            else:
                new_string = new_string + msg + ' '

        text = new_string
    return df

def preproccess_text(text_messages):
    # change words to lower case
    processed = text_messages.lower()
    
    # Remove tacky text
    processed = convert_tacky_text(processed)

    # Replace email addresses with 'emailaddress'
    processed = re.sub(r'^.+@[^\.].*\.[a-z]{2,}$', ' emailaddress ', processed)
        
    # Replace phone numbers (formats include paranthesis, spaces, no spaces, dashes) with 'phonenumber'
    processed = re.sub(r'(\()?(\+62|62|0)(\d{2,3})?\)?[ .-]?\d{2,4}[ .-]?\d{2,4}[ .-]?\d{2,4}', ' phonenumber ', processed)

    # Replace URLs with 'webaddress'
    processed = re.sub(r'[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)', ' webaddress ', processed)
    processed = processed.replace('http', '')
    processed = processed.replace('https', '')
    
    # Replace money symbols with 'moneysymbol' (£ can by typed with ALT key + 156)
    processed = re.sub(r'£|\$', 'moneysymbol ', processed)
    processed = processed.replace(' rp.', ' moneysymbol ')
    processed = processed.replace(' rp', ' moneysymbol ')
        
    # Replace numbers with 'number'
    processed = re.sub(r'\d+(\.\d+)?', ' number ', processed)

    # Remove punctuation
    processed = re.sub(r'[.,\/#!%\^&\*;:+{}=\-_`~()?]', ' ', processed)

    # Replace whitespace between terms with a single space
    processed = re.sub(r'\s+', ' ', processed)

    # Remove leading and trailing whitespace
    processed = re.sub(r'^\s+|\s+?$', '', processed)
    return processed

def preproccess_df(text_messages):
    # change words to lower case
    processed = text_messages.str.lower()
    
    # Remove tacky text
    processed = convert_tacky_text_df(processed)

    # Replace email addresses with 'emailaddress'
    processed = processed.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$', ' emailaddress ')
        
    # Replace phone numbers (formats include paranthesis, spaces, no spaces, dashes) with 'phonenumber'
    processed = processed.str.replace(r'(\()?(\+62|62|0)(\d{2,3})?\)?[ .-]?\d{2,4}[ .-]?\d{2,4}[ .-]?\d{2,4}', ' phonenumber' )

    # Replace URLs with 'webaddress'
    processed = processed.str.replace(r'[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)', ' webaddress ')
    processed = processed.str.replace('http', '')
    processed = processed.str.replace('https', '')
    
    # Replace money symbols with 'moneysymbol' (£ can by typed with ALT key + 156)
    processed = processed.str.replace(r'£|\$', ' moneysymbol ')
    processed = processed.str.replace(' rp.', ' moneysymbol ')
    processed = processed.str.replace(' rp', ' moneysymbol ')
        
    # Replace numbers with 'number'
    processed = processed.str.replace(r'\d+(\.\d+)?', ' number ')

    # Remove punctuation
    processed = processed.str.replace(r'[.,\/#!%\^&\*;:{}=\-_`~()?]', ' ')

    # Replace whitespace between terms with a single space
    processed = processed.str.replace(r'\s+', ' ')

    # Search for all non-letters and replace all non-letters with spa
    processed = processed.str.replace("[^a-zA-Z]", " ")

    # Remove leading and trailing whitespace
    processed = processed.str.replace(r'^\s+|\s+?$', '')
    
    return processed

# The find_features function will determine which of the 1500 word features are contained in the review
def find_features(message):
    words = word_tokenize(message)
    features = {}
    for word in word_features:
        features[word] = (word in words)

    return features

# Calculate total accuracy
def calculate_accuracy(classifier, sms_data, label):
    if len(sms_data) != len(label):
        return

    test_results = []
    for x in sms_data:
        result = classifier.classify(find_features(preproccess_text(x)))
        test_results.append(result)

    return accuracy_score(test_results, label)

# Write headers
def write_header(filename):
    if not os.path.exists('results'):
        os.mkdir('results')
    f = open('results/' + filename + ' K-Fold Results.csv', 'w')
    with f:
        fnames = ['ratio', 'k-fold 1', 'k-fold 2', 'k-fold 3', 'k-fold 4', 'k-fold 5', 'k-fold 6', 'k-fold 7', 'k-fold 8', 'k-fold 9', 'k-fold 10']
        writer = csv.DictWriter(f, fieldnames=fnames)    
        writer.writeheader()

# Write headers
def write_body(filename, data):
    if not os.path.exists('results'):
        os.mkdir('results')
    f = open('results/' + filename + ' K-Fold Results.csv', 'a')
    with f:
        fnames = ['ratio', 'k-fold 1', 'k-fold 2', 'k-fold 3', 'k-fold 4', 'k-fold 5', 'k-fold 6', 'k-fold 7', 'k-fold 8', 'k-fold 9', 'k-fold 10']
        writer = csv.DictWriter(f, fieldnames=fnames)    
        writer.writerow(data)

# Uncomment to load last open word features
word_features_f = open("word_features.pickle", "rb")
word_features = pickle.load(word_features_f)
word_features_f.close()

# Define models to train
names = ["K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression", "XGBoost", "SGD Classifier",
        "Naive Bayes", "SVM Linear"]

classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    XGBClassifier(),
    SGDClassifier(max_iter = 100),
    MultinomialNB(),
    SVC(kernel = 'linear')
]

models = zip(names, classifiers)

for name, model in models:
    # Create file and write CSV header
    write_header(name)
    # Define data ratios
    dataset_ratios = ['1;1;1', '1;1;2', '1;2;1', '2;1;1', '5.5;3.3;1']
    for dataset_ratio in dataset_ratios:
        print("===========================================================================================")
        df = []
        if dataset_ratio != '5.5;3.3;1':
            df = pd.read_csv('corpus/sms_corpus/data_'+dataset_ratio+'.txt', engine='python', sep="<%>", header=None)
        else:
            df = pd.read_csv('corpus/sms_corpus/data.txt', engine='python', sep="<%>", header=None)

        classes = df[0]
        sms_data = preproccess_df(df[1])

        encoder = LabelEncoder()
        Y = encoder.fit_transform(classes)

        # Define a seed for reproducibility
        seed = 1
        np.random.seed = seed
        # Set dataset ratio
        body = {}
        body['ratio'] = dataset_ratio
        
        # Initialize K Fold
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
        k_fold_index = 1
        for train_index, test_index in skf.split(sms_data, Y):
            train_messages = list(zip(sms_data[train_index], Y[train_index]))
            np.random.shuffle(train_messages)

            # call find_features function for each SMS message
            featuresets = [(find_features(text), label) for (text, label) in train_messages]

            # split the data into training and testing datasets
            training = featuresets

            nltk_model = SklearnClassifier(model)
            classifier = nltk_model.train(training)    
            accuracy = calculate_accuracy(classifier, sms_data[test_index], Y[test_index]) * 100     
            print(name, "Classifier; Ratio", dataset_ratio, "; Iteration", k_fold_index, "; Accuracy", "{:.2f}".format(accuracy))                       
            # Set k-fold iteration accuracy
            body['k-fold ' + str(k_fold_index)] = "{:.2f}".format(accuracy)
            k_fold_index += 1
        
        # Write iterations
        write_body(name, body)
