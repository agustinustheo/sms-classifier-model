import os
import re
import sys
import nltk
import random
import pickle
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.tokenize import PunktSentenceTokenizer, word_tokenize
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

df = pd.read_csv('sms_classifier_corpus/data.txt', engine='python', sep="<%>", header=None)

def preproccess_text(text_messages):
    # Replace email addresses with 'almtemail'
    processed = re.sub(r'^.+@[^\.].*\.[a-z]{2,}$', 'almtemail', text_messages)
        
    # Replace phone numbers (formats include paranthesis, spaces, no spaces, dashes) with 'nmrtlpn'
    processed = re.sub(r'(\+62 ((\d{3}([ -]\d{3,})([- ]\d{4,})?)\|(\d+)))\|(\(\d+\) \d+)\|\d{3}( \d+)+|(\d+[ -]\d+)\|\d+', 'nmrtlpn', processed)

    # Replace URLs with 'almtweb'
    processed = re.sub(r'[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)', 'almtweb', processed)
    processed = processed.replace('http', '')
    processed = processed.replace('https', '')

    # change words to lower case - Hello, HELLO, hello are all the same word
    processed = processed.lower()
    
    # Replace money symbols with 'symbuang' (£ can by typed with ALT key + 156)
    processed = re.sub(r'£|\$', 'symbuang ', processed)
    processed = processed.replace(' rp.', ' symbuang ')
    processed = processed.replace(' rp', ' symbuang ')
        
    # Replace numbers with 'noomr'
    processed = re.sub(r'\d+(\.\d+)?', 'noomr', processed)

    # Remove punctuation
    processed = re.sub(r'[.,\/#!%\^&\*;:+{}=\-_`~()?]', ' ', processed)
    processed = re.sub(r'\s[a-z]\s', '', processed)

    # Replace whitespace between terms with a single space
    processed = re.sub(r'\s+', ' ', processed)

    # Remove leading and trailing whitespace
    processed = re.sub(r'^\s+|\s+?$', '', processed)
    return processed

def preproccess_df(text_messages):
    # change words to lower case - Hello, HELLO, hello are all the same word
    processed = text_messages.str.lower()

    # Replace email addresses with 'almtemail'
    processed = processed.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$', 'almtemail')
        
    # Replace phone numbers (formats include paranthesis, spaces, no spaces, dashes) with 'nmrtlpn'
    processed = processed.str.replace(r'(\+62 ((\d{3}([ -]\d{3,})([- ]\d{4,})?)\|(\d+)))\|(\(\d+\) \d+)\|\d{3}( \d+)+\|(\d+[ -]\d+)\|\d+', 'nmrtlpn')

    # Replace URLs with 'almtweb'
    processed = processed.str.replace(r'[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)', 'almtweb')
    processed = processed.str.replace('http', '')
    processed = processed.str.replace('https', '')
    
    # Replace money symbols with 'symbuang' (£ can by typed with ALT key + 156)
    processed = processed.str.replace(r'£|\$', 'symbuang ')
    processed = processed.str.replace(' rp.', ' symbuang ')
    processed = processed.str.replace(' rp', ' symbuang ')
        
    # Replace numbers with 'noomr'
    processed = processed.str.replace(r'\d+(\.\d+)?', 'noomr')

    # Remove punctuation
    processed = processed.str.replace(r'[.,\/#!%\^&\*;:{}=\-_`~()?]', ' ')
    processed = processed.str.replace(r'\s[a-z]\s', '')

    # Replace whitespace between terms with a single space
    processed = processed.str.replace(r'\s+', ' ')

    # Remove leading and trailing whitespace
    processed = processed.str.replace(r'^\s+|\s+?$', '')
    
    return processed

#Train the sentence tokenizer
f=open("indonesian_sent_tokenizer_corpus/indonesian-promotion-text.txt", "r")
if f.mode == 'r':
    train_text = preproccess_text(f.read())
f.close()

path = 'indonesian_sent_tokenizer_corpus/tempo/txt2'
for foldername in os.listdir(path):
    new_path = path + '/' + foldername
    for filename in os.listdir(new_path):
        f=open(new_path + '/' + filename, "r")
        if f.mode == 'r':
            train_text = train_text + ' ' + preproccess_text(f.read())
        f.close()

f=open("indonesian_sent_tokenizer_corpus/paragraf-promosi.txt", "r")
if f.mode == 'r':
    train_text = train_text + ' ' + preproccess_text(f.read())
f.close()

indonesian_sent_tokenizer = PunktSentenceTokenizer(train_text)

classes = df[0]
sms_data = preproccess_df(df[1])

encoder = LabelEncoder()
Y = encoder.fit_transform(classes)

# create bag-of-words
all_words = []

for message in sms_data:
    words = word_tokenize(message)
    for w in words:
        all_words.append(w)
        
all_words = nltk.FreqDist(all_words)

# use the 1500 most common words as features
word_features = list(all_words.keys())[:1500]

# The find_features function will determine which of the 1500 word features are contained in the review
def find_features(message):
    words = word_tokenize(message)
    features = {}
    for word in word_features:
        features[word] = (word in words)

    return features

# Now lets do it for all the messages
messages = list(zip(sms_data, Y))

# define a seed for reproducibility
seed = 1
np.random.seed = seed
np.random.shuffle(messages)

# call find_features function for each SMS message
featuresets = [(find_features(text), label) for (text, label) in messages]


# we can split the featuresets into training and testing datasets using sklearn
from sklearn import model_selection

# split the data into training and testing datasets
training = featuresets

# Define models to train
names = ["K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression", "SGD Classifier",
         "Naive Bayes", "SVM Linear"]

classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    SGDClassifier(max_iter = 100),
    MultinomialNB(),
    SVC(kernel = 'linear')
]

models = zip(names, classifiers)

normal_msg = 0
promo_msg = 0
spam_msg = 0

for name, model in models:
    nltk_model = SklearnClassifier(model)
    classifier = nltk_model.train(training)
    result = classifier.classify(find_features(preproccess_text('hey')))
    if result == 0:
        normal_msg = normal_msg + 1
    elif result == 1:
        promo_msg = promo_msg + 1
    elif result == 2:
        spam_msg = spam_msg + 1
    
if normal_msg >= promo_msg and normal_msg >= spam_msg:
    best_result = "normal message"
    confidence = normal_msg / (normal_msg + promo_msg + spam_msg)
elif promo_msg >= normal_msg and promo_msg >= spam_msg:
    best_result = "promotion message"
    confidence = promo_msg / (normal_msg + promo_msg + spam_msg)
elif spam_msg >= normal_msg and spam_msg >= promo_msg:
    best_result = "spam message"
    confidence = spam_msg / (normal_msg + promo_msg + spam_msg)

print("Algorithm Confidence = {}".format(confidence*100))
print("Model thinks this is a {}".format(best_result))

print(preproccess_text('+62 813-444-5555'))