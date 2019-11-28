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

df = pd.read_csv('corpus/sms_corpus/data.txt', engine='python', sep="<%>", header=None)

# Indonesian SMS Preprocessing
def convertTackyText(text):
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

def preproccess_text(text_messages):
    # change words to lower case
    processed = text_messages.lower()

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
    processed = convertTackyText(processed.str)

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

#Train the sentence tokenizer
f=open("corpus/indonesian_sent_tokenizer_corpus/indonesian-promotion-text.txt", "r")
if f.mode == 'r':
    train_text = preproccess_text(f.read())
f.close()

path = 'corpus/indonesian_sent_tokenizer_corpus/kompas/txt'
for foldername in os.listdir(path):
    new_path = path + '/' + foldername
    for filename in os.listdir(new_path):
        f=open(new_path + '/' + filename, "r")
        if f.mode == 'r':
            train_text = train_text + ' ' + preproccess_text(f.read())
        f.close()

path = 'corpus/indonesian_sent_tokenizer_corpus/tempo/txt'
for foldername in os.listdir(path):
    new_path = path + '/' + foldername
    for filename in os.listdir(new_path):
        f=open(new_path + '/' + filename, "r")
        if f.mode == 'r':
            train_text = train_text + ' ' + preproccess_text(f.read())
        f.close()

path = 'corpus/indonesian_sent_tokenizer_corpus/tempo/txt2'
for foldername in os.listdir(path):
    new_path = path + '/' + foldername
    for filename in os.listdir(new_path):
        f=open(new_path + '/' + filename, "r")
        if f.mode == 'r':
            train_text = train_text + ' ' + preproccess_text(f.read())
        f.close()

f=open("corpus/indonesian_sent_tokenizer_corpus/paragraf-promosi.txt", "r")
if f.mode == 'r':
    train_text = train_text + ' ' + preproccess_text(f.read())
f.close()

indonesian_sent_tokenizer = PunktSentenceTokenizer(train_text)

id_token = open('../sms_classifier_pickle/indonesian_sent_tokenizer.pickle', 'wb')
pickle.dump(indonesian_sent_tokenizer, id_token)
id_token.close

# create bag-of-words
all_words = []

for message in sms_data:
    words = word_tokenize(message)
    for w in words:
        all_words.append(w)
        
all_words = nltk.FreqDist(all_words)

# use the 1500 most common words as features
word_features = list(all_words.keys())[:1500]

fa = open('../sms_classifier_pickle/word_features.pickle', 'wb')
pickle.dump(word_features, fa)
fa.close

# # Uncomment to load last open word features
# word_features_f = open("word_features.pickle", "rb")
# word_features = pickle.load(word_features_f)
# word_features_f.close()

classes = df[0]
sms_data = preproccess_df(df[1])

encoder = LabelEncoder()
Y = encoder.fit_transform(classes)

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
    f = open('../sms_classifier_pickle/' + name + ' Classifier.pickle', 'wb')
    pickle.dump(classifier, f)
    f.close
    result = classifier.classify(find_features(preproccess_text('hey mau minta tolong dong bantuin')))
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