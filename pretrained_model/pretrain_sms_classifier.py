import os
import re
import sys
import nltk
from nltk.tokenize import PunktSentenceTokenizer, word_tokenize
from sklearn.preprocessing import LabelEncoder
import random
import pickle
import pandas as pd
import numpy as np

df = pd.read_csv('sms_classifier_corpus/data.txt', engine='python', sep="<%>", header=None)

def preproccess_text(text_messages):
    # change words to lower case - Hello, HELLO, hello are all the same word
    processed = text_messages.lower()

    # Replace email addresses with 'almtemail'
    processed = re.sub(r'^.+@[^\.].*\.[a-z]{2,}$', 'almtemail', processed)

    # Replace URLs with 'almtweb'
    processed = re.sub(r'[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)', 'almtweb', processed)
    processed = processed.replace('http', '')
    processed = processed.replace('https', '')
    
    # Replace money symbols with 'symbuang' (£ can by typed with ALT key + 156)
    processed = re.sub(r'£|\$', 'symbuang ', processed)
    processed = processed.replace(' rp.', ' symbuang ')
    processed = processed.replace(' rp', ' symbuang ')
        
    # Replace 10 digit phone numbers (formats include paranthesis, spaces, no spaces, dashes) with 'phonenumber'
    processed = re.sub(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$', 'nmrtlpn', processed)
        
    # Replace numbers with 'noomr'
    processed = re.sub(r'\d+(\.\d+)?', 'noomr', processed)

    # Remove punctuation
    processed = re.sub(r'[.,\/#!%\^&\*;:{}=\-_`~()?]', ' ', processed)
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

    # Replace URLs with 'almtweb'
    processed = processed.str.replace(r'[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)', 'almtweb')
    processed = processed.str.replace('http', '')
    processed = processed.str.replace('https', '')
    
    # Replace money symbols with 'symbuang' (£ can by typed with ALT key + 156)
    processed = processed.str.replace(r'£|\$', 'symbuang ')
    processed = processed.str.replace(' rp.', ' symbuang ')
    processed = processed.str.replace(' rp', ' symbuang ')
        
    # Replace 10 digit phone numbers (formats include paranthesis, spaces, no spaces, dashes) with 'phonenumber'
    processed = processed.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$', 'nmrtlpn')
        
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
print('Number of words: {}'.format(len(all_words)))
print('Most common words: {}'.format(all_words.most_common(10)))

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
training, testing = model_selection.train_test_split(featuresets, test_size = 0.25, random_state=seed)

print(len(training))
print(len(testing))


# We can use sklearn algorithms in NLTK
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC

model = SklearnClassifier(SVC(kernel = 'linear'))

# train the model on the training data
a = model.train(training)

# and test on the testing dataset!
accuracy = nltk.classify.accuracy(model, testing)*100
print("SVC Accuracy: {}".format(accuracy))



from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

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

for name, model in models:
    nltk_model = SklearnClassifier(model)
    nltk_model.train(training)
    accuracy = nltk.classify.accuracy(nltk_model, testing)*100
    print("{} Accuracy: {}".format(name, accuracy))
    
# Ensemble methods - Voting classifier
from sklearn.ensemble import VotingClassifier

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

models = list(zip(names, classifiers))

nltk_ensemble = SklearnClassifier(VotingClassifier(estimators = models, voting = 'hard', n_jobs = -1))
nltk_ensemble.train(training)
accuracy = nltk.classify.accuracy(nltk_model, testing)*100
print("Voting Classifier: Accuracy: {}".format(accuracy))

### make class label prediction for testing set
##txt_features, labels = list(zip(*testing))
##
##prediction = nltk_ensemble.classify_many(txt_features)
### print a confusion matrix and a classification report
##print(classification_report(labels, prediction))

classifier = nltk.NaiveBayesClassifier.train(training)
print(classifier.classify(find_features(preproccess_text('INGIN SELESAIKAN MASALAH SEPERTI 1.banyak hutang 2.Biayah rumah tangga 3.Buka usaha solusinya whatsappp; 082338028568 At klik di; www.bebasakses18.blogspot.com'))))
