import re
import nltk
import pickle
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder

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

# Open word features
word_features_f = open("sms_classifier_pickle/word_features.pickle", "rb")
word_features = pickle.load(word_features_f)
word_features_f.close()

# Define models to train
# names = ["K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression", "SGD Classifier", "Naive Bayes", "SVM Linear"]
names = ["K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression", "SGD Classifier", "Naive Bayes", "SVM Linear"]

normal_msg = 0
promo_msg = 0
spam_msg = 0

for name in names:
    classifier_s = open("sms_classifier_pickle/ratio(5.5;3.3;1)/" + name + ' Classifier.pickle', "rb")
    sms_classifier = pickle.load(classifier_s)
    classifier_s.close()
    
    df = pd.read_csv('training_script/corpus/sms_corpus/test_data.txt', engine='python', sep="<%>", header=None)
    classes = df[0]
    sms_data = preproccess_df(df[1])
    
    encoder = LabelEncoder()
    labels_test = encoder.fit_transform(classes)

    test_results = []
    for x in sms_data:
        result = sms_classifier.classify(find_features(preproccess_text(x)))
        test_results.append(result)

    test_string = find_features(preproccess_text('Seminar Spektakuler Develop Bisnis StarUp TourTravel Dgn Web Auto 14OKT18 13:00 HARRIS HOTEL TEBET Segera daftar seat terbatas Daftar: nama/NoHp WA 08112012929'))
    result = sms_classifier.classify(test_string)

    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(test_results, labels_test)
    
    if result == 0:
        normal_msg = normal_msg + 1
        result = "normal message"
    elif result == 1:
        promo_msg = promo_msg + 1
        result = "promotion message"
    elif result == 2:
        spam_msg = spam_msg + 1
        result = "spam message"

    print(name + " Classifier Accuracy = {}".format(accuracy*100))
    print(name + " Model thinks this is a {}".format(result))
    print("")
    
if normal_msg >= promo_msg and normal_msg >= spam_msg:
    best_result = "normal message"
    confidence = normal_msg / (normal_msg + promo_msg + spam_msg)
elif promo_msg >= normal_msg and promo_msg >= spam_msg:
    best_result = "promotion message"
    confidence = promo_msg / (normal_msg + promo_msg + spam_msg)
elif spam_msg >= normal_msg and spam_msg >= promo_msg:
    best_result = "spam message"
    confidence = spam_msg / (normal_msg + promo_msg + spam_msg)

print("Overall Algorithm Confidence = {}".format(confidence*100))
print("Overall results is a {}".format(best_result))