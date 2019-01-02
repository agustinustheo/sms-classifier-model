import re
from nltk.tokenize import word_tokenize
import pickle

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

#Open word features
word_features_f = open("sms_classifier_pickle/word_features.pickle", "rb")
word_features = pickle.load(word_features_f)
word_features_f.close()

# The find_features function will determine which of the 1500 word features are contained in the review
def find_features(message):
    words = word_tokenize(message)
    features = {}
    for word in word_features:
        features[word] = (word in words)

    return features

classifier_s = open("sms_classifier_pickle/sms_classifier.pickle", "rb")
sms_classifier = pickle.load(classifier_s)
classifier_s.close()

classified_text = sms_classifier.classify(find_features(preproccess_text('Dapatkan hematnya produk dibawah ini promo hanya berlaku s.d 2 Januari 2019')))
if classified_text == 0:
    classified_text = "Normal Message"
elif classified_text == 1:
    classified_text = "Promotion Message"
elif classified_text == 2:
    classified_text = "Spam Message"

print(classified_text)
