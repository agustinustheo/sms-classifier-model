import os
import re
import sys
import nltk
import random
import pickle
import numpy as np
import pandas as pd
from nltk.tokenize import Pword_tokenize

# Indonesian SMS Preprocessing
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

classes_name = ['Normal', 'Promo', 'Spam']
corpus_name = ['data.txt', 'test_data.txt']

for idx in range(0, len(corpus_name)):
    df = pd.read_csv('corpus/sms_corpus/'+corpus_name[idx], engine='python', sep="<%>", names=['classes','messages'], header=None)
    df = df.sort_values(by='classes', ascending=True)

    classes = df['classes']
    sms_data = preproccess_df(df['messages'])

    encoder = LabelEncoder()
    Y = encoder.fit_transform(classes)

    # Now lets do it for all the messages
    messages_list = list(zip(sms_data, Y))
    
    print('\n'+corpus_name[idx])
    print("-----------------")
    prev_class = 999
    len_of_words = 0
    len_of_sentence = 0
    total_num_chars = 0
    for x in messages_list:
        message = x[0]
        message_class = x[1]
        if prev_class != message_class and prev_class != 999:
            print(classes_name[prev_class]+":")
            print("Avg. length(Char): "+str(total_num_chars/len_of_sentence))
            print("Avg. number of words: "+str(len_of_words))
            print("Avg. word length(Char): "+str(total_num_chars/len_of_words)+"\n")
            len_of_words = 0
            len_of_sentence = 0
            total_num_chars = 0

        message_words = word_tokenize(message)
        word_num_in_message = len(message_words)
        len_of_words += word_num_in_message
        len_of_sentence += 1
        for y in message_words:
            total_num_chars += len(y)
        prev_class = message_class
    
    # Print last class
    print(classes_name[prev_class]+":")
    print("Avg. length(Char): "+str(total_num_chars/len_of_sentence))
    print("Avg. number of words: "+str(len_of_words))
    print("Avg. word length(Char): "+str(total_num_chars/len_of_words)+"\n")