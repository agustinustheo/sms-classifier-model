import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

# Read corpus data
df = pd.read_csv('corpus/sms_corpus/data.txt', engine='python', sep="<%>", header=None)

classes = df[0]
sms_data = df[1]

encoder = LabelEncoder()
Y = encoder.fit_transform(classes)

skf = StratifiedKFold(n_splits=10)

def count_data_per_class(dataset_type, data):
    print(dataset_type, "dataset:", len(data))

    first_data = 0
    second_data = 0
    third_data = 0
    for index in data:
        if Y[index] == 0:
            first_data += 1
        elif Y[index] == 1:
            second_data += 1
        elif Y[index] == 2:
            third_data += 1
    print("First class count:", first_data)
    print("Second class count:", second_data)
    print("Third class count:", third_data)
    print("===============================")

if __name__ == "__main__":
    for train_index, test_index in skf.split(sms_data, Y):
        count_data_per_class("Train", train_index)
        count_data_per_class("Test", test_index)