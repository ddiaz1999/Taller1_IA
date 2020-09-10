import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
import data as df

def get_Normalize(dataframe):
    x = dataframe.values  # returns a numpy array
    min_max_scaler = MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    x = pd.DataFrame(x_scaled)
    return x

def processing_data(file):
    data = df.dataframe(file)
    data.delete_miss_data()
    data.One_Hot_Encoding()
    features = data.get_characteristics()
    X, Y = data.get_Split_Data()
    X = get_Normalize(X)
    return X, Y, features

if __name__ == "__main__":

    # Read data from csv file
    data_path = 'C:/Users/lenovo/Desktop/JHON_2030/IA/TALLER_1/dataset/'

    train_data_filename = 'training.csv'
    train_data_file = os.path.join(data_path, train_data_filename)

    test_data_filename = 'test.csv'
    test_data_file = os.path.join(data_path, test_data_filename)

    X_train, Y_train, characteristics_train = processing_data(train_data_file)
    X_test, Y_test, characteristics_test = processing_data(test_data_file)

    features_difference = list(set(characteristics_train) - set(characteristics_test))
    X_train = X_train.drop(characteristics_train.index(features_difference[0]), axis=1)

    classifier = svm.SVC(kernel='linear', gamma='auto', C=2)
    classifier.fit(X_train, Y_train)
    Y_predict = classifier.predict(X_test)

    print(Y_predict)
