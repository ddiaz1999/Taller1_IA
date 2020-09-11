''' Ruler 1         2         3         4         5         6         7        '
/*******************************************************************************
*                                                                              *
*                                           /$$                                *
*                                          |__/                                *
*                 /$$$$$$/$$$$    /$$$$$$   /$$  /$$$$$$$                      *
*                | $$_  $$_  $$  |____  $$ | $$ | $$__  $$                     *
*                | $$ \ $$ \ $$   /$$$$$$$ | $$ | $$  \ $$                     *
*                | $$ | $$ | $$  /$$__  $$ | $$ | $$  | $$                     *
*                | $$ | $$ | $$ |  $$$$$$$ | $$ | $$  | $$                     *
*                |__/ |__/ |__/  \_______/ |__/ |__/  |__/                     *
*                                                                              *
*                  Developed by:                                               *
*                                                                              *
*                            Jhon Hader Fernandez                              *
*                     - jhon_fernandez@javeriana.edu.co                        *
*                                                                              *
*                             Diego Fernando Diaz                              *
*                        - di-diego@javeriana.edu.co                           *
*                                                                              *
*                          Oscar Geovanny Baracaldo                            *
*                       - obaracaldo@javeriana.edu.co                          *
*                                                                              *
*                       Pontificia Universidad Javeriana                       *
*                            Bogota DC - Colombia                              *
*                                  Sep - 2020                                  *
*                                                                              *
*****************************************************************************'''

#------------------------------------------------------------------------------#
#                          IMPORT MODULES AND LIBRARIES                        #
#------------------------------------------------------------------------------#

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import plot_confusion_matrix
from sklearn import svm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import data as df
import models

#------------------------------------------------------------------------------#
#                                   FUNCTIONS                                  #
#------------------------------------------------------------------------------#

def get_Normalize(dataframe):
    x = dataframe.values  # returns a numpy array
    min_max_scaler = MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    x = pd.DataFrame(x_scaled)
    return x

def processing_data(file):
    data = df.dataframe(file)
    data.delete_miss_data(verbose=False)
    data.One_Hot_Encoding()
    features = data.get_characteristics()
    X, Y = data.get_Split_Data()
    X = get_Normalize(X)
    return X, Y, features

#------------------------------------------------------------------------------#
#                                     MAIN                                     #
#------------------------------------------------------------------------------#

if __name__ == "__main__":

    #Ingrese aquí el path de donde guardó el dataset
    #data_path = 'C:/Users/lenovo/Desktop/JHON_2030/IA/dataset/'
    data_path = 'C:/Users/di-di/PycharmProjects/Taller1_IA/dataset/'

    train_data_filename = 'training.csv'
    train_data_file = os.path.join(data_path, train_data_filename)

    test_data_filename = 'test.csv'
    test_data_file = os.path.join(data_path, test_data_filename)

    X_train, Y_train, features_train = processing_data(train_data_file)
    X_test, Y_test, features_test = processing_data(test_data_file)

    features_difference = list(set(features_train) - set(features_test))
    X_train = X_train.drop(features_train.index(features_difference[0]), axis=1)

    my_classifier = models.model(X_train, Y_train, X_test, Y_test)

    ## Metodo de SVM
    my_classifier.SVM()
    Y_predict = my_classifier.get_predict_SVM()
    print('Mean accuracy SVM:', my_classifier.get_score_SVM())
    my_classifier.confussion_Matrix('SVM')

    ## Metodo de algoritmo de Perceptron
    my_classifier.Perceptron()
    Y_predict = my_classifier.get_predict_Perceptron()
    print('Mean accuracy Perceptron:', my_classifier.get_score_Perceptron())
    my_classifier.confussion_Matrix('Perceptron')

    ## Metodo de discriminante de Fischer
    my_classifier.Fischer()
    Y_predict = my_classifier.get_predict_Fischer()
    print('Mean accuracy Fischer:', my_classifier.get_score_Fischer())
    my_classifier.confussion_Matrix('Fischer')