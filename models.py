''' Ruler 1         2         3         4         5         6         7        '
/*******************************************************************************
*                                                                              *
*                                        /$$             /$$                   *
*                                       | $$            | $$                   *
*         /$$$$$$/$$$$    /$$$$$$    /$$$$$$$   /$$$$$$  | $$   /$$$$$$$       *
*        | $$_  $$_  $$  /$$__  $$  /$$__  $$  /$$__  $$ | $$  /$$_____/       *
*        | $$ \ $$ \ $$ | $$  \ $$ | $$  | $$ | $$$$$$$$ | $$ |  $$$$$$        *
*        | $$ | $$ | $$ | $$  | $$ | $$  | $$ | $$_____/ | $$  \____  $$       *
*        | $$ | $$ | $$ |  $$$$$$/ |  $$$$$$$ |  $$$$$$$ | $$  /$$$$$$$/       *
*        |__/ |__/ |__/  \______/   \_______/  \_______/ |__/ |_______/        *
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

from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn import linear_model

#------------------------------------------------------------------------------#
#                                 DATAFRAME CLASS                              #
#------------------------------------------------------------------------------#

class model():

    def __init__(self, X_train, Y_train, X_test, Y_test):
        self.__data_train = (X_train, Y_train)
        self.__data_test = (X_test, Y_test)

    def SVM(self):
        self.__classifier_SVM = svm.SVC(kernel='linear', gamma='auto')
        self.__classifier_SVM.fit(self.__data_train[0], self.__data_train[1])

    def get_predict_SVM(self):
        Y_predict = self.__classifier_SVM.predict(self.__data_test[0])
        return Y_predict

    def get_score_SVM(self):
        mean_accuracy = self.__classifier_SVM.score(self.__data_train[0], self.__data_train[1])
        return mean_accuracy

    def Perceptron(self):
        self.__classfier_Perceptron = linear_model.Perceptron(tol=1e-3, random_state=0)
        self.__classfier_Perceptron.fit(self.__data_train[0], self.__data_train[1])

    def get_predict_Perceptron(self):
        Y_predict = self.__classfier_Perceptron.predict(self.__data_test[0])
        return Y_predict

    def get_score_Perceptron(self):
        mean_accuracy = self.__classfier_Perceptron.score(self.__data_train[0], self.__data_train[1])
        return mean_accuracy

    def confussion_Matrix(self, model):
        target_names = np.array(['<=50k', '>50k'])
        target_names.reshape((len(target_names),))
        if model == 'SVM':
            dis = plot_confusion_matrix(self.__classfier_SVM, self.__data_test[0], self.__data_test[1], cmap=plt.cm.Blues, display_labels=target_names)
            plt.show()
        elif model == 'Perceptron':
            dis = plot_confusion_matrix(self.__classfier_Perceptron, self.__data_test[0], self.__data_test[1],
                                        cmap=plt.cm.Blues, display_labels=target_names)
            plt.show()
        else:
            print('Error 404 not found: Do not exist this model')