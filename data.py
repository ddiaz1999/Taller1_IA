''' Ruler 1         2         3         4         5         6         7        '
/*******************************************************************************
*                                                                              *
*                       /$$              /$$                                   *
*                      | $$             | $$                                   *
*                  /$$$$$$$   /$$$$$$   /$$$$$$     /$$$$$$                    *
*                 /$$__  $$  |____  $$ |_  $$_/    |____  $$                   *
*                | $$  | $$   /$$$$$$$   | $$       /$$$$$$$                   *
*                | $$  | $$  /$$__  $$   | $$ /$$  /$$__  $$                   *
*                |  $$$$$$$ |  $$$$$$$   |  $$$$/ |  $$$$$$$                   *
*                 \_______/  \_______/    \___/    \_______/                   *
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

import pandas as pd
import numpy as np
import prettytable


#------------------------------------------------------------------------------#
#                                 DATAFRAME CLASS                              #
#------------------------------------------------------------------------------#

class dataframe():

    def __init__(self, file):
        self.__data = pd.read_csv(file, na_values=' ?')
        self.__original_lenght_data = self.__data.shape[0]

    def get_Data_Info(self):
        x = prettytable.PrettyTable(["Data amount", "characteristics"])
        x.add_row([self.__original_lenght_data, len(self.get_characteristics())])
        print(x, '\n')

    def get_data_balance(self):
        incomes_data = self.__data['income'].value_counts().tolist()
        percent_incomes_more_than_50k = incomes_data[1] * 100 / self.__original_lenght_data
        percent_incomes_less_than_50k = 100 - percent_incomes_more_than_50k
        data_balance = prettytable.PrettyTable(["Feature","<=50K", ">50K"])
        data_balance.add_row(['Amount', incomes_data[0], incomes_data[1]])
        data_balance.add_row(['Percentage', round(percent_incomes_less_than_50k, 2), round(percent_incomes_more_than_50k, 2)])
        print(' Data balance')
        print(data_balance, '\n')

    def get_characteristics(self):
        charateristics = list(self.__data.columns.tolist())
        charateristics.remove('income')
        return charateristics

    def delete_miss_data(self, verbose=False):
        self.__data = data = self.__data.dropna()
        features = prettytable.PrettyTable(["Description", "Amount"])
        if verbose:
            features.add_row(['Original lenght of data', self.__original_lenght_data])
            features.add_row(['After remove miss data lenght', data.shape[0]])
            features.add_row(['Total miss values', self.__original_lenght_data-data.shape[0]])
            print(' Delete miss data: features')
            print(features, '\n')

    def One_Hot_Encoding(self):
        self.__data['Sex'] = self.__data['Sex'].str.strip().replace(['Female', 'Male'], [0, 1])
        self.__data['income'] = self.__data['income'].str.strip().replace(['<=50K', '>50K'], [0, 1])

        categorical_cols = self.__data.select_dtypes(include=['object']).columns.tolist()
        self.__data[categorical_cols] = self.__data[categorical_cols].astype('category')
        self.__data = pd.get_dummies(data=self.__data, columns=categorical_cols)
        self.__data = self.__data.astype('int')

    def get_Split_Data(self):
        X, Y = self.__data.drop('income', axis=1), self.__data['income']
        return X, Y
