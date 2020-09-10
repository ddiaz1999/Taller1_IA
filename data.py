import pandas as pd
import numpy as np
import prettytable

class dataframe():

    def __init__(self, file):
        self.__train_data = pd.read_csv(file, na_values=' ?')
        self.__original_lenght_data = self.__train_data.shape[0]

    def get_Data_Info(self):
        x = prettytable.PrettyTable(["Data amount", "characteristics"])
        x.add_row([self.__original_lenght_data, len(self.get_characteristics())])
        print(x, '\n')

    def get_data_balance(self):
        incomes_data = self.__train_data['income'].value_counts().tolist()
        percent_incomes_more_than_50k = incomes_data[1] * 100 / self.__original_lenght_data
        percent_incomes_less_than_50k = 100 - percent_incomes_more_than_50k
        data_balance = prettytable.PrettyTable(["Feature","<=50K", ">50K"])
        data_balance.add_row(['Amount', incomes_data[0], incomes_data[1]])
        data_balance.add_row(['Percentage', round(percent_incomes_less_than_50k, 2), round(percent_incomes_more_than_50k, 2)])
        print(' Data balance')
        print(data_balance, '\n')

    def get_characteristics(self):
        charateristics = list(self.__train_data.columns.tolist())
        charateristics.remove('income')
        return charateristics

    def delete_miss_data(self):
        self.__train_data = train_data = self.__train_data.dropna()
        features = prettytable.PrettyTable(["Description", "Amount"])
        features.add_row(['Original lenght of data', self.__original_lenght_data])
        features.add_row(['After remove miss data lenght', train_data.shape[0]])
        features.add_row(['Total miss values', self.__original_lenght_data-train_data.shape[0]])
        print(' Delete miss data: features')
        print(features, '\n')

    def One_Hot_Encoding(self):
        self.__train_data['Sex'] = self.__train_data['Sex'].str.strip().replace(['Female', 'Male'], [0, 1])
        self.__train_data['income'] = self.__train_data['income'].str.strip().replace(['<=50K', '>50K'], [0, 1])

        categorical_cols = self.__train_data.select_dtypes(include=['object']).columns.tolist()
        self.__train_data[categorical_cols] = self.__train_data[categorical_cols].astype('category')
        self.__train_data = pd.get_dummies(data=self.__train_data, columns=categorical_cols)
        self.__train_data = self.__train_data.astype('int')

    def get_Split_Data(self):
        X_train = self.__train_data.drop('income', axis=1)
        Y_train = self.__train_data.drop(self.get_characteristics(), axis=1)
        return X_train, Y_train