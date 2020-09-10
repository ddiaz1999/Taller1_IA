import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import data as df

def get_Normalize(dataframe):
    x = dataframe.values  # returns a numpy array
    min_max_scaler = MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    x = pd.DataFrame(x_scaled)
    return x

if __name__ == "__main__":

    # Read data from csv file
    train_data_path = 'C:/Users/lenovo/Desktop/JHON_2030/IA/TALLER_1/dataset/'
    train_data_filename = 'training.csv'
    train_data_file = os.path.join(train_data_path, train_data_filename)

    data_train = df.dataframe(train_data_file) # cargamos el dataframe
    #data_train.get_Data_Info()
    #data_train.get_data_balance()
    data_train.One_Hot_Encoding()
    new_charactertistics = data_train.get_characteristics()
    X, Y_train = data_train.get_Split_Data()
    X_train = get_Normalize(X)
    #print(X['Workclass_ Private'])
    #print(X_train[new_charactertistics.index('Workclass_ Private')])
