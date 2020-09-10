import os
import pandas as pd
import numpy as np

if __name__ == "__main__":

    # Read data from csv file
    path = 'C:/Users/lenovo/Desktop/JHON_2030/IA/TALLER_1/dataset/'
    filename = 'training.csv'
    file = os.path.join(path, filename)
    train_data = pd.read_csv(file, na_values=' ?')

    # See total data
    original_lenght_data = train_data.shape[0]
    train_data = train_data.dropna()
    print('\nOriginal lenght of data:', original_lenght_data)
    print('After remove miss data lenght:', train_data.shape[0])
    print('Total miss values:', original_lenght_data-train_data.shape[0], '\n')

    # See data balance
    '''
    total_data = train_data.shape[0]
    data_balance = np.array(train_data['income'].value_counts())
    percent_incomes_more_than_50k = data_balance[1] * 100 / total_data
    percent_incomes_less_than_50k = 100 - percent_incomes_more_than_50k
    print('\nData balance')
    print('>50', round(percent_incomes_more_than_50k, 2))
    print('<=50', round(percent_incomes_less_than_50k, 2), '\n')
    '''

    # Characteristics
    characteristics = train_data.columns[:-1]

    # Cualitative data integer codification
    #train_data['Sex'].str.strip().replace(['Female', 'Male'], [0, 1], inplace=True)
    #train_data['Race'].str.strip().replace(['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'], [0, 1, 2, 3, 4], inplace=True)
    #train_data['Workclass'].str.strip().replace(['Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 'Self-emp-inc', 'Federal-gov', 'Without-pay'], [0, 1, 2, 3, 4, 5, 6], inplace=True)

    Integer_coding = {}
    for col_name in train_data.columns:
        if (train_data[col_name].dtype == 'object'):
            x = train_data[col_name].value_counts().keys().tolist()
            x.sort()
            Integer_coding[col_name] = x
            train_data[col_name] = train_data[col_name].astype('category')
            train_data[col_name] = train_data[col_name].cat.codes

    X_train, Y_train = train_data.drop(train_data.columns[-1], axis=1), train_data.drop(characteristics, axis=1)
    print(X_train.shape, Y_train.shape)
