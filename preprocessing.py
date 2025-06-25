import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import warnings

warnings.filterwarnings('ignore')

all_columns = []


# drop non-relevant columns (Geodata etc)
def drop_geodata(data):
    drop_list = data.columns.tolist()[0:data.columns.tolist().index('UserLanguage') + 1]

    data.drop(inplace=True, axis='columns', columns=drop_list)
    return data

# remove non-consent data and exclusion criteria
def drop_exlusion(data):
    data.dropna(subset=['Consent', 'Q2', 'Q4', 'Q3'], inplace=True)
    data = data[data['Consent'].isin(['Yes.'])]
    data = data[data['Q2'].isin(['16+'])]
    data = data[data['Q4'].isin(['AI Bachelor'])]
    data = data[data['Q3'].isin(['Year 2', 'Year 3'])]

    # make necessary columns numeric
    useful = [c for c in all_columns if c[-2:] in ['_1', '_2', '_3']]
    data[useful] = data[useful].apply(pd.to_numeric)
    return data

# align data columns
def align_columns(data):
    columns_3 = [col for col in all_columns if col[0] == '3' and not col[0:3] == '3_2']
    for col in columns_3:
        new_col = '2_' + col[2:]
        if new_col in all_columns:
            # merge cols
            data[new_col] = np.where(data[new_col].notna(), data[new_col], data[col])

            data.drop([col], axis=1, inplace=True)

    return data

# remove no_take courses from dataframe

# 2nd year

# for every data row:
# get no take values
# make all values in no_take list nan
# later on fill every nan in with mean-correction

def remove_noTake(data, year):
    for index, row in data.iterrows():
        # print(index)
        # print(row)

        # print(row['2_FS_2'].dtype)
        # break

        no_take = row['2_NoTake']

        if pd.isna(no_take):
            continue

        no_take_list = no_take.split(',')
        abbreviations = []

        for no in no_take_list:
            mini = no.split()
            short_version = ''

            for m in mini:
                if m[0].isupper():
                    short_version += m[0]

            abbreviations.append(short_version)

        # print(abbreviations)
        start = '2_'

        for abb in abbreviations:
            row[start + abb + '_1'] = pd.NA
            row[start + abb + '_2'] = pd.NA
            row[start + abb + '_3'] = pd.NA

    # 3rd year
    if year == 3:
        for index, row in data.iterrows():
            # print(index)
            # print(row)
            if row['Q3'] == 'Year 2':
                continue

            no_take = row['3_2NoTake']

            if pd.isna(no_take):
                continue

            no_take_list = no_take.split(',')
            abbreviations = []

            for no in no_take_list:
                mini = no.split()
                short_version = ''

                for m in mini:
                    if m[0].isupper():
                        short_version += m[0]

                abbreviations.append(short_version)

            # print(abbreviations)
            start = '2_'

            for abb in abbreviations:
                row[start + abb + '_1'] = pd.NA
                row[start + abb + '_2'] = pd.NA
                row[start + abb + '_3'] = pd.NA

    return


def load_and_preprocess():
    global all_columns

    data = pd.read_csv('trial_data.csv')
    all_columns = data.columns.tolist()

    data = drop_geodata(data)
    data = drop_exlusion(data)
    data = align_columns(data)

    # drop acr/arw, ppd, rdsm courses
    data = data.drop(['3_ACR_1', '3_ACR_2', '3_ACR_3',
                      '3_ACR_1.1', '3_ACR_2.1', '3_ACR_3.1',
                      '2_ARW_1', '2_ARW_2', '2_ARW_3',
                      '2_PPD1_1', '2_PPD1_2', '2_PPD1_3',
                      '2_RDSM_1', '2_RDSM_2', '2_RDSM_3'], axis=1)

    data = data.drop(['Consent', 'Q2', 'Q4'], axis=1)

    # # drop rows that have less than 5 actual values
    # data = data.dropna(thresh=5)

    remove_noTake(data, 2)
    remove_noTake(data, 3)

    return data


# data = load_and_preprocess()

