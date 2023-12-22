import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr
from math import sqrt
from scipy.stats import pointbiserialr
from math import sqrt
from sklearn.model_selection import cross_val_score
from sklearn import svm
import time
import copy

states_rating = {}

    
def getMerit(subset, label, data):
    k = len(subset)

    # average feature-class correlation
    rcf_all = []
    for feature in subset:
        coeff = pointbiserialr( data[label], data[feature] )
        rcf_all.append( abs( coeff.correlation ) )
    rcf = np.mean( rcf_all )

    # average feature-feature correlation
    corr = data[data[subset].duplicated(keep=False)].corr()
    corr.values[np.tril_indices_from(corr.values)] = np.nan
    corr = abs(corr)
    rff = corr.unstack().mean()

    return (k * rcf) / sqrt(k + k * (k-1) * rff)

def select_relevant_features(data):
    columns_to_drop = ['Provider Name', 'CMS Certification Number (CCN)', 'Address', 'ZIP Code', 'Telephone Number',
                       'Certification Date', 'DTC Performance Categorization', 'PPR Performance Categorization']
    with open("initial_column_names.txt", "r") as file:
        file_column_names = [line.strip() for line in file.readlines()]
    columns_to_drop.extend([name for name in file_column_names if name.startswith("Footnote")])
    data = data.drop(columns=columns_to_drop, errors='ignore')
    return data


def custom_round(value):
    integer_part = int(value)
    decimal_part = value - integer_part
    if decimal_part < 0.25:
        return integer_part
    elif decimal_part < 0.75:
        return integer_part + 0.5
    else:
        return integer_part + 1


def calculate_rating_per_states(data):
    states = data['State'].values
    for state in set(states):
        selected_rows = data[data['State'] == state]
        numerical_values = pd.to_numeric(selected_rows['Quality of patient care star rating'], errors='coerce')
        numerical_values = numerical_values[~np.isnan(numerical_values)]
        mean_value = np.mean(numerical_values)
        states_rating[state] = mean_value


def encode_columns_to_numeric(data, columns):
    encoded_data = data.copy()
    for column in columns:
        unique_values = encoded_data[column].unique()
        label_mapping = {value: index for index, value in enumerate(unique_values)}
        encoded_data[column] = encoded_data[column].apply(lambda x: label_mapping[x])
    return encoded_data


def normalize_booleans(data):
    data.replace({True: 1, False: 0, "Yes": 1, "No": 0}, inplace=True)
    return data


def min_max_normalize(data, columns_to_normalize):
    for column in columns_to_normalize:
        min_value = data[column].min()
        max_value = data[column].max()
        data[column] = (data[column] - min_value) / (max_value - min_value)
    return data


def read_csv():
    file_path = 'HH_Provider_Oct2023.csv'
    with open(file_path) as file:
        first_line = file.readline().strip()
        column_names = first_line.split(',')
    column_names = [name.replace("'", '').replace('`', '').replace('"', '') for name in column_names]
    '''
    with open("initial_column_names.txt", 'w') as file:
       for name in column_names:
            file.write(name + '\n')
    '''
    data = pd.read_csv(file_path, names=column_names, skiprows=1)
    data.replace(',', '', regex=True, inplace=True)
    return data


def delete_rows_with_missing_categoric_value(data):
    categorical_columns = ['Type of Ownership', 'PPH Performance Categorization', 'State', 'City/Town',
                           'Offers Nursing Care Services', 'Offers Physical Therapy Services',
                           'Offers Occupational Therapy Services', 'Offers Speech Pathology Services',
                           'Offers Medical Social Services', 'Offers Home Health Aide Services',
                           'Quality of patient care star rating']
    missing_values = data[categorical_columns].eq("-").any(axis=1)
    filtered_data = data[~missing_values]
    return filtered_data


def compute_mean_value_for_column(data, column_name):
    numerical_values = pd.to_numeric(data[column_name], errors='coerce')
    numerical_values = numerical_values[~np.isnan(numerical_values)]
    if not np.isnan(numerical_values).all():
        mean_value = np.mean(numerical_values)
        return mean_value
    else:
        return np.nan


def replace_missing_rating(row):
    if row['Quality of patient care star rating'] == '-':
        return states_rating.get(row['State'])
    else:
        return row['Quality of patient care star rating']


def complete_data_with_mean(data):
    for column_name in data.columns:
        if "-" in data[column_name].values:
            if column_name == 'Quality of patient care star rating':
                indexes = data[data[column_name] == "-"].index.values
                for index in indexes:
                    data.at[index, 'Quality of patient care star rating'] = custom_round(states_rating[data['State'][index]])
                # selected_rows = data[data['State'].str.lower() == data[column_name].str.lower()]
                # mean_value = compute_mean_value_for_column(selected_rows, column_name)
            else:
                mean_value = compute_mean_value_for_column(data, column_name)
                data[column_name] = data[column_name].replace("-", mean_value)
                data[column_name] = data[column_name].fillna(mean_value)
    return data


def normalize_data_custom(data):
    normalized_data = data.copy()
    normalized_data = normalized_data.apply(pd.to_numeric, errors='ignore')
    for column in normalized_data.select_dtypes(include=['number']).columns:
        min_val = normalized_data[column].min()
        max_val = normalized_data[column].max()
        if min_val != max_val:
            normalized_data[column] = (normalized_data[column] - min_val) / (max_val - min_val)
    return normalized_data


def preprocessing():
    data = read_csv()
    data = select_relevant_features(data)
    data = delete_rows_with_missing_categoric_value(data)
    calculate_rating_per_states(data)
    data = complete_data_with_mean(data)

    data_encoded = pd.get_dummies(data, columns=['Type of Ownership', 'PPH Performance Categorization'])  # one-hot
    # data_encoded.to_csv("data_before_normalization.csv", index=False)
    data_normalized = normalize_booleans(data_encoded)
    data_normalized = encode_columns_to_numeric(data_normalized, columns=['State', 'City/Town'])  # dictionary
    data_normalized = normalize_data_custom(data_normalized)
    data_normalized.to_csv("output_file.csv", index=False)
    return data_normalized


def data_analysis(data):
    df = pd.DataFrame(data)
    selected_columns = df.columns[8:42]
    mean_values = df[selected_columns].mean()
    median_values = df[selected_columns].median()
    summary_df = pd.DataFrame({'Mean': mean_values, 'Median': median_values})
    summary_df.plot(kind='bar')
    plt.title('Mean and median values')
    plt.savefig('means_and_medians.png')
    plt.clf()
    categories = list(states_rating.keys())
    values = list(states_rating.values())
    plt.bar(categories, values, color='blue')
    plt.xlabel('Categories')
    plt.ylabel('Values')
    plt.title('Histogram')
    plt.savefig('ratings.png')


if __name__ == '__main__':
    preprocessed_data = preprocessing()

    '''
    final_column_names = preprocessed_data.columns.tolist()
    # Append the column names to the file
    with open("final_column_names.txt", "a") as file:
        for column_name in final_column_names:
            file.write(column_name + '\n')
    '''

    data_analysis(preprocessed_data)
