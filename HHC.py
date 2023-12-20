import numpy as np
import pandas as pd


def select_relevant_features(data):
    with open("column_names.txt", "r") as file:
        for name in file.readlines():
            if name.split()[0] == "Footnote":
                column_name_to_drop = name.strip()
                data = data.drop(column_name_to_drop, axis=1)
    return data


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
    with open("column_names.txt", 'w') as file:
       for name in column_names:
            file.write(name + '\n')
            '''
    data = pd.read_csv(file_path, names=column_names, skiprows=1)
    data.replace(',', '', regex=True, inplace=True)
    return data


def delete_rows_with_most_common_answer(data):
    dash_counts = (data == "-").sum(axis=1)
    rows_to_delete = dash_counts > (data.shape[1] / 2)
    filtered_data = data[~rows_to_delete]
    return filtered_data


def compute_median_value_for_column(data, column_name):
    numerical_values = pd.to_numeric(data[column_name], errors='coerce')
    numerical_values = numerical_values[~np.isnan(numerical_values) & (numerical_values != '-')]
    if not np.isnan(numerical_values).all():
        median_value = np.median(numerical_values)
        return median_value
    else:
        return np.nan


def complete_data_with_median(data):
    for column_name in data.columns:
        if "-" in data[column_name].values:
            median_value = compute_median_value_for_column(data, column_name)
            data[column_name] = data[column_name].replace("-", median_value)
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
    data = delete_rows_with_most_common_answer(data)
    data = complete_data_with_median(data)

    data_encoded = pd.get_dummies(data,
                                  columns=['Type of Ownership', 'DTC Performance Categorization',
                                           'PPR Performance Categorization', 'PPH Performance Categorization'])

    data_normalized = normalize_booleans(data_encoded)
    data_normalized = encode_columns_to_numeric(data_normalized,
                                                columns=['State', 'City/Town', 'Provider Name', 'Address', 'Certification Date'])
    data_normalized = normalize_data_custom(data_normalized)

    data_normalized.to_csv("output_file.csv", index=False)
    return data_normalized
    #print(data_normalized)


if __name__ == '__main__':
    prepocessed_data = preprocessing()
    print(prepocessed_data)
