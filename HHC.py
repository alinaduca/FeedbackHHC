import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def select_relevant_features(data):
    columns_to_drop = ['Provider Name', 'CMS Certification Number (CCN)', 'Address', 'ZIP Code', 'Telephone Number',
                       'Certification Date', 'DTC Performance Categorization', 'PPR Performance Categorization']
    with open("initial_column_names.txt", "r") as file:
        file_column_names = [line.strip() for line in file.readlines()]
    columns_to_drop.extend([name for name in file_column_names if name.startswith("Footnote")])
    data = data.drop(columns=columns_to_drop, errors='ignore')
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
    with open("initial_column_names.txt", 'w') as file:
       for name in column_names:
            file.write(name + '\n')
    '''
    data = pd.read_csv(file_path, names=column_names, skiprows=1)
    data.replace(',', '', regex=True, inplace=True)
    return data


def delete_rows_with_missing_categoric_value(data):
    categoric_columns = ['Type of Ownership', 'PPH Performance Categorization', 'State', 'City/Town']
    missing_values = data[categoric_columns].eq("-").any(axis=1)
    filtered_data = data[~missing_values]
    return filtered_data


def compute_median_value_for_column(data, column_name):
    numerical_values = pd.to_numeric(data[column_name], errors='coerce')
    numerical_values = numerical_values[~np.isnan(numerical_values)]
    if not np.isnan(numerical_values).all():
        median_value = np.median(numerical_values)
        return median_value
    else:
        return np.nan


def complete_data_with_median(data):
    for column_name in data.columns:
        if "-" in data[column_name].values:
            if column_name == 'Quality of patient care star rating':
                selected_rows = data[data['State'].str.lower() == data[column_name].str.lower()]
                median_value = compute_median_value_for_column(selected_rows, column_name)
            else:
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
    data = delete_rows_with_missing_categoric_value(data)
    data = complete_data_with_median(data)

    data_encoded = pd.get_dummies(data, columns=['Type of Ownership', 'PPH Performance Categorization']) #one-hot

    data_normalized = normalize_booleans(data_encoded)
    data_normalized = encode_columns_to_numeric(data_normalized, columns=['State', 'City/Town']) #dictionary

    data_normalized = normalize_data_custom(data_normalized)
    data_normalized.to_csv("output_file.csv", index=False)
    return data_normalized


def data_analysis(data):
    df = pd.DataFrame(data)
    selected_columns = df.columns[12:46]
    mean_values = df[selected_columns].mean()
    median_values = df[selected_columns].median()
    summary_df = pd.DataFrame({'Mean': mean_values, 'Median': median_values})
    summary_df.plot(kind='bar')
    plt.title('Mean and median values')
    plt.savefig('means_and_medians.png')


if __name__ == '__main__':
    preprocessed_data = preprocessing()

    '''
    final_column_names = preprocessed_data.columns.tolist()
    # Append the column names to the file
    with open("final_column_names.txt", "a") as file:
        for column_name in final_column_names:
            file.write(column_name + '\n')
            '''

    print(preprocessed_data)
    #data_analysis(preprocessed_data)
