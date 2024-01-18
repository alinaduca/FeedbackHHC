import csv


def write_to_csv(csvfile_path, feature_names, X_test, y_test, y_pred):
    with open(csvfile_path, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(feature_names + ['Predicted Values'])
        for i in range(len(y_pred)):
            row = [X_test.iloc[i][0], X_test.iloc[i][1], X_test.iloc[i][2], X_test.iloc[i][3], X_test.iloc[i][4], X_test.iloc[i][5], X_test.iloc[i][6], y_test.iloc[i], y_pred[i]]
            csv_writer.writerow(row)


def write_metrics_to_csv(csvfile_path, metrics_names, metrics_score):
    '''Performance metrics'''
    with open(csvfile_path, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(metrics_names)
        csv_writer.writerow(metrics_score)
