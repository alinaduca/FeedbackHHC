import csv
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, mean_squared_error, \
    ConfusionMatrixDisplay, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt


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


def calculate_accuracy(values, predictions):
    misclassif = 0
    for i in range(len(values)):
        if values[i] != predictions[i]:
            misclassif += 1
    return (len(values) - misclassif) / len(values)


def transform_categorical(value):
    if value * 5 < 2.5:
        return 0
    else:
        return 1


def custom_round_decimal(value):
    value *= 5
    integer_part = int(value)
    decimal_part = value - integer_part
    if decimal_part < 0.25:
        return integer_part * 1.0
    elif decimal_part < 0.75:
        return integer_part + 0.5
    else:
        return (integer_part + 1) * 1.0


def custom_round(value):
    value *= 5
    integer_part = int(value)
    decimal_part = value - integer_part
    if decimal_part < 0.5:
        return integer_part
    else:
        return integer_part + 1


def generate_ROC(y_test, y_probs_rf, y_probs_nn):
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_probs_rf)
    roc_auc_rf = auc(fpr_rf, tpr_rf)

    fpr_nn, tpr_nn, _ = roc_curve(y_test, y_probs_nn)
    roc_auc_nn = auc(fpr_nn, tpr_nn)

    # Afisarea ROC curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr_rf, tpr_rf, color='darkorange', lw=2, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
    plt.plot(fpr_nn, tpr_nn, color='green', lw=2, label=f'Neural Network (AUC = {roc_auc_nn:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig("ROC_curve.png")


if __name__ == "__main__":
    predicted_dataset = pd.read_csv("predictions_RandomForest_1_100.csv")
    predicted_dataset_nn = pd.read_csv("predictions_NeuralNetwork.csv")
    predicted_values_nn = predicted_dataset_nn['Predicted Values']
    true_values = predicted_dataset['Quality of patient care star rating']
    predicted_values = predicted_dataset['Predicted Values']
    generate_ROC(true_values.apply(transform_categorical), predicted_values.apply(transform_categorical), predicted_values_nn.apply(transform_categorical))
    predicted_values = predicted_values.apply(custom_round)
    true_values = true_values.apply(custom_round)
    predicted_values_nn = predicted_values_nn.apply(custom_round)
    cm = confusion_matrix(true_values, predicted_values)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    plt.savefig('confusion_matrix_RandomForest1_100.png')

    # Accuracy
    accuracy = calculate_accuracy(true_values, predicted_values)
    recall = recall_score(true_values, predicted_values, average='weighted')
    precision = precision_score(true_values, predicted_values, average='weighted')
    # accuracy_sklearn = accuracy_score(true_values, predicted_values)
    f1 = f1_score(true_values, predicted_values, average='weighted')
    # report = classification_report(true_values, predicted_values)
    write_metrics_to_csv("PerformanceMetrics_RandomForest_1_100.csv", ["Accuracy", "Precision", "Recall", "F1 Score"],
                         [accuracy, precision, recall, f1])
