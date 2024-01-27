import csv
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, mean_squared_error, ConfusionMatrixDisplay, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt


def write_metrics_to_csv(csvfile_path, metrics_names, metrics_score):
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
    plt.savefig("ROC_categorical.png")


def visualize_misclassified_points(correct_points, misclassified_points, index_feature_1, index_feature_2, algorithm):
    plt.clf()
    if len(correct_points) == 0:
        print("No correctly classified points.")
    else:
        plt.scatter([feature[index_feature_1] for feature in correct_points], [feature[index_feature_2] for feature in correct_points],
                    c='black', marker='o', label='Correctly Classified')

    if len(misclassified_points) == 0:
        print("No misclassified points.")
    else:
        # misclassified_features = [point[0] for point in misclassified_points]
        plt.scatter([feature[index_feature_1] for feature in misclassified_points],
                    [feature[index_feature_2] for feature in misclassified_points], c='red', marker='x', label='Misclassified')
    plt.xlabel('How often patients got better at walking or moving around')
    plt.ylabel('How often patients got better at taking their drugs correctly by mouth')
    plt.title('Classified Points')
    plt.legend()
    filename = 'visualize_misclassified_points' + algorithm + '.png'
    plt.savefig(filename)


if __name__ == "__main__":
    predicted_dataset_rf = pd.read_csv("corr-based/predictions_NeuralNetwork_sklearn.csv")
    predicted_dataset_nn = pd.read_csv("corr-based/predictions_NeuralNetwork_sklearn.csv")
    predicted_values_nn = predicted_dataset_nn['Predicted Values']
    true_values = predicted_dataset_rf['Quality of patient care star rating']
    true_values = true_values.apply(custom_round)
    predicted_values_rf = predicted_dataset_rf['Predicted Values']
    mse_rf = mean_squared_error(true_values, predicted_values_rf)
    mse_nn = mean_squared_error(true_values, predicted_values_nn)

    generate_ROC(true_values.apply(transform_categorical), predicted_values_rf.apply(transform_categorical), predicted_values_nn.apply(transform_categorical))
    cm = confusion_matrix(true_values, predicted_values_rf)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    plt.savefig('confusion_matrix_RandomForest_sklearn.png')

    plt.clf()
    cmnn = confusion_matrix(true_values, predicted_values_nn)
    ConfusionMatrixDisplay(confusion_matrix=cmnn).plot()
    plt.savefig('confusion_matrix_NeuralNetwork_sklearn.png')

    # accuracy_sklearn = accuracy_score(true_values, predicted_values)
    # report = classification_report(true_values, predicted_values)
    accuracy = calculate_accuracy(true_values, predicted_values_rf)
    recall = recall_score(true_values, predicted_values_rf, average='weighted')
    precision = precision_score(true_values, predicted_values_rf, average='weighted')
    f1 = f1_score(true_values, predicted_values_rf, average='weighted')
    write_metrics_to_csv("corr-based/PerformanceMetrics_RandomForest_sklearn.csv", ["Accuracy", "Precision", "Recall", "F1 Score", "Mean Squared Error"],
                         [accuracy, precision, recall, f1, mse_rf])

    accuracy_nn = calculate_accuracy(true_values, predicted_values_nn)
    recall_nn = recall_score(true_values, predicted_values_nn, average='weighted')
    precision_nn = precision_score(true_values, predicted_values_nn, average='weighted')
    f1_nn = f1_score(true_values, predicted_values_nn, average='weighted')
    write_metrics_to_csv("corr-based/PerformanceMetrics_NeuralNetwork_sklearn.csv",
                         ["Accuracy", "Precision", "Recall", "F1 Score", "Mean Squared Error"],
                         [accuracy_nn, precision_nn, recall_nn, f1_nn, mse_nn])

    misclassified_instances_by_nn = []
    correct_instances_by_nn = []
    misclassified_instances_by_rf = []
    correct_instances_by_rf = []

    for i in range(len(predicted_dataset_rf.values)):
        instance_rf = predicted_dataset_rf.values[i]
        instance_nn = predicted_dataset_nn.values[i]
        instance_rf[-2] = instance_nn[-2] = custom_round(instance_rf[-2])
        predicted_dataset_rf.values[i] = instance_rf
        predicted_dataset_nn.values[i] = instance_nn
        if instance_rf[-2] != instance_rf[-1]:
            misclassified_instances_by_rf.append(instance_rf)
        else:
            correct_instances_by_rf.append(instance_rf)
        if instance_nn[-2] != instance_nn[-1]:
            misclassified_instances_by_nn.append(instance_nn)
        else:
            correct_instances_by_nn.append(instance_nn)

    with open("corr-based/Misclassified_RF.csv", mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(predicted_dataset_rf.columns.to_list())
        for i in range(len(misclassified_instances_by_rf)):
            csv_writer.writerow(misclassified_instances_by_rf[i])

    with open("corr-based/Misclassified_NN.csv", mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(predicted_dataset_nn.columns.to_list())
        for i in range(len(misclassified_instances_by_nn)):
            csv_writer.writerow(misclassified_instances_by_nn[i])
    visualize_misclassified_points(correct_instances_by_rf, misclassified_instances_by_rf, 0, 1, "RF")
    visualize_misclassified_points(correct_instances_by_nn, misclassified_instances_by_nn, 0, 1, "NN")
