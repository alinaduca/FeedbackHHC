import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression


def custom_round(value):
    value *= 5
    integer_part = int(value)
    decimal_part = value - integer_part
    if decimal_part < 0.5:
        return integer_part
    else:
        return integer_part + 1


data_rf = pd.read_csv("corr-based/predictions_RandomForest_sklearn.csv")
predicted_rf = data_rf["Predicted Values"]
true_values = data_rf["Quality of patient care star rating"]
data_nn = pd.read_csv("corr-based/predictions_NeuralNetwork_sklearn.csv")
predicted_nn = data_nn["Quality of patient care star rating"]

true_values = true_values.apply(custom_round)
predicted_rf = predicted_rf.apply(custom_round)
predicted_nn = predicted_nn.apply(custom_round)

y_true_onehot = label_binarize(true_values, classes=[1, 2, 3, 4, 5])
classifier_rf = OneVsRestClassifier(LogisticRegression())
y_score_rf = classifier_rf.fit(predicted_rf.values.reshape(-1, 1), y_true_onehot).predict_proba(
    predicted_rf.values.reshape(-1, 1))

classifier_nn = OneVsRestClassifier(LogisticRegression())
y_score_nn = classifier_nn.fit(predicted_nn.values.reshape(-1, 1), y_true_onehot).predict_proba(
    predicted_nn.values.reshape(-1, 1))

fpr_rf = dict()
tpr_rf = dict()
roc_auc_rf = dict()
fpr_nn = dict()
tpr_nn = dict()
roc_auc_nn = dict()

for i in range(y_true_onehot.shape[1]):
    fpr_rf[i], tpr_rf[i], _ = roc_curve(y_true_onehot[:, i], y_score_rf[:, i])
    roc_auc_rf[i] = auc(fpr_rf[i], tpr_rf[i])
    fpr_nn[i], tpr_nn[i], _ = roc_curve(y_true_onehot[:, i], y_score_nn[:, i])
    roc_auc_nn[i] = auc(fpr_nn[i], tpr_nn[i])

plt.figure(figsize=(10, 8))
colors = ['blue', 'orange', 'green', 'red', 'purple']

for i, color in zip(range(y_true_onehot.shape[1]), colors):
    plt.plot(fpr_rf[i], tpr_rf[i], color=color, lw=2, linestyle='--',
             label=f'Random Forest Class {i + 1} (AUC = {roc_auc_rf[i]:.2f})')
    plt.plot(fpr_nn[i], tpr_nn[i], color=color, lw=2, linestyle='-',
             label=f'Neural Network Class {i + 1} (AUC = {roc_auc_nn[i]:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label="Random")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for each class (RF vs NN)')
plt.legend(loc="lower right")
plt.savefig("corr-based/ROC_per_label.png")
