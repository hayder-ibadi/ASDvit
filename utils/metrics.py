import evaluate
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, roc_auc_score
import itertools
import matplotlib.pyplot as plt

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions = eval_pred.predictions
    label_ids = eval_pred.label_ids
    predicted_labels = predictions.argmax(axis=1)
    acc_score = accuracy.compute(predictions=predicted_labels, references=label_ids)['accuracy']
    return {
        "accuracy": acc_score
    }


def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues, figsize=(10, 8)):
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    fmt = '.0f'
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


def calculate_and_print_metrics(y_true, y_pred, labels_list):
    """Calculates and prints accuracy, F1 score, confusion matrix, and classification report."""
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

    if len(labels_list) <= 150:
        cm = confusion_matrix(y_true, y_pred)
        plot_confusion_matrix(cm, labels_list, figsize=(8, 6))

    print()
    print("Classification report:")
    print()
    print(classification_report(y_true, y_pred, target_names=labels_list, digits=4))

def calculate_other_metrics(y_true, y_pred, outputs):
    """Calculates and prints sensitivity, specificity, PPV, NPV, and AUC."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    auc = roc_auc_score(y_true, outputs.predictions[:, 1])

    print("Sensitivity (Recall):", sensitivity)
    print("Specificity:", specificity)
    print("Negative Predictive Value (NPV):", npv)
    print("Positive Predictive Value (PPV):", ppv)
    print("Area Under the ROC Curve (AUC):", auc)
