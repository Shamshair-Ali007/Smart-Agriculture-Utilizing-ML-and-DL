# evaluate_model.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns

def evaluate_model(y_true, y_pred, model_name):
    """
    Evaluate the performance of a model by calculating accuracy, precision, recall, F1-score, and plotting a confusion matrix.

    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    model_name (str): Name of the model (for labeling plots and reports).
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"{model_name} Accuracy: {accuracy * 100:.2f}%")
    print(f"{model_name} Precision: {precision * 100:.2f}%")
    print(f"{model_name} Recall: {recall * 100:.2f}%")
    print(f"{model_name} F1-Score: {f1 * 100:.2f}%")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{model_name} Confusion Matrix")
    plt.show()
    
    # Classification Report
    print(f"\nClassification Report for {model_name}:\n")
    print(classification_report(y_true, y_pred, target_names=[str(i) for i in np.unique(y_true)]))
