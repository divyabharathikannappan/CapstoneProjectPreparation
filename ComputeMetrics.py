import numpy as np
from sklearn.metrics import confusion_matrix

def calculate_recall(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TP = cm[1, 1]  
    FN = cm[1, 0]  
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    return recall

def calculate_accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TP = cm[1, 1]  
    TN = cm[0, 0] 
    accuracy = (TP + TN) / np.sum(cm) if np.sum(cm) > 0 else 0
    return accuracy

def calculate_precision(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TP = cm[1, 1]  
    FP = cm[0, 1] 
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    return precision

