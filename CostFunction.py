from sklearn.metrics import confusion_matrix

def cost_sensitive_metric(y_true, y_pred):
    cost_of_fp = 5   # Cost of misclassifying low-risk as high-risk
    cost_of_fn = 50  # Cost of misclassifying high-risk as low-risk
    false_positives = ((y_pred == 1) & (y_true == 0)).sum()
    false_negatives = ((y_pred == 0) & (y_true == 1)).sum()
    total_cost = (false_positives * cost_of_fp) + (false_negatives * cost_of_fn)
    return total_cost