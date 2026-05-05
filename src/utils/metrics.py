from sklearn.metrics import classification_report, accuracy_score

def evaluate_classification(y_true, y_pred):

    accuracy = accuracy_score(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        zero_division=0
    )

    return accuracy, report