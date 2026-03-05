from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import mlflow


def evaluate_model(model, X_test, y_test):

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    recall = recall_score(y_test, preds)
    precision = precision_score(y_test, preds)

    mlflow.log_metric("test_accuracy", acc)
    mlflow.log_metric("test_f1_score", f1)
    mlflow.log_metric("test_recall", recall)
    mlflow.log_metric("test_precision", precision)

    return acc, f1, recall, precision
