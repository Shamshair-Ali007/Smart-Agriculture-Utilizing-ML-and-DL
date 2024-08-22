from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_svm(X_train, y_train):
    model = svm.SVC(kernel='rbf', C=1.0, gamma='scale')
    model.fit(X_train, y_train)
    return model

def evaluate_svm(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return accuracy, precision, recall, f1
