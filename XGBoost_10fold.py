import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_xgboost(X_train, y_train):
    model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
    scores = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy')
    model.fit(X_train, y_train)
    return model, scores.mean()

def evaluate_xgboost(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return accuracy, precision, recall, f1
