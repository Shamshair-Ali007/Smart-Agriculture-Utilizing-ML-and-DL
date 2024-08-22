import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_lstm(input_shape, num_classes):
    model = Sequential()
    model.add(LSTM(128, activation='tanh', return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(LSTM(128, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
    return model

def train_lstm(X_train, y_train, input_shape, num_classes):
    kfold = KFold(n_splits=10, shuffle=True)
    fold_scores = []
    for train_idx, val_idx in kfold.split(X_train):
        model = build_lstm(input_shape, num_classes)
        history = model.fit(X_train[train_idx], y_train[train_idx], validation_data=(X_train[val_idx], y_train[val_idx]), epochs=50, batch_size=32, verbose=0)
        fold_scores.append(history.history['val_accuracy'][-1])
    model.fit(X_train, y_train, epochs=50, batch_size=32)
    return model, np.mean(fold_scores)

def evaluate_lstm(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test)
    precision = precision_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average='weighted')
    recall = recall_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average='weighted')
    f1 = f1_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average='weighted')
    return accuracy, precision, recall, f1
