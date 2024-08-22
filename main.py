# main.py

import os
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.svm import SVC
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, LSTM, TimeDistributed
from evaluate_model import evaluate_model

# Set dataset paths
dataset_path = 'path/to/your/dataset'
image_size = (128, 128)  # Adjust size based on your dataset

# Load dataset
datagen = ImageDataGenerator(rescale=1./255)
data_generator = datagen.flow_from_directory(dataset_path, target_size=image_size, batch_size=32, class_mode='categorical')

# Splitting data into training and validation sets (80:20 ratio)
X_train, X_val, y_train, y_val = train_test_split(data_generator[0][0], data_generator[0][1], test_size=0.2, random_state=42)

# Flattening for SVM and XGBoost
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)

# SVM Model
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train_flat, np.argmax(y_train, axis=1))

# XGBoost Model
xgb_model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
xgb_model.fit(X_train_flat, np.argmax(y_train, axis=1))

# RNN Model
model_rnn = Sequential()
model_rnn.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)))
model_rnn.add(MaxPooling2D(pool_size=(2, 2)))
model_rnn.add(Flatten())
model_rnn.add(Dense(64, activation='relu'))
model_rnn.add(Dropout(0.5))
model_rnn.add(Dense(len(data_generator.class_indices), activation='softmax'))
model_rnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_rnn.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32)

# LSTM Model
model_lstm = Sequential()
model_lstm.add(TimeDistributed(Conv2D(32, kernel_size=(3, 3), activation='relu'), input_shape=(None, image_size[0], image_size[1], 3)))
model_lstm.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
model_lstm.add(TimeDistributed(Flatten()))
model_lstm.add(LSTM(128, return_sequences=False))
model_lstm.add(Dense(len(data_generator.class_indices), activation='softmax'))
model_lstm.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_lstm.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32)

# Evaluating Models
# SVM Evaluation
y_pred_svm = svm_model.predict(X_val_flat)
evaluate_model(np.argmax(y_val, axis=1), y_pred_svm, "SVM")

# XGBoost Evaluation
y_pred_xgb = xgb_model.predict(X_val_flat)
evaluate_model(np.argmax(y_val, axis=1), y_pred_xgb, "XGBoost")

# RNN Evaluation
y_pred_rnn = np.argmax(model_rnn.predict(X_val), axis=-1)
evaluate_model(np.argmax(y_val, axis=1), y_pred_rnn, "RNN")

# LSTM Evaluation
y_pred_lstm = np.argmax(model_lstm.predict(X_val), axis=-1)
evaluate_model(np.argmax(y_val, axis=1), y_pred_lstm, "LSTM")

