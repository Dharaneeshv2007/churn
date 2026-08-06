import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input, Reshape
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

def build_gru_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim, 1)),
        GRU(128, return_sequences=True),
        Dropout(0.3),
        GRU(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_gru_model(model, X, y):
    from sklearn.utils import class_weight
    # Ensure correct input shape
    X = X.reshape((X.shape[0], X.shape[1], 1))
    # Compute class weights
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y),
        y=y
    )
    class_weights = dict(enumerate(class_weights))
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(
        X, y,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        verbose=0,
        callbacks=[early_stop],
        class_weight=class_weights
    )
    return history

def evaluate_model(model, X, y):
    # Ensure correct input shape
    X = X.reshape((X.shape[0], X.shape[1], 1))
    # Use threshold 0.4 for better recall
    y_pred_prob = model.predict(X)
    y_pred = (y_pred_prob > 0.4).astype(int)
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    cm = confusion_matrix(y, y_pred).tolist()
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'confusion_matrix': cm}
