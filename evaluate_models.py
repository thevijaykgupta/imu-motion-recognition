"""
Comprehensive Evaluation Suite for IMU Motion Recognition
Generates publication-quality figures and evaluation metrics
"""

import numpy as np
import os
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time

def main():
    # Setup output directories
    os.makedirs('./plots', exist_ok=True)
    os.makedirs('./plots/evaluation_csv', exist_ok=True)

    # Load data
    X_train = np.loadtxt('./data/UCI HAR Dataset/train/X_train.txt')
    X_test = np.loadtxt('./data/UCI HAR Dataset/test/X_test.txt')
    y_train = np.loadtxt('./data/UCI HAR Dataset/train/y_train.txt', dtype=int) - 1
    y_test = np.loadtxt('./data/UCI HAR Dataset/test/y_test.txt', dtype=int) - 1

    activity_labels = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']

    # Scale data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_test_norm = scaler.transform(X_test)
    X_test_dl = X_test_norm.reshape(X_test_norm.shape[0], X_test_norm.shape[1], 1)

    print("="*60)
    print("COMPREHENSIVE EVALUATION SUITE")
    print("="*60)

    # Load models
    import tensorflow as tf
    from tensorflow import keras

    models = {}
    results = {}

    # ML Models
    models['XGBoost'] = joblib.load('./models/xgb_model.pkl')
    models['SVM'] = joblib.load('./models/svm_model.pkl')
    models['Random Forest'] = joblib.load('./models/rf_model.pkl')

    # DL Models
    models['CNN-LSTM'] = keras.models.load_model('./models/cnn_lstm_model.h5', compile=False)
    models['LSTM'] = keras.models.load_model('./models/lstm_model.h5', compile=False)
    models['CNN'] = keras.models.load_model('./models/cnn_model.h5', compile=False)

    inference_times = {}
    predictions = {}

    for name, model in models.items():
        if name in ['CNN-LSTM', 'LSTM', 'CNN']:
            start = time.time()
            y_pred = np.argmax(model.predict(X_test_dl, verbose=0), axis=1)
            inference_times[name] = (time.time() - start) / len(y_test)
        else:
            start = time.time()
            y_pred = model.predict(X_test_norm)
            inference_times[name] = (time.time() - start) / len(y_test)

        predictions[name] = y_pred
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_weighted': precision_score(y_test, y_pred, average='weighted'),
            'recall_weighted': recall_score(y_test, y_pred, average='weighted'),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'mcc': matthews_corrcoef(y_test, y_pred),
            'kappa': cohen_kappa_score(y_test, y_pred),
            'balanced_acc': balanced_accuracy_score(y_test, y_pred),
            'inference_time': inference_times[name]
        }

    # Save CSV metrics
    metrics_df = pd.DataFrame(results).T
    metrics_df.to_csv('./plots/evaluation_csv/model_metrics.csv')
    print("[OK] Model metrics saved to CSV")

    # Print results
    for name, metrics in results.items():
        print(f"  {name}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_weighted']:.4f}")

if __name__ == "__main__":
    main()