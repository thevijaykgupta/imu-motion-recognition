"""
Human Motion Recognition using IMU Data
End-to-End Pipeline: Data Loading → Preprocessing → ML & DL Models → Evaluation
UCI HAR Dataset (6 activities: Walking, Upstairs, Downstairs, Sitting, Standing, Laying)
"""

import numpy as np
import pandas as pd
import os
import urllib.request
import zipfile
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

# ============================================================================
# 1. DATA LOADING & PREPROCESSING
# ============================================================================

class IMUDataLoader:
    """Load and preprocess UCI HAR Dataset"""
    
    def __init__(self, dataset_path='./data/UCI HAR Dataset'):
        self.dataset_path = dataset_path
        self.activity_labels = {
            1: 'WALKING', 2: 'WALKING_UPSTAIRS', 3: 'WALKING_DOWNSTAIRS',
            4: 'SITTING', 5: 'STANDING', 6: 'LAYING'
        }
        
    def download_dataset(self):
        """Download UCI HAR Dataset"""
        if os.path.exists(self.dataset_path):
            print(f"Dataset already exists at {self.dataset_path}")
            return
        
        print("Downloading UCI HAR Dataset...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
        zip_path = "UCI_HAR_Dataset.zip"
        
        try:
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall()
            os.remove(zip_path)
            print("Dataset downloaded successfully!")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Please download manually from: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones")
    
    def load_data(self):
        """Load train and test data"""
        X_train = np.loadtxt(f'{self.dataset_path}/train/X_train.txt')
        y_train = np.loadtxt(f'{self.dataset_path}/train/y_train.txt', dtype=int)
        
        X_test = np.loadtxt(f'{self.dataset_path}/test/X_test.txt')
        y_test = np.loadtxt(f'{self.dataset_path}/test/y_test.txt', dtype=int)
        
        # Convert labels to 0-indexed
        y_train = y_train - 1
        y_test = y_test - 1
        
        return X_train, y_train, X_test, y_test
    
    def load_raw_signals(self):
        """Load raw accelerometer and gyroscope signals"""
        train_signals = {}
        test_signals = {}
        
        signal_types = ['total_acc_x', 'total_acc_y', 'total_acc_z', 
                       'body_acc_x', 'body_acc_y', 'body_acc_z',
                       'body_gyro_x', 'body_gyro_y', 'body_gyro_z']
        
        for signal in signal_types:
            train_signals[signal] = np.loadtxt(f'{self.dataset_path}/train/Inertial Signals/{signal}_train.txt')
            test_signals[signal] = np.loadtxt(f'{self.dataset_path}/test/Inertial Signals/{signal}_test.txt')
        
        return train_signals, test_signals


class SignalPreprocessor:
    """Preprocess IMU signals with filtering and normalization"""
    
    def __init__(self, window_size=128, overlap=0.5):
        self.window_size = window_size
        self.overlap = overlap
        self.scaler = StandardScaler()
        
    def sliding_window(self, signal, window_size, overlap):
        """Create overlapping windows from signal"""
        step = int(window_size * (1 - overlap))
        windows = []
        for i in range(0, len(signal) - window_size + 1, step):
            windows.append(signal[i:i + window_size])
        return np.array(windows)
    
    def normalize_signal(self, X_train, X_test):
        """Normalize features using StandardScaler"""
        X_train_norm = self.scaler.fit_transform(X_train)
        X_test_norm = self.scaler.transform(X_test)
        return X_train_norm, X_test_norm
    
    def butter_lowpass_filter(self, data, cutoff=5, fs=50, order=4):
        """Apply Butterworth low-pass filter"""
        from scipy.signal import butter, filtfilt
        nyquist = fs / 2
        normalized_cutoff = cutoff / nyquist
        b, a = butter(order, normalized_cutoff, btype='low')
        return filtfilt(b, a, data, axis=0)
    
    def preprocess_raw_signals(self, train_signals, test_signals, filter_signals=True):
        """Convert raw signals to preprocessed format"""
        all_signals = {}
        
        for key in train_signals.keys():
            if filter_signals:
                train_signals[key] = self.butter_lowpass_filter(train_signals[key])
                test_signals[key] = self.butter_lowpass_filter(test_signals[key])
            
            all_signals[f'{key}_train'] = train_signals[key]
            all_signals[f'{key}_test'] = test_signals[key]
        
        return all_signals


# ============================================================================
# 2. FEATURE EXTRACTION
# ============================================================================

class FeatureExtractor:
    """Extract statistical and time-domain features from IMU data"""
    
    @staticmethod
    def extract_features(signal_window):
        """Extract 13 features from signal window"""
        features = []
        
        # Time-domain features
        features.append(np.mean(signal_window))           # Mean
        features.append(np.std(signal_window))            # Std Dev
        features.append(np.max(signal_window))            # Max
        features.append(np.min(signal_window))            # Min
        features.append(np.median(signal_window))         # Median
        features.append(np.ptp(signal_window))            # Peak-to-peak
        
        # Energy and entropy
        features.append(np.sum(signal_window ** 2))       # Energy
        features.append(np.std(np.diff(signal_window)))   # Jerk (derivative std)
        
        # Quartiles
        q1, q3 = np.percentile(signal_window, [25, 75])
        features.append(q3 - q1)                          # Interquartile range
        
        # Zero crossing rate
        zero_crossings = np.sum(np.abs(np.diff(np.sign(signal_window)))) / 2
        features.append(zero_crossings)
        
        # Skewness and Kurtosis
        from scipy.stats import skew, kurtosis
        features.append(skew(signal_window))
        features.append(kurtosis(signal_window))
        
        # RMS
        features.append(np.sqrt(np.mean(signal_window ** 2)))
        
        return np.array(features)
    
    @staticmethod
    def extract_from_dataset(X_data, window_size=128, overlap=0.5):
        """Extract features from entire dataset"""
        step = int(window_size * (1 - overlap))
        all_features = []
        
        for sample_idx in range(X_data.shape[0]):
            sample_features = []
            sample = X_data[sample_idx].reshape(-1, 1)
            
            for i in range(0, len(sample) - window_size + 1, step):
                window = sample[i:i + window_size].flatten()
                features = FeatureExtractor.extract_features(window)
                sample_features.append(features)
            
            if sample_features:
                all_features.append(np.mean(sample_features, axis=0))
        
        return np.array(all_features)


# ============================================================================
# 3. MACHINE LEARNING MODELS
# ============================================================================

class MLModels:
    """Traditional ML models for activity recognition"""
    
    def __init__(self):
        self.models = {}
    
    def build_svm(self, kernel='rbf', C=100):
        """Build SVM classifier"""
        return SVC(kernel=kernel, C=C, probability=True, random_state=42)
    
    def build_xgb(self, n_estimators=200, max_depth=7, learning_rate=0.1):
        """Build XGBoost classifier"""
        return xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss'
        )
    
    def build_rf(self, n_estimators=200, max_depth=20):
        """Build Random Forest classifier"""
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
    
    def train_models(self, X_train, y_train):
        """Train all ML models"""
        print("\n" + "="*60)
        print("TRAINING MACHINE LEARNING MODELS")
        print("="*60)
        
        # SVM
        print("\nTraining SVM...")
        self.models['svm'] = self.build_svm()
        self.models['svm'].fit(X_train, y_train)
        
        # XGBoost
        print("Training XGBoost...")
        self.models['xgb'] = self.build_xgb()
        self.models['xgb'].fit(X_train, y_train)
        
        # Random Forest
        print("Training Random Forest...")
        self.models['rf'] = self.build_rf()
        self.models['rf'].fit(X_train, y_train)
        
        print("ML models trained successfully!")
    
    def predict(self, X_test, model_name='xgb'):
        """Make predictions with specific model"""
        return self.models[model_name].predict(X_test)
    
    def get_probabilities(self, X_test, model_name='xgb'):
        """Get prediction probabilities"""
        return self.models[model_name].predict_proba(X_test)


# ============================================================================
# 4. DEEP LEARNING MODELS
# ============================================================================

class DLModels:
    """Deep Learning models for activity recognition"""
    
    @staticmethod
    def build_cnn_lstm(input_shape, num_classes, sequence_length=128):
        """Build CNN-LSTM hybrid model for temporal-spatial feature extraction"""
        model = models.Sequential([
            # Reshape input for CNN
            layers.Reshape((sequence_length, input_shape[-1], 1), input_shape=input_shape),
            
            # CNN branch - Extract spatial features
            layers.Conv2D(64, kernel_size=(3, 1), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 1)),
            
            layers.Conv2D(128, kernel_size=(3, 1), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 1)),
            
            # Reshape for LSTM
            layers.Reshape((sequence_length // 4, 128)),
            
            # LSTM branch - Extract temporal dependencies
            layers.LSTM(256, activation='relu', return_sequences=True, dropout=0.3),
            layers.LSTM(128, activation='relu', dropout=0.3),
            
            # Dense layers
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model
    
    @staticmethod
    def build_lstm(input_shape, num_classes):
        """Build pure LSTM model"""
        model = models.Sequential([
            layers.LSTM(256, activation='relu', return_sequences=True, 
                       input_shape=input_shape, dropout=0.3),
            layers.LSTM(128, activation='relu', dropout=0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model
    
    @staticmethod
    def build_cnn(input_shape, num_classes):
        """Build 1D CNN model"""
        model = models.Sequential([
            layers.Conv1D(64, kernel_size=3, activation='relu', 
                         padding='same', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),
            
            layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),
            
            layers.Conv1D(256, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),
            
            layers.GlobalAveragePooling1D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model
    
    @staticmethod
    def build_attention_lstm(input_shape, num_classes):
        """Build LSTM with Attention mechanism"""
        inputs = layers.Input(shape=input_shape)
        
        # LSTM layers
        lstm_out = layers.LSTM(256, activation='relu', return_sequences=True, 
                               dropout=0.3)(inputs)
        lstm_out = layers.LSTM(128, activation='relu', return_sequences=True, 
                               dropout=0.3)(lstm_out)
        
        # Attention mechanism
        attention = layers.Dense(1, activation='tanh')(lstm_out)
        attention = layers.Flatten()(attention)
        attention = layers.Activation('softmax')(attention)
        attention = layers.RepeatVector(lstm_out.shape[-1])(attention)
        attention = layers.Permute((2, 1))(attention)
        
        context = layers.Multiply()([lstm_out, attention])
        context = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(context)
        
        # Dense layers
        output = layers.Dense(256, activation='relu')(context)
        output = layers.Dropout(0.4)(output)
        output = layers.Dense(128, activation='relu')(output)
        output = layers.Dropout(0.3)(output)
        output = layers.Dense(num_classes, activation='softmax')(output)
        
        model = models.Model(inputs=inputs, outputs=output)
        return model


# ============================================================================
# 5. TRAINING & EVALUATION
# ============================================================================

class ModelTrainer:
    """Train and evaluate models"""
    
    def __init__(self, num_classes=6):
        self.num_classes = num_classes
        self.history = {}
    
    def train_dl_model(self, model, X_train, y_train, X_val, y_val, 
                       epochs=100, batch_size=32, model_name='cnn_lstm'):
        """Train deep learning model"""
        print(f"\n{'='*60}")
        print(f"TRAINING {model_name.upper()} MODEL")
        print(f"{'='*60}")
        
        # Callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        
        # Compile
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        self.history[model_name] = history
        print(f"{model_name} training completed!")
        return model, history
    
    def evaluate_model(self, model, X_test, y_test, model_name='model'):
        """Evaluate model performance"""
        y_pred = np.argmax(model.predict(X_test), axis=1)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"\n{model_name} Evaluation:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score (weighted): {f1:.4f}")
        
        return accuracy, f1, y_pred
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, labels, title='Confusion Matrix'):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        return plt
    
    @staticmethod
    def print_classification_report(y_true, y_pred, labels, title='Classification Report'):
        """Print detailed classification report"""
        print(f"\n{title}")
        print(classification_report(y_true, y_pred, target_names=labels))


# ============================================================================
# 6. MAIN PIPELINE
# ============================================================================

def main():
    """Complete pipeline: Data Loading → Training → Evaluation"""
    
    # Activity labels
    activity_labels = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 
                      'SITTING', 'STANDING', 'LAYING']
    
    # ========== DATA LOADING ==========
    print("\n" + "="*60)
    print("LOADING UCI HAR DATASET")
    print("="*60)
    
    data_loader = IMUDataLoader()
    data_loader.download_dataset()
    X_train, y_train, X_test, y_test = data_loader.load_data()
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Number of activities: {len(activity_labels)}")
    
    # ========== PREPROCESSING ==========
    print("\n" + "="*60)
    print("PREPROCESSING DATA")
    print("="*60)
    
    preprocessor = SignalPreprocessor()
    X_train_norm, X_test_norm = preprocessor.normalize_signal(X_train, X_test)
    print("Data normalized using StandardScaler")
    
    # Train-val split for DL models
    X_train_dl, X_val_dl, y_train_dl, y_val_dl = train_test_split(
        X_train_norm, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Reshape for DL models (samples, timesteps, features)
    X_train_dl = X_train_dl.reshape(X_train_dl.shape[0], X_train_dl.shape[1], 1)
    X_val_dl = X_val_dl.reshape(X_val_dl.shape[0], X_val_dl.shape[1], 1)
    X_test_dl = X_test_norm.reshape(X_test_norm.shape[0], X_test_norm.shape[1], 1)
    
    print(f"DL Training set shape: {X_train_dl.shape}")
    print(f"DL Validation set shape: {X_val_dl.shape}")
    print(f"DL Test set shape: {X_test_dl.shape}")
    
    # ========== MACHINE LEARNING MODELS ==========
    ml_trainer = MLModels()
    ml_trainer.train_models(X_train_norm, y_train)
    
    print("\nML Models - Test Set Performance:")
    ml_results = {}
    for model_name in ['svm', 'xgb', 'rf']:
        y_pred = ml_trainer.predict(X_test_norm, model_name)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        ml_results[model_name] = {'accuracy': acc, 'f1': f1, 'predictions': y_pred}
        print(f"  {model_name.upper()}: Accuracy={acc:.4f}, F1={f1:.4f}")
    
    # ========== DEEP LEARNING MODELS ==========
    print("\n" + "="*60)
    print("TRAINING DEEP LEARNING MODELS")
    print("="*60)
    
    dl_trainer = ModelTrainer(num_classes=6)
    
    # CNN-LSTM (Hybrid Model)
    print("\n[1/3] CNN-LSTM Hybrid Model")
    model_cnn_lstm = DLModels.build_cnn_lstm(
        input_shape=(X_train_dl.shape[1], X_train_dl.shape[2]), 
        num_classes=6,
        sequence_length=X_train_dl.shape[1]
    )
    model_cnn_lstm, hist_cnn_lstm = dl_trainer.train_dl_model(
        model_cnn_lstm, X_train_dl, y_train_dl, X_val_dl, y_val_dl,
        epochs=100, batch_size=32, model_name='cnn_lstm'
    )
    
    # LSTM Model
    print("\n[2/3] LSTM Model")
    model_lstm = DLModels.build_lstm(
        input_shape=(X_train_dl.shape[1], X_train_dl.shape[2]), 
        num_classes=6
    )
    model_lstm, hist_lstm = dl_trainer.train_dl_model(
        model_lstm, X_train_dl, y_train_dl, X_val_dl, y_val_dl,
        epochs=100, batch_size=32, model_name='lstm'
    )
    
    # CNN Model
    print("\n[3/3] CNN Model")
    model_cnn = DLModels.build_cnn(
        input_shape=(X_train_dl.shape[1], X_train_dl.shape[2]), 
        num_classes=6
    )
    model_cnn, hist_cnn = dl_trainer.train_dl_model(
        model_cnn, X_train_dl, y_train_dl, X_val_dl, y_val_dl,
        epochs=100, batch_size=32, model_name='cnn'
    )
    
    # ========== EVALUATION ==========
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET")
    print("="*60)
    
    print("\nMachine Learning Models:")
    for model_name in ['svm', 'xgb', 'rf']:
        print(f"  {model_name.upper()}: {ml_results[model_name]['accuracy']:.4f}")
    
    print("\nDeep Learning Models:")
    dl_results = {}
    
    acc_cnn_lstm, f1_cnn_lstm, y_pred_cnn_lstm = dl_trainer.evaluate_model(
        model_cnn_lstm, X_test_dl, y_test, 'CNN-LSTM'
    )
    dl_results['cnn_lstm'] = {'accuracy': acc_cnn_lstm, 'f1': f1_cnn_lstm, 'predictions': y_pred_cnn_lstm}
    
    acc_lstm, f1_lstm, y_pred_lstm = dl_trainer.evaluate_model(
        model_lstm, X_test_dl, y_test, 'LSTM'
    )
    dl_results['lstm'] = {'accuracy': acc_lstm, 'f1': f1_lstm, 'predictions': y_pred_lstm}
    
    acc_cnn, f1_cnn, y_pred_cnn = dl_trainer.evaluate_model(
        model_cnn, X_test_dl, y_test, 'CNN'
    )
    dl_results['cnn'] = {'accuracy': acc_cnn, 'f1': f1_cnn, 'predictions': y_pred_cnn}
    
    # ========== SUMMARY ==========
    print("\n" + "="*60)
    print("ACCURACY SUMMARY")
    print("="*60)
    
    all_results = {**ml_results, **dl_results}
    results_df = pd.DataFrame({
        'Model': list(all_results.keys()),
        'Accuracy': [all_results[m]['accuracy'] for m in all_results.keys()],
        'F1-Score': [all_results[m]['f1'] if 'f1' in all_results[m] else 0 for m in all_results.keys()]
    }).sort_values('Accuracy', ascending=False)
    
    print("\n" + results_df.to_string(index=False))
    
    best_model = results_df.iloc[0]['Model']
    best_accuracy = results_df.iloc[0]['Accuracy']
    print(f"\nBest Performing Model: {best_model.upper()}")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    
    # ========== SAVE MODELS ==========
    print("\n" + "="*60)
    print("SAVING MODELS")
    print("="*60)
    
    model_cnn_lstm.save('./models/cnn_lstm_model.h5')
    model_lstm.save('./models/lstm_model.h5')
    model_cnn.save('./models/cnn_model.h5')
    
    print("DL models saved to ./models/")
    
    # Save ML models
    import joblib
    os.makedirs('./models', exist_ok=True)
    joblib.dump(ml_trainer.models['xgb'], './models/xgb_model.pkl')
    joblib.dump(ml_trainer.models['svm'], './models/svm_model.pkl')
    joblib.dump(ml_trainer.models['rf'], './models/rf_model.pkl')
    print("ML models saved to ./models/")
    
    return {
        'ml_models': ml_trainer,
        'dl_models': {
            'cnn_lstm': model_cnn_lstm,
            'lstm': model_lstm,
            'cnn': model_cnn
        },
        'results': results_df,
        'test_data': (X_test_norm, X_test_dl, y_test),
        'activity_labels': activity_labels
    }


if __name__ == "__main__":
    results = main()
