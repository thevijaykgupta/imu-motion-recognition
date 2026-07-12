"""
Real-Time Inference Engine for Human Motion Recognition
Works with live IMU data from smartphones, smartwatches, or sensors
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib
from collections import deque
import json
from datetime import datetime
from scipy.signal import butter, filtfilt

class RealTimeIMUPredictor:
    """Real-time activity recognition from IMU streams"""
    
    def __init__(self, model_path='./models/cnn_lstm_model.h5', 
                 model_type='cnn_lstm', window_size=128, sampling_rate=50):
        """
        Initialize real-time predictor
        Args:
            model_path: Path to trained model
            model_type: 'cnn_lstm', 'lstm', 'cnn', 'xgb', or 'svm'
            window_size: Number of samples per prediction window
            sampling_rate: IMU sampling rate (Hz)
        """
        self.model_path = model_path
        self.model_type = model_type
        self.window_size = window_size
        self.sampling_rate = sampling_rate
        
        self.activity_labels = {
            0: 'WALKING', 1: 'WALKING_UPSTAIRS', 2: 'WALKING_DOWNSTAIRS',
            3: 'SITTING', 4: 'STANDING', 5: 'LAYING'
        }
        
        # Load model
        if model_type in ['cnn_lstm', 'lstm', 'cnn']:
            self.model = keras.models.load_model(model_path)
            self.is_dl_model = True
        else:
            self.model = joblib.load(model_path)
            self.is_dl_model = False
        
        # Signal buffer for streaming data
        self.signal_buffer = deque(maxlen=window_size)
        
        # Statistics for normalization
        self.mean = 0
        self.std = 1
        
        # Smoothing for predictions
        self.prediction_history = deque(maxlen=5)
        
    def set_normalization_stats(self, mean, std):
        """Set mean and std for feature normalization"""
        self.mean = mean
        self.std = std
    
    def normalize_signal(self, signal):
        """Normalize signal using pre-computed statistics"""
        if self.std == 0:
            return signal
        return (signal - self.mean) / self.std
    
    def butter_lowpass_filter(self, data, cutoff=5, order=4):
        """Apply low-pass filter to remove noise"""
        nyquist = self.sampling_rate / 2
        normalized_cutoff = cutoff / nyquist
        if normalized_cutoff >= 1:
            return data
        b, a = butter(order, normalized_cutoff, btype='low')
        return filtfilt(b, a, data)
    
    def add_imu_sample(self, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z):
        """
        Add single IMU sample to buffer
        Args:
            accel_x, accel_y, accel_z: Accelerometer values (m/s²)
            gyro_x, gyro_y, gyro_z: Gyroscope values (deg/s)
        """
        # Combine all 6 axes
        sample = np.array([accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z])
        
        # Normalize
        sample = self.normalize_signal(sample)
        
        self.signal_buffer.append(sample)
    
    def add_imu_batch(self, imu_data):
        """
        Add batch of IMU samples
        Args:
            imu_data: Array of shape (N, 6) with [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
        """
        for sample in imu_data:
            self.add_imu_sample(*sample)
    
    def is_buffer_ready(self):
        """Check if buffer has enough samples for prediction"""
        return len(self.signal_buffer) == self.window_size
    
    def predict(self, return_probabilities=False):
        """
        Predict activity from buffered IMU data
        Returns:
            activity_label, confidence_score, [probabilities]
        """
        if not self.is_buffer_ready():
            return None, 0.0
        
        # Prepare data
        X = np.array(list(self.signal_buffer))
        
        if self.is_dl_model:
            # DL models expect (1, window_size, features)
            X = X.reshape(1, self.window_size, -1)
            
            # Predict
            probabilities = self.model.predict(X, verbose=0)[0]
            prediction = np.argmax(probabilities)
            confidence = np.max(probabilities)
            
        else:
            # ML models expect (1, features)
            X = X.reshape(1, -1)
            
            # Predict
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]
            confidence = np.max(probabilities)
        
        # Apply smoothing
        self.prediction_history.append(prediction)
        
        activity = self.activity_labels.get(int(prediction), 'UNKNOWN')
        
        if return_probabilities:
            prob_dict = {self.activity_labels[i]: float(probabilities[i]) 
                        for i in range(len(probabilities))}
            return activity, float(confidence), prob_dict
        
        return activity, float(confidence)
    
    def predict_smoothed(self, return_probabilities=False):
        """
        Predict activity with temporal smoothing
        Returns most common activity from last 5 predictions
        """
        if not self.prediction_history:
            return None, 0.0
        
        # Get most common prediction
        predictions = list(self.prediction_history)
        unique, counts = np.unique(predictions, return_counts=True)
        smoothed_pred = unique[np.argmax(counts)]
        
        activity = self.activity_labels.get(int(smoothed_pred), 'UNKNOWN')
        confidence = np.max(counts) / len(predictions)
        
        if return_probabilities:
            # Get average probabilities from history
            prob_dict = {}
            for act_idx in range(6):
                prob_dict[self.activity_labels[act_idx]] = 0.0
            
            for pred in predictions:
                prob_dict[self.activity_labels[int(pred)]] += 1.0 / len(predictions)
            
            return activity, float(confidence), prob_dict
        
        return activity, float(confidence)
    
    def reset_buffer(self):
        """Clear buffer for new prediction cycle"""
        self.signal_buffer.clear()
        self.prediction_history.clear()


class SimulatedIMUStream:
    """Simulate IMU data stream for testing"""
    
    def __init__(self, test_data, test_labels, activity_labels, 
                 window_size=128, overlap=0.5):
        """
        Args:
            test_data: Test IMU data (N, features)
            test_labels: Ground truth labels
            activity_labels: Label to activity name mapping
            window_size: Window size for predictions
            overlap: Overlap between windows
        """
        self.test_data = test_data
        self.test_labels = test_labels
        self.activity_labels = activity_labels
        self.window_size = window_size
        self.step = int(window_size * (1 - overlap))
        
        self.current_sample = 0
        self.windows = []
        self.true_activities = []
        
        self._create_windows()
    
    def _create_windows(self):
        """Create overlapping windows from test data"""
        for i in range(0, len(self.test_data) - self.window_size + 1, self.step):
            window = self.test_data[i:i + self.window_size]
            label = self.test_labels[i]
            
            self.windows.append(window)
            self.true_activities.append(self.activity_labels[int(label)])
    
    def get_next_window(self):
        """Get next window for testing"""
        if self.current_sample >= len(self.windows):
            return None, None
        
        window = self.windows[self.current_sample]
        true_activity = self.true_activities[self.current_sample]
        
        self.current_sample += 1
        return window, true_activity
    
    def reset(self):
        """Reset stream to beginning"""
        self.current_sample = 0


class RealTimeEvaluator:
    """Evaluate model performance on simulated real-time stream"""
    
    def __init__(self):
        self.predictions = []
        self.ground_truth = []
        self.confidences = []
    
    def add_prediction(self, predicted_activity, ground_truth_activity, confidence):
        """Record prediction"""
        self.predictions.append(predicted_activity)
        self.ground_truth.append(ground_truth_activity)
        self.confidences.append(confidence)
    
    def compute_metrics(self):
        """Compute accuracy and other metrics"""
        from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
        
        accuracy = accuracy_score(self.ground_truth, self.predictions)
        
        # Map activities to indices for confusion matrix
        activity_map = {
            'WALKING': 0, 'WALKING_UPSTAIRS': 1, 'WALKING_DOWNSTAIRS': 2,
            'SITTING': 3, 'STANDING': 4, 'LAYING': 5
        }
        
        y_true = np.array([activity_map[a] for a in self.ground_truth])
        y_pred = np.array([activity_map[a] for a in self.predictions])
        
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'avg_confidence': np.mean(self.confidences),
            'predictions': self.predictions,
            'ground_truth': self.ground_truth
        }


# ============================================================================
# TESTING SCRIPT
# ============================================================================

def test_real_time_prediction(model_path, test_data, test_labels, 
                             model_type='cnn_lstm', mean=0, std=1):
    """Test real-time prediction on simulated stream"""
    
    activity_labels = {
        0: 'WALKING', 1: 'WALKING_UPSTAIRS', 2: 'WALKING_DOWNSTAIRS',
        3: 'SITTING', 4: 'STANDING', 5: 'LAYING'
    }
    
    # Initialize predictor
    predictor = RealTimeIMUPredictor(
        model_path=model_path,
        model_type=model_type,
        window_size=128,
        sampling_rate=50
    )
    predictor.set_normalization_stats(mean, std)
    
    # Create simulated stream
    stream = SimulatedIMUStream(test_data, test_labels, activity_labels)
    
    # Evaluate
    evaluator = RealTimeEvaluator()
    
    print("\n" + "="*60)
    print("REAL-TIME PREDICTION TEST")
    print("="*60)
    print(f"Model: {model_type}")
    print(f"Total windows to process: {len(stream.windows)}\n")
    
    window_count = 0
    while True:
        window, true_activity = stream.get_next_window()
        
        if window is None:
            break
        
        # Add samples to predictor
        predictor.add_imu_batch(window)
        
        if predictor.is_buffer_ready():
            # Get prediction
            predicted_activity, confidence = predictor.predict()
            
            evaluator.add_prediction(predicted_activity, true_activity, confidence)
            
            window_count += 1
            if window_count % 20 == 0:
                print(f"Processed {window_count} windows...")
            
            # Reset buffer for next prediction
            predictor.reset_buffer()
    
    # Print results
    metrics = evaluator.compute_metrics()
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Average Confidence: {metrics['avg_confidence']:.4f}")
    print(f"\nTotal Predictions: {len(evaluator.predictions)}")
    
    return metrics


if __name__ == "__main__":
    # Example usage
    print("Real-Time IMU Activity Recognition")
    print("This module provides real-time prediction capabilities")
    print("\nUsage:")
    print("  predictor = RealTimeIMUPredictor('./models/cnn_lstm_model.h5')")
    print("  predictor.add_imu_sample(accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z)")
    print("  if predictor.is_buffer_ready():")
    print("      activity, confidence = predictor.predict()")
