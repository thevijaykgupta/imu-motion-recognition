"""
Visualization and Analysis Module
Includes confusion matrices, training curves, feature importance, and signal analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from scipy import signal
import os

class VisualizationEngine:
    """Generate comprehensive visualizations for model analysis"""
    
    def __init__(self, output_dir='./visualizations'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (14, 6)
    
    def plot_training_history(self, histories, model_names, metric='accuracy'):
        """Plot training and validation curves"""
        fig, axes = plt.subplots(1, len(histories), figsize=(16, 5))
        
        if len(histories) == 1:
            axes = [axes]
        
        for idx, (history, name) in enumerate(zip(histories, model_names)):
            train_metric = history.history[metric]
            val_metric = history.history[f'val_{metric}']
            epochs = range(1, len(train_metric) + 1)
            
            axes[idx].plot(epochs, train_metric, 'b-', label='Training', linewidth=2)
            axes[idx].plot(epochs, val_metric, 'r-', label='Validation', linewidth=2)
            axes[idx].set_title(f'{name} - {metric.capitalize()}', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel(metric.capitalize())
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/training_history_{metric}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training history plot saved")
    
    def plot_confusion_matrices(self, y_true, predictions_dict, activity_labels):
        """Plot confusion matrices for multiple models"""
        n_models = len(predictions_dict)
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()
        
        for idx, (model_name, y_pred) in enumerate(predictions_dict.items()):
            cm = confusion_matrix(y_true, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=activity_labels, yticklabels=activity_labels,
                       cbar_kws={'label': 'Count'})
            
            axes[idx].set_title(f'{model_name.upper()}', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrices plot saved")
    
    def plot_model_comparison(self, results_df):
        """Plot model accuracy comparison"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#2ecc71' if x == results_df['Accuracy'].max() else '#3498db' 
                 for x in results_df['Accuracy']]
        
        bars = ax.barh(results_df['Model'], results_df['Accuracy'], color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{width:.4f}', ha='left', va='center', fontweight='bold')
        
        ax.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Model comparison plot saved")
    
    def plot_class_distribution(self, y_train, y_test, activity_labels):
        """Plot class distribution in train and test sets"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Training set
        unique_train, counts_train = np.unique(y_train, return_counts=True)
        axes[0].bar(range(len(activity_labels)), counts_train, 
                   color='#3498db', edgecolor='black', linewidth=1.5)
        axes[0].set_xticks(range(len(activity_labels)))
        axes[0].set_xticklabels(activity_labels, rotation=45, ha='right')
        axes[0].set_title('Training Set Distribution', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Number of Samples')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Test set
        unique_test, counts_test = np.unique(y_test, return_counts=True)
        axes[1].bar(range(len(activity_labels)), counts_test, 
                   color='#e74c3c', edgecolor='black', linewidth=1.5)
        axes[1].set_xticks(range(len(activity_labels)))
        axes[1].set_xticklabels(activity_labels, rotation=45, ha='right')
        axes[1].set_title('Test Set Distribution', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Number of Samples')
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/class_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Class distribution plot saved")
    
    def plot_raw_signals(self, X_data, y_labels, activity_labels, n_activities=3):
        """Plot sample raw IMU signals for each activity"""
        fig, axes = plt.subplots(n_activities, 2, figsize=(14, 10))
        
        sensor_names = ['Accel X,Y,Z', 'Gyro X,Y,Z']
        
        for activity_idx in range(min(n_activities, len(activity_labels))):
            # Find sample for this activity
            sample_idx = np.where(y_labels == activity_idx)[0][0]
            sample = X_data[sample_idx]
            
            # Plot accelerometer (first 3 features)
            accel = sample[:3]
            axes[activity_idx, 0].plot(accel[0], label='X', linewidth=1.5)
            axes[activity_idx, 0].plot(accel[1], label='Y', linewidth=1.5)
            axes[activity_idx, 0].plot(accel[2], label='Z', linewidth=1.5)
            axes[activity_idx, 0].set_title(f'{activity_labels[activity_idx]} - Accelerometer', 
                                           fontsize=11, fontweight='bold')
            axes[activity_idx, 0].set_ylabel('Acceleration (m/s²)')
            axes[activity_idx, 0].legend(loc='upper right')
            axes[activity_idx, 0].grid(True, alpha=0.3)
            
            # Plot gyroscope (last 3 features)
            gyro = sample[3:6] if len(sample) >= 6 else sample[3:]
            axes[activity_idx, 1].plot(gyro[0] if len(gyro) > 0 else [0], 
                                      label='X', linewidth=1.5)
            axes[activity_idx, 1].plot(gyro[1] if len(gyro) > 1 else [0], 
                                      label='Y', linewidth=1.5)
            axes[activity_idx, 1].plot(gyro[2] if len(gyro) > 2 else [0], 
                                      label='Z', linewidth=1.5)
            axes[activity_idx, 1].set_title(f'{activity_labels[activity_idx]} - Gyroscope', 
                                           fontsize=11, fontweight='bold')
            axes[activity_idx, 1].set_ylabel('Angular Velocity (deg/s)')
            axes[activity_idx, 1].legend(loc='upper right')
            axes[activity_idx, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/raw_signals.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Raw signals plot saved")
    
    def plot_frequency_analysis(self, X_data, y_labels, activity_labels, n_samples=3):
        """Plot frequency domain analysis (FFT) for different activities"""
        # Load raw inertial signals for FFT analysis
        dataset_path = './data/UCI HAR Dataset'
        body_acc_x = np.loadtxt(f'{dataset_path}/train/Inertial Signals/body_acc_x_train.txt')
        body_gyro_x = np.loadtxt(f'{dataset_path}/train/Inertial Signals/body_gyro_x_train.txt')

        fig, axes = plt.subplots(n_samples, 2, figsize=(14, 10))

        for activity_idx in range(n_samples):
            if activity_idx >= len(activity_labels):
                break

            # Find sample for this activity
            sample_idx = np.where(y_labels == activity_idx)[0][0]

            # Use raw inertial signals for FFT
            accel_signal = body_acc_x[sample_idx]
            gyro_signal = body_gyro_x[sample_idx]

            # Compute FFT for accelerometer
            fft_accel = np.abs(np.fft.fft(accel_signal))
            freqs = np.fft.fftfreq(len(accel_signal), d=0.02)  # 50Hz sampling

            axes[activity_idx, 0].semilogy(freqs[:len(freqs)//2],
                                          fft_accel[:len(fft_accel)//2], linewidth=1.5)
            axes[activity_idx, 0].set_title(f'{activity_labels[activity_idx]} - Accel FFT',
                                           fontsize=11, fontweight='bold')
            axes[activity_idx, 0].set_xlabel('Frequency (Hz)')
            axes[activity_idx, 0].set_ylabel('Power')
            axes[activity_idx, 0].grid(True, alpha=0.3, which='both')

            # Compute FFT for gyroscope
            fft_gyro = np.abs(np.fft.fft(gyro_signal))
            freqs = np.fft.fftfreq(len(gyro_signal), d=0.02)

            axes[activity_idx, 1].semilogy(freqs[:len(freqs)//2],
                                          fft_gyro[:len(fft_gyro)//2], linewidth=1.5, color='red')
            axes[activity_idx, 1].set_title(f'{activity_labels[activity_idx]} - Gyro FFT',
                                           fontsize=11, fontweight='bold')
            axes[activity_idx, 1].set_xlabel('Frequency (Hz)')
            axes[activity_idx, 1].set_ylabel('Power')
            axes[activity_idx, 1].grid(True, alpha=0.3, which='both')

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/frequency_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Frequency analysis plot saved")
    
    def plot_per_class_metrics(self, y_true, predictions_dict, activity_labels):
        """Plot per-class accuracy for each model"""
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        metrics_data = []
        
        for model_name, y_pred in predictions_dict.items():
            for class_idx, label in enumerate(activity_labels):
                mask = y_true == class_idx
                
                if np.sum(mask) > 0:
                    class_acc = np.sum((y_pred[mask] == y_true[mask])) / np.sum(mask)
                    metrics_data.append({
                        'Model': model_name,
                        'Activity': label,
                        'Accuracy': class_acc
                    })
        
        df = pd.DataFrame(metrics_data)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        pivot_data = df.pivot(index='Activity', columns='Model', values='Accuracy')
        pivot_data.plot(kind='bar', ax=ax, width=0.8, edgecolor='black', linewidth=1.5)
        
        ax.set_title('Per-Class Accuracy Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Activity')
        ax.legend(title='Model', loc='lower right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/per_class_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Per-class metrics plot saved")
    
    def plot_confidence_distribution(self, confidences, predictions, ground_truth, 
                                    activity_labels):
        """Plot confidence score distribution"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Overall confidence distribution
        axes[0].hist(confidences, bins=50, color='#3498db', edgecolor='black', alpha=0.7)
        axes[0].axvline(np.mean(confidences), color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {np.mean(confidences):.3f}')
        axes[0].set_title('Overall Confidence Distribution', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Confidence Score')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Confidence for correct vs incorrect predictions
        correct_conf = [confidences[i] for i in range(len(predictions)) 
                       if predictions[i] == ground_truth[i]]
        incorrect_conf = [confidences[i] for i in range(len(predictions)) 
                         if predictions[i] != ground_truth[i]]
        
        axes[1].hist([correct_conf, incorrect_conf], bins=40, 
                    label=['Correct', 'Incorrect'], 
                    color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
        axes[1].set_title('Confidence: Correct vs Incorrect Predictions', 
                         fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Confidence Score')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/confidence_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confidence distribution plot saved")


class ComprehensiveAnalysis:
    """Generate comprehensive analysis report"""
    
    def __init__(self, output_dir='./analysis_report'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_report(self, results_dict, model_predictions, y_test, activity_labels):
        """Generate comprehensive analysis report"""
        report = []
        report.append("="*80)
        report.append("HUMAN MOTION RECOGNITION - COMPREHENSIVE ANALYSIS REPORT")
        report.append("="*80)
        report.append("")
        
        # 1. Model Performance Summary
        report.append("1. MODEL PERFORMANCE SUMMARY")
        report.append("-"*80)
        
        for model_name, metrics in results_dict.items():
            report.append(f"\n{model_name.upper()}:")
            report.append(f"  Accuracy:  {metrics.get('accuracy', 0):.4f}")
            report.append(f"  F1-Score:  {metrics.get('f1', 0):.4f}")
        
        report.append("\n" + "="*80)
        
        # 2. Per-Class Performance
        report.append("\n2. PER-CLASS PERFORMANCE ANALYSIS")
        report.append("-"*80)
        
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        best_model_name = list(model_predictions.keys())[0]
        y_pred = model_predictions[best_model_name]
        
        for class_idx, activity in enumerate(activity_labels):
            mask = y_test == class_idx
            if np.sum(mask) > 0:
                class_acc = np.sum((y_pred[mask] == y_test[mask])) / np.sum(mask)
                report.append(f"\n{activity}:")
                report.append(f"  Accuracy:     {class_acc:.4f}")
                report.append(f"  Samples:      {np.sum(mask)}")
        
        report.append("\n" + "="*80)
        
        # 3. Recommendations
        report.append("\n3. RECOMMENDATIONS & INSIGHTS")
        report.append("-"*80)
        report.append("\n[OK] Model Selection:")
        report.append("  - CNN-LSTM provides best temporal-spatial feature extraction")
        report.append("  - XGBoost offers fastest inference with comparable accuracy")
        report.append("  - SVM requires less memory but may struggle with complex patterns")
        
        report.append("\n[OK] Performance Optimization:")
        report.append("  - Apply data augmentation for underrepresented activities")
        report.append("  - Fine-tune model hyperparameters with cross-validation")
        report.append("  - Implement ensemble methods combining multiple models")
        report.append("  - Use temporal smoothing for real-time predictions")
        
        report.append("\n[OK] Deployment Considerations:")
        report.append("  - Deploy CNN-LSTM for production systems requiring high accuracy")
        report.append("  - Use XGBoost on edge devices with memory constraints")
        report.append("  - Implement fallback to simpler models if main model fails")
        report.append("  - Monitor model drift with continuous evaluation")
        
        report.append("\n" + "="*80)
        
        # Save report
        report_text = "\n".join(report)
        
        with open(f'{self.output_dir}/analysis_report.txt', 'w') as f:
            f.write(report_text)
        
        print(f"[OK] Analysis report saved to {self.output_dir}/analysis_report.txt")
        
        return report_text


if __name__ == "__main__":
    print("Visualization and Analysis Module")
    print("Used in conjunction with imu_motion_recognition.py")
    print("\nUsage:")
    print("  viz = VisualizationEngine('./visualizations')")
    print("  viz.plot_training_history(histories, model_names)")
    print("  viz.plot_confusion_matrices(y_true, predictions, labels)")
