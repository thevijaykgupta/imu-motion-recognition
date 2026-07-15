"""
COMPLETE END-TO-END PIPELINE EXECUTOR
Integrates: Data Loading → Preprocessing → ML & DL Training → Evaluation → Visualization
Run this script for full human motion recognition project
"""

import numpy as np
import os
import sys
from imu_motion_recognition import (
    IMUDataLoader, SignalPreprocessor, FeatureExtractor,
    MLModels, DLModels, ModelTrainer
)
from realtime_inference import RealTimeIMUPredictor, RealTimeEvaluator, SimulatedIMUStream
from analysis_visualization import VisualizationEngine, ComprehensiveAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

class CompleteHARPipeline:
    """Complete Human Activity Recognition Pipeline"""
    
    def __init__(self, dataset_path='./data/UCI HAR Dataset', models_dir='./models', viz_dir='./visualizations'):
        self.dataset_path = dataset_path
        self.models_dir = models_dir
        self.viz_dir = viz_dir
        
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(viz_dir, exist_ok=True)
        
        self.activity_labels = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 
                               'SITTING', 'STANDING', 'LAYING']
        
        self.results = {}
    
    def step_1_download_data(self):
        """Step 1: Download UCI HAR Dataset"""
        print("\n" + "="*70)
        print("STEP 1: DOWNLOADING DATASET")
        print("="*70)
        
        loader = IMUDataLoader(self.dataset_path)
        loader.download_dataset()
        
        return loader
    
    def step_2_load_and_preprocess(self, loader):
        """Step 2: Load and preprocess data"""
        print("\n" + "="*70)
        print("STEP 2: LOADING & PREPROCESSING DATA")
        print("="*70)
        
        # Load data
        print("\nLoading UCI HAR Dataset...")
        X_train, y_train, X_test, y_test = loader.load_data()
        print(f"[OK] Training set: {X_train.shape}")
        print(f"[OK] Test set: {X_test.shape}")
        
        # Preprocess
        print("\nPreprocessing signals...")
        preprocessor = SignalPreprocessor()
        X_train_norm, X_test_norm = preprocessor.normalize_signal(X_train, X_test)
        print("[OK] Data normalized (StandardScaler)")
        
        # Create train-val split for DL models
        X_train_dl, X_val_dl, y_train_dl, y_val_dl = train_test_split(
            X_train_norm, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # Reshape for DL models
        X_train_dl = X_train_dl.reshape(X_train_dl.shape[0], X_train_dl.shape[1], 1)
        X_val_dl = X_val_dl.reshape(X_val_dl.shape[0], X_val_dl.shape[1], 1)
        X_test_dl = X_test_norm.reshape(X_test_norm.shape[0], X_test_norm.shape[1], 1)
        
        print(f"[OK] DL shapes - Train: {X_train_dl.shape}, Val: {X_val_dl.shape}, Test: {X_test_dl.shape}")
        
        return {
            'X_train_norm': X_train_norm,
            'X_test_norm': X_test_norm,
            'X_train_dl': X_train_dl,
            'X_val_dl': X_val_dl,
            'X_test_dl': X_test_dl,
            'y_train': y_train,
            'y_train_dl': y_train_dl,
            'y_val_dl': y_val_dl,
            'y_test': y_test,
            'preprocessor': preprocessor
        }
    
    def step_3_train_ml_models(self, data):
        """Step 3: Train machine learning models"""
        print("\n" + "="*70)
        print("STEP 3: TRAINING MACHINE LEARNING MODELS")
        print("="*70)
        
        ml_models = MLModels()
        ml_models.train_models(data['X_train_norm'], data['y_train'])
        
        print("\nML Model Evaluation:")
        ml_results = {}
        for model_name in ['svm', 'xgb', 'rf']:
            y_pred = ml_models.predict(data['X_test_norm'], model_name)
            acc = accuracy_score(data['y_test'], y_pred)
            f1 = f1_score(data['y_test'], y_pred, average='weighted')
            ml_results[model_name] = {
                'accuracy': acc, 
                'f1': f1, 
                'predictions': y_pred,
                'model': ml_models.models[model_name]
            }
            print(f"  {model_name.upper():15} | Accuracy: {acc:.4f} | F1: {f1:.4f}")
        
        return ml_models, ml_results
    
    def step_4_train_dl_models(self, data):
        """Step 4: Train deep learning models"""
        print("\n" + "="*70)
        print("STEP 4: TRAINING DEEP LEARNING MODELS")
        print("="*70)
        
        trainer = ModelTrainer(num_classes=6)
        dl_models = {}
        dl_results = {}
        histories = {}
        
        # CNN-LSTM (Hybrid)
        print("\n[1/3] CNN-LSTM (Hybrid) Model")
        model_cnn_lstm = DLModels.build_cnn_lstm(
            input_shape=(data['X_train_dl'].shape[1], data['X_train_dl'].shape[2]),
            num_classes=6,
            sequence_length=data['X_train_dl'].shape[1]
        )
        model_cnn_lstm, hist = trainer.train_dl_model(
            model_cnn_lstm, data['X_train_dl'], data['y_train_dl'],
            data['X_val_dl'], data['y_val_dl'],
            epochs=100, batch_size=32, model_name='cnn_lstm'
        )
        dl_models['cnn_lstm'] = model_cnn_lstm
        histories['cnn_lstm'] = hist
        
        # LSTM
        print("\n[2/3] LSTM Model")
        model_lstm = DLModels.build_lstm(
            input_shape=(data['X_train_dl'].shape[1], data['X_train_dl'].shape[2]),
            num_classes=6
        )
        model_lstm, hist = trainer.train_dl_model(
            model_lstm, data['X_train_dl'], data['y_train_dl'],
            data['X_val_dl'], data['y_val_dl'],
            epochs=100, batch_size=32, model_name='lstm'
        )
        dl_models['lstm'] = model_lstm
        histories['lstm'] = hist
        
        # CNN
        print("\n[3/3] CNN Model")
        model_cnn = DLModels.build_cnn(
            input_shape=(data['X_train_dl'].shape[1], data['X_train_dl'].shape[2]),
            num_classes=6
        )
        model_cnn, hist = trainer.train_dl_model(
            model_cnn, data['X_train_dl'], data['y_train_dl'],
            data['X_val_dl'], data['y_val_dl'],
            epochs=100, batch_size=32, model_name='cnn'
        )
        dl_models['cnn'] = model_cnn
        histories['cnn'] = hist
        
        # Evaluate DL models
        print("\nDL Model Evaluation:")
        for model_name, model in dl_models.items():
            y_pred = np.argmax(model.predict(data['X_test_dl'], verbose=0), axis=1)
            acc = accuracy_score(data['y_test'], y_pred)
            f1 = f1_score(data['y_test'], y_pred, average='weighted')
            dl_results[model_name] = {
                'accuracy': acc,
                'f1': f1,
                'predictions': y_pred,
                'model': model
            }
            print(f"  {model_name.upper():15} | Accuracy: {acc:.4f} | F1: {f1:.4f}")
        
        return dl_models, dl_results, trainer, histories
    
    def step_5_comprehensive_evaluation(self, ml_results, dl_results, data):
        """Step 5: Comprehensive model evaluation"""
        print("\n" + "="*70)
        print("STEP 5: COMPREHENSIVE EVALUATION")
        print("="*70)
        
        # Aggregate results
        all_results = {**ml_results, **dl_results}
        
        # Summary
        print("\n" + "="*62)
        print("                    MODEL PERFORMANCE COMPARISON")
        print("="*62)
        
        results_list = []
        for model_name in sorted(all_results.keys(), 
                                key=lambda x: all_results[x]['accuracy'], reverse=True):
            acc = all_results[model_name]['accuracy']
            f1 = all_results[model_name]['f1']
            model_type = "ML" if model_name in ml_results else "DL"
            
            results_list.append({
                'Model': model_name,
                'Type': model_type,
                'Accuracy': acc,
                'F1-Score': f1
            })
            
            print(f"| {model_name:15} [{model_type}]  Acc: {acc:.4f}  F1: {f1:.4f} {' '*20}")

        print("="*62)
        
        best_model = results_list[0]
        print(f"\n[OK] Best Model: {best_model['Model'].upper()}")
        print(f"  Accuracy: {best_model['Accuracy']:.4f} | F1-Score: {best_model['F1-Score']:.4f}")
        
        return all_results, results_list
    
    def step_6_save_models(self, ml_models, dl_models):
        """Step 6: Save all trained models"""
        print("\n" + "="*70)
        print("STEP 6: SAVING MODELS")
        print("="*70)
        
        import joblib
        
        # Save DL models
        print("\nSaving Deep Learning Models:")
        for model_name, model in dl_models.items():
            path = f'{self.models_dir}/{model_name}_model.h5'
            model.save(path)
            print(f"  [OK] {model_name}: {path}")
        
        # Save ML models
        print("\nSaving Machine Learning Models:")
        for model_name in ['svm', 'xgb', 'rf']:
            path = f'{self.models_dir}/{model_name}_model.pkl'
            joblib.dump(ml_models.models[model_name], path)
            print(f"  [OK] {model_name}: {path}")
    
    def step_7_visualization(self, all_results, ml_results, dl_results, 
                            data, histories):
        """Step 7: Generate visualizations"""
        print("\n" + "="*70)
        print("STEP 7: GENERATING VISUALIZATIONS")
        print("="*70)
        
        viz = VisualizationEngine(self.viz_dir)
        
        # Training curves
        print("\nGenerating training curves...")
        histories_list = [histories['cnn_lstm'], histories['lstm'], histories['cnn']]
        model_names = ['CNN-LSTM', 'LSTM', 'CNN']
        viz.plot_training_history(histories_list, model_names, metric='accuracy')
        
        # Confusion matrices
        print("Generating confusion matrices...")
        predictions_dict = {name: results['predictions'] 
                           for name, results in all_results.items()}
        viz.plot_confusion_matrices(data['y_test'], predictions_dict, self.activity_labels)
        
        # Model comparison
        print("Generating model comparison...")
        import pandas as pd
        results_df = pd.DataFrame({
            'Model': list(all_results.keys()),
            'Accuracy': [all_results[m]['accuracy'] for m in all_results.keys()]
        }).sort_values('Accuracy', ascending=False)
        viz.plot_model_comparison(results_df)
        
        # Class distribution
        print("Generating class distribution...")
        viz.plot_class_distribution(data['y_train'], data['y_test'], self.activity_labels)
        
        # Raw signals
        print("Generating raw signal plots...")
        viz.plot_raw_signals(data['X_train_norm'], data['y_train'], self.activity_labels)
        
        # Frequency analysis
        print("Generating frequency analysis...")
        viz.plot_frequency_analysis(data['X_train_norm'], data['y_train'], 
                                    self.activity_labels)
        
        # Per-class metrics
        print("Generating per-class metrics...")
        viz.plot_per_class_metrics(data['y_test'], predictions_dict, self.activity_labels)
        
        print(f"\n[OK] All visualizations saved to: {self.viz_dir}/")
    
    def step_8_real_time_testing(self, dl_models, data):
        """Step 8: Test real-time inference"""
        print("\n" + "="*70)
        print("STEP 8: REAL-TIME INFERENCE TESTING")
        print("="*70)
        
        from realtime_inference import RealTimeIMUPredictor, SimulatedIMUStream
        
        # Test with CNN-LSTM
        print("\nTesting Real-Time Inference (CNN-LSTM)...")
        
        predictor = RealTimeIMUPredictor(
            model_path=f'{self.models_dir}/cnn_lstm_model.h5',
            model_type='cnn_lstm',
            window_size=128,
            sampling_rate=50
        )
        
        # Set normalization stats
        predictor.set_normalization_stats(
            mean=np.mean(data['X_train_norm']),
            std=np.std(data['X_train_norm'])
        )
        
        # Simulate stream
        stream = SimulatedIMUStream(data['X_test_norm'], data['y_test'], 
                                   {i: self.activity_labels[i] for i in range(6)})
        
        correct_predictions = 0
        total_windows = 0
        
        for _ in range(min(100, len(stream.windows))):
            window, true_activity = stream.get_next_window()
            if window is None:
                break
            
            predictor.add_imu_batch(window)
            
            if predictor.is_buffer_ready():
                predicted_activity, confidence = predictor.predict()
                
                if predicted_activity == true_activity:
                    correct_predictions += 1
                
                total_windows += 1
                predictor.reset_buffer()
        
        rt_accuracy = correct_predictions / total_windows if total_windows > 0 else 0
        print(f"[OK] Real-Time Accuracy (CNN-LSTM): {rt_accuracy:.4f}")
        print(f"  Processed {total_windows} windows")
    
    def step_9_generate_report(self, all_results, data):
        """Step 9: Generate comprehensive analysis report"""
        print("\n" + "="*70)
        print("STEP 9: GENERATING ANALYSIS REPORT")
        print("="*70)
        
        analyzer = ComprehensiveAnalysis(f'{self.viz_dir}/report')
        predictions_dict = {name: results['predictions'] 
                           for name, results in all_results.items()}
        report = analyzer.generate_report(all_results, predictions_dict, 
                                         data['y_test'], self.activity_labels)
        print("\n" + report)
    
    def run_complete_pipeline(self):
        """Execute complete pipeline"""
        print("\n")
        print("="*66)
        print("         HUMAN MOTION RECOGNITION - COMPLETE PIPELINE")
        print("      UCI HAR Dataset | ML & DL Models | Real-Time Inference")
        print("="*66)
        
        try:
            # Step 1: Download data
            loader = self.step_1_download_data()
            
            # Step 2: Load and preprocess
            data = self.step_2_load_and_preprocess(loader)
            
            # Step 3: Train ML models
            ml_models, ml_results = self.step_3_train_ml_models(data)
            
            # Step 4: Train DL models
            dl_models, dl_results, trainer, histories = self.step_4_train_dl_models(data)
            
            # Step 5: Comprehensive evaluation
            all_results, results_list = self.step_5_comprehensive_evaluation(
                ml_results, dl_results, data
            )
            
            # Step 6: Save models
            self.step_6_save_models(ml_models, dl_models)
            
            # Step 7: Generate visualizations
            self.step_7_visualization(all_results, ml_results, dl_results, data, histories)
            
            # Step 8: Real-time testing
            self.step_8_real_time_testing(dl_models, data)
            
            # Step 9: Generate report
            self.step_9_generate_report(all_results, data)
            
            # Final summary
            print("\n" + "="*66)
            print("                    PIPELINE EXECUTION COMPLETED")
            print("="*66)
            print("|  Data downloaded and preprocessed")
            print("|  ML models trained (SVM, XGBoost, Random Forest)")
            print("|  DL models trained (CNN-LSTM, LSTM, CNN)")
            print("|  Models evaluated and compared")
            print("|  Models saved to ./models/")
            print("|  Visualizations generated in ./visualizations/")
            print("|  Real-time inference tested")
            print("|  Analysis report generated")
            print("="*66)
            
            print(f"\nModels saved in: {self.models_dir}/")
            print(f"Visualizations in: {self.viz_dir}/")
            
            return True
            
        except Exception as e:
            print(f"\n[ERROR] Error during pipeline execution: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main execution point"""
    pipeline = CompleteHARPipeline()
    success = pipeline.run_complete_pipeline()
    
    if success:
        print("\n[OK] Pipeline completed successfully!")
        return 0
    else:
        print("\n[FAIL] Pipeline failed!")
        return 1


if __name__ == "__main__":
    exit(main())
