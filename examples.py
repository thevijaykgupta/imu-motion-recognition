"""
QUICK START GUIDE - Example Usage Patterns
Run individual components or full pipeline
"""

# ============================================================================
# EXAMPLE 1: Run Complete Pipeline (Recommended for First Time)
# ============================================================================

def example_1_complete_pipeline():
    """
    Run the entire pipeline from data download to evaluation
    This is the recommended starting point
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: COMPLETE PIPELINE")
    print("="*70)
    
    from run_complete_pipeline import CompleteHARPipeline
    
    pipeline = CompleteHARPipeline()
    success = pipeline.run_complete_pipeline()
    
    if success:
        print("\n[OK] Pipeline completed successfully!")
    else:
        print("\n[X] Pipeline failed!")


# ============================================================================
# EXAMPLE 2: Train Individual Models
# ============================================================================

def example_2_individual_models():
    """
    Train and evaluate individual models
    Useful for experimenting with specific models
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: TRAINING INDIVIDUAL MODELS")
    print("="*70)
    
    from imu_motion_recognition import IMUDataLoader, SignalPreprocessor, MLModels
    import numpy as np
    from sklearn.metrics import accuracy_score
    
    # Load data
    loader = IMUDataLoader()
    loader.download_dataset()
    X_train, y_train, X_test, y_test = loader.load_data()
    
    # Preprocess
    preprocessor = SignalPreprocessor()
    X_train_norm, X_test_norm = preprocessor.normalize_signal(X_train, X_test)
    
    # Train XGBoost only
    print("\nTraining XGBoost...")
    ml = MLModels()
    xgb_model = ml.build_xgb(n_estimators=200, max_depth=7)
    xgb_model.fit(X_train_norm, y_train)
    
    # Evaluate
    y_pred = xgb_model.predict(X_test_norm)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"[OK] XGBoost Accuracy: {accuracy:.4f}")


# ============================================================================
# EXAMPLE 3: Real-Time Inference
# ============================================================================

def example_3_real_time_inference():
    """
    Simulate real-time IMU data streaming and predictions
    Shows how to use the real-time prediction engine
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: REAL-TIME INFERENCE")
    print("="*70)
    
    from realtime_inference import RealTimeIMUPredictor, SimulatedIMUStream
    from imu_motion_recognition import IMUDataLoader, SignalPreprocessor
    import numpy as np
    
    # Load and preprocess data
    loader = IMUDataLoader()
    X_train, y_train, X_test, y_test = loader.load_data()
    
    preprocessor = SignalPreprocessor()
    X_train_norm, X_test_norm = preprocessor.normalize_signal(X_train, X_test)
    
    # Initialize predictor with pre-trained model
    print("\nInitializing Real-Time Predictor...")
    predictor = RealTimeIMUPredictor(
        model_path='./models/cnn_lstm_model.h5',
        model_type='cnn_lstm',
        window_size=128
    )
    
    # Set normalization stats
    predictor.set_normalization_stats(
        mean=np.mean(X_train_norm),
        std=np.std(X_train_norm)
    )
    
    # Create simulated stream
    activity_labels = {
        0: 'WALKING', 1: 'WALKING_UPSTAIRS', 2: 'WALKING_DOWNSTAIRS',
        3: 'SITTING', 4: 'STANDING', 5: 'LAYING'
    }
    stream = SimulatedIMUStream(X_test_norm, y_test, activity_labels)
    
    # Process stream
    print("Processing IMU stream...")
    predictions = []
    
    for i in range(min(50, len(stream.windows))):
        window, true_activity = stream.get_next_window()
        
        if window is None:
            break
        
        # Add samples one at a time (simulating real streaming)
        predictor.add_imu_batch(window)
        
        if predictor.is_buffer_ready():
            predicted_activity, confidence = predictor.predict_smoothed()
            
            is_correct = "[OK]" if predicted_activity == true_activity else "[X]"
            print(f"{is_correct} Predicted: {predicted_activity:20} | Confidence: {confidence:.2%} | True: {true_activity}")
            
            predictions.append(predicted_activity == true_activity)
            predictor.reset_buffer()
    
    accuracy = sum(predictions) / len(predictions) if predictions else 0
    print(f"\nReal-Time Accuracy: {accuracy:.2%}")


# ============================================================================
# EXAMPLE 4: Load and Use Pre-trained Models
# ============================================================================

def example_4_pretrained_models():
    """
    Load pre-trained models and make predictions
    Useful for deployment scenarios
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: LOADING PRE-TRAINED MODELS")
    print("="*70)
    
    import tensorflow as tf
    import joblib
    import numpy as np
    from imu_motion_recognition import IMUDataLoader, SignalPreprocessor
    from sklearn.metrics import accuracy_score
    
    # Load data
    loader = IMUDataLoader()
    X_train, y_train, X_test, y_test = loader.load_data()
    
    preprocessor = SignalPreprocessor()
    X_train_norm, X_test_norm = preprocessor.normalize_signal(X_train, X_test)
    
    # Load ML model (XGBoost)
    print("\nLoading XGBoost model...")
    xgb_model = joblib.load('./models/xgb_model.pkl')
    y_pred = xgb_model.predict(X_test_norm)
    acc_xgb = accuracy_score(y_test, y_pred)
    print(f"[OK] XGBoost Accuracy: {acc_xgb:.4f}")
    
    # Load DL model (CNN-LSTM)
    print("Loading CNN-LSTM model...")
    X_test_dl = X_test_norm.reshape(X_test_norm.shape[0], X_test_norm.shape[1], 1)
    cnn_lstm = tf.keras.models.load_model('./models/cnn_lstm_model.h5')
    y_pred_dl = np.argmax(cnn_lstm.predict(X_test_dl, verbose=0), axis=1)
    acc_dl = accuracy_score(y_test, y_pred_dl)
    print(f"[OK] CNN-LSTM Accuracy: {acc_dl:.4f}")


# ============================================================================
# EXAMPLE 5: Custom Data Processing
# ============================================================================

def example_5_custom_data():
    """
    Process and analyze custom IMU data
    Shows how to work with your own data
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: PROCESSING CUSTOM DATA")
    print("="*70)
    
    import numpy as np
    from imu_motion_recognition import SignalPreprocessor, FeatureExtractor
    from sklearn.preprocessing import StandardScaler
    
    # Simulate custom IMU data
    print("\nSimulating custom IMU data...")
    n_samples = 1000
    n_features = 561  # Standard UCI HAR feature count
    
    X_custom = np.random.randn(n_samples, n_features) * 50  # Random data
    y_custom = np.random.randint(0, 6, n_samples)  # 6 activities
    
    # Preprocess
    preprocessor = SignalPreprocessor()
    X_normalized = preprocessor.scaler.fit_transform(X_custom)
    
    print(f"[OK] Data shape: {X_normalized.shape}")
    print(f"  Mean: {X_normalized.mean():.4f}")
    print(f"  Std:  {X_normalized.std():.4f}")
    
    # Extract features (if working with raw signals)
    print("\nExtracting features...")
    X_features = FeatureExtractor.extract_from_dataset(
        X_normalized[:100],  # Use subset for demo
        window_size=128,
        overlap=0.5
    )
    print(f"[OK] Extracted features shape: {X_features.shape}")


# ============================================================================
# EXAMPLE 6: Visualization & Analysis
# ============================================================================

def example_6_visualization():
    """
    Generate visualizations and analysis plots
    Shows how to use the visualization engine
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: VISUALIZATION & ANALYSIS")
    print("="*70)
    
    from analysis_visualization import VisualizationEngine, ComprehensiveAnalysis
    from imu_motion_recognition import IMUDataLoader, SignalPreprocessor
    import numpy as np
    
    # Load data
    loader = IMUDataLoader()
    X_train, y_train, X_test, y_test = loader.load_data()
    
    preprocessor = SignalPreprocessor()
    X_train_norm, X_test_norm = preprocessor.normalize_signal(X_train, X_test)
    
    # Create visualization engine
    viz = VisualizationEngine('./my_visualizations')
    
    activity_labels = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 
                      'SITTING', 'STANDING', 'LAYING']
    
    # Generate plots
    print("\nGenerating visualizations...")
    viz.plot_class_distribution(y_train, y_test, activity_labels)
    viz.plot_raw_signals(X_train_norm, y_train, activity_labels, n_activities=3)
    viz.plot_frequency_analysis(X_train_norm, y_train, activity_labels)
    
    print("[OK] Visualizations saved to ./my_visualizations/")


# ============================================================================
# EXAMPLE 7: Model Comparison & Selection
# ============================================================================

def example_7_model_comparison():
    """
    Compare different models and select the best one
    Useful for choosing which model to deploy
    """
    print("\n" + "="*70)
    print("EXAMPLE 7: MODEL COMPARISON & SELECTION")
    print("="*70)
    
    from imu_motion_recognition import IMUDataLoader, SignalPreprocessor, MLModels, DLModels, ModelTrainer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score
    import numpy as np
    
    # Load and preprocess data
    loader = IMUDataLoader()
    X_train, y_train, X_test, y_test = loader.load_data()
    
    preprocessor = SignalPreprocessor()
    X_train_norm, X_test_norm = preprocessor.normalize_signal(X_train, X_test)
    
    # Compare ML models
    print("\nComparing ML Models...")
    ml = MLModels()
    ml.train_models(X_train_norm, y_train)
    
    results = {}
    for model_name in ['svm', 'xgb', 'rf']:
        y_pred = ml.predict(X_test_norm, model_name)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        results[model_name] = {'accuracy': acc, 'f1': f1}
        print(f"  {model_name.upper():15} Accuracy: {acc:.4f}  F1: {f1:.4f}")
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n[OK] Best ML Model: {best_model[0].upper()}")
    print(f"  Accuracy: {best_model[1]['accuracy']:.4f}")


# ============================================================================
# EXAMPLE 8: Hyperparameter Tuning
# ============================================================================

def example_8_hyperparameter_tuning():
    """
    Tune model hyperparameters for better performance
    Shows how to experiment with different configurations
    """
    print("\n" + "="*70)
    print("EXAMPLE 8: HYPERPARAMETER TUNING")
    print("="*70)
    
    from imu_motion_recognition import IMUDataLoader, SignalPreprocessor, MLModels
    from sklearn.metrics import accuracy_score
    import numpy as np
    
    # Load data
    loader = IMUDataLoader()
    X_train, y_train, X_test, y_test = loader.load_data()
    
    preprocessor = SignalPreprocessor()
    X_train_norm, X_test_norm = preprocessor.normalize_signal(X_train, X_test)
    
    # Test different XGBoost configurations
    print("\nTesting XGBoost Configurations...")
    ml = MLModels()
    
    configs = [
        {'n_estimators': 100, 'max_depth': 5},
        {'n_estimators': 200, 'max_depth': 7},
        {'n_estimators': 300, 'max_depth': 9},
    ]
    
    best_acc = 0
    best_config = None
    
    for config in configs:
        xgb = ml.build_xgb(**config)
        xgb.fit(X_train_norm, y_train)
        
        y_pred = xgb.predict(X_test_norm)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"  Config: {config} → Accuracy: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_config = config
    
    print(f"\n[OK] Best Configuration: {best_config}")
    print(f"  Best Accuracy: {best_acc:.4f}")


# ============================================================================
# MAIN - Run Examples
# ============================================================================

def main():
    """
    Run selected examples
    """
    print("\n")
    print("="*64)
    print("    HUMAN MOTION RECOGNITION - QUICK START EXAMPLES")
    print("="*64)
    
    examples = {
        1: ("Complete Pipeline (Recommended)", example_1_complete_pipeline),
        2: ("Individual Model Training", example_2_individual_models),
        3: ("Real-Time Inference", example_3_real_time_inference),
        4: ("Load Pre-trained Models", example_4_pretrained_models),
        5: ("Process Custom Data", example_5_custom_data),
        6: ("Visualization & Analysis", example_6_visualization),
        7: ("Model Comparison", example_7_model_comparison),
        8: ("Hyperparameter Tuning", example_8_hyperparameter_tuning),
    }
    
    print("\nAvailable Examples:")
    for key, (name, _) in examples.items():
        print(f"  [{key}] {name}")
    
    try:
        choice = input("\nSelect example (1-8) or 'all' for complete pipeline: ").strip().lower()
        
        if choice == 'all':
            print("\nRunning Complete Pipeline...")
            example_1_complete_pipeline()
        elif choice.isdigit() and 1 <= int(choice) <= len(examples):
            name, func = examples[int(choice)]
            print(f"\nRunning: {name}")
            func()
        else:
            print("Invalid choice!")
            
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
