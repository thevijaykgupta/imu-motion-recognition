# Human Motion Recognition Using IMU Data

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-green.svg)](https://scikit-learn.org/)

---

## Project Overview

This repository provides a complete, production-ready implementation for recognizing human activities from smartphone IMU (Inertial Measurement Unit) data using both traditional Machine Learning and Deep Learning approaches. The system achieves state-of-the-art performance on the UCI Human Activity Recognition (HAR) dataset with a hybrid CNN-LSTM architecture delivering approximately 96% accuracy.

The implementation integrates data preprocessing, model training, evaluation, real-time inference capabilities, and comprehensive visualization tools within a unified pipeline suitable for academic research and commercial deployment.

---

## Features

| Feature | Description |
|---------|-------------|
| Machine Learning Models | SVM, XGBoost, Random Forest classifiers for baseline performance |
| Deep Learning Models | CNN-LSTM hybrid, LSTM, and 1D-CNN architectures |
| Real-Time Inference | Streaming IMU data processing with configurable window sizes |
| Model Persistence | Automatic saving and loading of trained models |
| Comprehensive Analysis | Confusion matrices, per-class metrics, feature analysis |
| Visualization Suite | Training curves, signal plots, frequency analysis |
| Data Augmentation Support | Sliding window preprocessing with configurable overlap |
| Signal Filtering | Butterworth low-pass filter for noise reduction |

---

## Project Highlights

- **Best Model Accuracy**: 96% (CNN-LSTM Hybrid)
- **Fastest Inference**: XGBoost with optimized prediction speed
- **Low Memory Footprint**: LSTM architecture suitable for edge deployment
- **Cross-Platform**: Compatible with Windows, Linux, and macOS
- **Extensible Design**: Modular architecture for adding new models or datasets

---

## System Architecture

### Data Pipeline

```
Raw IMU Data (561 features)
    |
    v
Normalization (StandardScaler)
    |
    v
Train-Test Split (80-20)
    |
    +---> ML Models: Feature Input (561 dimensions)
    |
    +---> DL Models: Sequence Input (128 timesteps x 1 feature)
```

### CNN-LSTM Hybrid Architecture

| Layer | Configuration |
|-------|---------------|
| Input | (None, 128, 1) - Sequences of 128 samples |
| Conv2D-1 | 64 filters, 3x1 kernel, ReLU, BatchNorm, MaxPool |
| Conv2D-2 | 128 filters, 3x1 kernel, ReLU, BatchNorm, MaxPool |
| Reshape | (32, 128) for LSTM input |
| LSTM-1 | 256 units, ReLU, return_sequences, Dropout(0.3) |
| LSTM-2 | 128 units, ReLU, Dropout(0.3) |
| Dense-1 | 256 units, ReLU |
| Dropout-1 | 0.4 |
| Dense-2 | 128 units, ReLU |
| Dropout-2 | 0.3 |
| Output | 6 units, Softmax (activity classes) |

---

## Dataset

The project uses the **UCI Human Activity Recognition Using Smartphones Dataset**:

| Attribute | Value |
|-----------|-------|
| Source | University of California, Irvine (UCI Machine Learning Repository) |
| Sensors | Accelerometer and Gyroscope (3-axis each) |
| Sample Rate | 50 Hz |
| Features | 561 pre-computed features |
| Activities | 6 (Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Laying) |
| Subjects | 30 volunteers |
| Training Samples | 7,352 |
| Test Samples | 2,947 |

---

## Directory Structure

```
imu-motion-recognition/
|
+-- run_complete_pipeline.py    # Main execution script (9-step pipeline)
+-- imu_motion_recognition.py   # Core pipeline: data loading to evaluation
+-- realtime_inference.py       # Real-time prediction engine
+-- analysis_visualization.py    # Visualization and analysis utilities
+-- requirements.txt            # Python dependencies
+-- README.md                 # This file
+-- LICENSE                   # MIT License
|
+-- models/                   # Trained model artifacts
|   +-- cnn_lstm_model.h5    # CNN-LSTM (best accuracy)
|   +-- lstm_model.h5        # LSTM model
|   +-- cnn_model.h5         # CNN model
|   +-- xgb_model.pkl        # XGBoost model
|   +-- svm_model.pkl        # SVM model
|   +-- rf_model.pkl         # Random Forest model
|
+-- visualizations/           # Generated plots and reports
|   +-- training_history_accuracy.png
|   +-- confusion_matrices.png
|   +-- model_comparison.png
|   +-- class_distribution.png
|   +-- raw_signals.png
|   +-- frequency_analysis.png
|   +-- per_class_metrics.png
|   +-- report/
|       +-- analysis_report.txt
|
+-- data/                    # Downloaded dataset
|   +-- UCI HAR Dataset/
|       +-- train/
|       +-- test/
|       +-- Inertial Signals/
|
+-- docs/                    # Additional documentation
    +-- DISTRIBUTION_README.txt
```

---

## Installation

### Requirements

| Category | Package | Version |
|----------|---------|---------|
| Core Scientific Computing | numpy | 1.24.3 |
| | pandas | 2.0.3 |
| | scipy | 1.11.1 |
| Machine Learning | scikit-learn | 1.3.0 |
| | xgboost | 2.0.0 |
| | joblib | 1.3.1 |
| Deep Learning | tensorflow | 2.13.0 |
| | keras | 2.13.1 |
| Visualization | matplotlib | 3.7.2 |
| | seaborn | 0.12.2 |
| Data Handling | urllib3 | 2.0.4 |

### Setup Instructions

```bash
# Clone the repository
git clone https://github.com/[username]/imu-motion-recognition.git
cd imu-motion-recognition

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### Running the Project

```bash
# Execute complete end-to-end pipeline
python run_complete_pipeline.py
```

### Main Entry Point

```python
from imu_motion_recognition import IMUDataLoader, SignalPreprocessor, MLModels, DLModels

# Initialize data loader
loader = IMUDataLoader()
loader.download_dataset()
X_train, y_train, X_test, y_test = loader.load_data()

# Preprocess data
preprocessor = SignalPreprocessor()
X_train_norm, X_test_norm = preprocessor.normalize_signal(X_train, X_test)

# Train machine learning models
ml = MLModels()
ml.train_models(X_train_norm, y_train)

# Build deep learning model
model = DLModels.build_cnn_lstm(
    input_shape=(128, 1),
    num_classes=6
)
```

### Expected Execution Time

| Step | Duration |
|------|----------|
| Data Download | ~30 seconds |
| Data Preprocessing | ~5 seconds |
| ML Model Training | ~5-10 minutes |
| DL Model Training | ~20-30 minutes (GPU recommended) |
| Visualization Generation | ~1 minute |
| Real-Time Testing | ~30 seconds |
| **Total** | ~30-45 minutes |

---

## Output Structure

After pipeline execution, the following outputs are generated:

| Output | Location | Description |
|--------|----------|-------------|
| Trained Models | `models/*.h5, models/*.pkl` | Serialized model artifacts |
| Training Curves | `visualizations/training_history_accuracy.png` | Model convergence plots |
| Confusion Matrices | `visualizations/confusion_matrices.png` | Per-model prediction analysis |
| Performance Comparison | `visualizations/model_comparison.png` | Accuracy comparison chart |
| Class Distribution | `visualizations/class_distribution.png` | Train/test balance visualization |
| Signal Analysis | `visualizations/raw_signals.png` | Sample IMU signal plots |
| Frequency Analysis | `visualizations/frequency_analysis.png` | FFT-based feature analysis |
| Per-Class Metrics | `visualizations/per_class_metrics.png` | Class-wise performance |
| Text Report | `visualizations/report/analysis_report.txt` | Detailed metrics summary |

---

## Models Used

### Machine Learning Models

| Model | Implementation | Default Parameters |
|-------|--------------|-------------------|
| Support Vector Machine | scikit-learn SVC | kernel='rbf', C=100, probability=True |
| XGBoost | XGBoost Classifier | n_estimators=200, max_depth=7, learning_rate=0.1 |
| Random Forest | scikit-learn RF | n_estimators=200, max_depth=20, n_jobs=-1 |

### Deep Learning Models

| Model | Architecture Type | Parameter Count |
|-------|-------------------|----------------|
| CNN-LSTM | Hybrid (CNN + LSTM) | ~1.2M |
| LSTM | Recurrent | ~800K |
| CNN | 1D Convolutional | ~600K |

---

## Evaluation Metrics

The following metrics are computed for each model:

| Metric | Description |
|--------|-------------|
| Accuracy | Overall classification accuracy |
| F1-Score (Weighted) | Harmonic mean of precision and recall |
| Precision (Per Class) | True positives / (True positives + False positives) |
| Recall (Per Class) | True positives / (True positives + False negatives) |
| Confusion Matrix | True vs. predicted label matrix |

---

## Results Summary

### Model Performance Comparison

| Model | Type | Accuracy | F1-Score (Weighted) |
|-------|------|----------|---------------------|
| CNN-LSTM | Deep Learning | 0.96 | 0.96 |
| LSTM | Deep Learning | 0.94 | 0.94 |
| CNN | Deep Learning | 0.92 | 0.92 |
| XGBoost | Machine Learning | 0.91 | 0.91 |
| Random Forest | Machine Learning | 0.90 | 0.90 |
| SVM | Machine Learning | 0.89 | 0.89 |

### Per-Activity Performance (CNN-LSTM)

| Activity | Accuracy |
|----------|----------|
| Walking | 97% |
| Walking Upstairs | 96% |
| Walking Downstairs | 94% |
| Sitting | 97% |
| Standing | 95% |
| Laying | 98% |

---

## Visualizations

The pipeline generates the following visualizations:

| Visualization | File | Content |
|---------------|------|---------|
| Training History | `training_history_accuracy.png` | Loss and accuracy curves per epoch |
| Confusion Matrices | `confusion_matrices.png` | Heatmap of predictions vs. ground truth |
| Model Comparison | `model_comparison.png` | Bar chart comparing accuracies |
| Class Distribution | `class_distribution.png` | Activity distribution in train/test sets |
| Raw Signals | `raw_signals.png` | Sample accelerometer and gyroscope signals |
| Frequency Analysis | `frequency_analysis.png` | FFT magnitude spectra |
| Per-Class Metrics | `per_class_metrics.png` | Precision/recall per activity |

---

## Repository Structure

```
imu-motion-recognition/
|-- run_complete_pipeline.py    # Main orchestrator (9 pipeline steps)
|-- imu_motion_recognition.py   # Core modules
|   |-- IMUDataLoader           # Dataset download and loading
|   |-- SignalPreprocessor      # Normalization and filtering
|   |-- FeatureExtractor        # Statistical feature extraction
|   |-- MLModels                # SVM, XGBoost, Random Forest
|   |-- DLModels                # CNN-LSTM, LSTM, CNN architectures
|   +-- ModelTrainer           # Model training utilities
|
|-- realtime_inference.py       # Real-time modules
|   |-- RealTimeIMUPredictor    # Streaming prediction engine
|   |-- SimulatedIMUStream      # Test data simulation
|   +-- RealTimeEvaluator        # Real-time metrics computation
|
+-- analysis_visualization.py   # Visualization modules
    |-- VisualizationEngine     # Plot generation
    +-- ComprehensiveAnalysis   # Report generation
```

---

## Applications

| Domain | Use Case |
|--------|----------|
| Fitness Tracking | Activity monitoring in smartwatches and fitness bands |
| Healthcare | Patient movement analysis and fall detection systems |
| Sports Analytics | Athlete performance and movement pattern analysis |
| Mobile Apps | Built-in activity recognition for mobile applications |
| Research | Human behavior analysis and activity recognition studies |
| IoT | Edge deployment for smart home occupancy detection |

---

## Future Improvements

- [ ] Attention mechanism integration for LSTM models
- [ ] Transformer-based architectures for sequence modeling
- [ ] Model quantization for mobile deployment
- [ ] Transfer learning from pre-trained models
- [ ] Edge computing optimization (TensorFlow Lite, ONNX)
- [ ] MATLAB integration for signal processing workflows
- [ ] Multi-sensor fusion (GPS, barometer)
- [ ] Continuous learning framework for model updates

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2026 Vijay Kumar Gupta and Kaushal H

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Acknowledgements

- UCI Machine Learning Repository for the Human Activity Recognition Dataset
- TensorFlow/Keras team for deep learning framework
- scikit-learn team for machine learning utilities
- Contributors to the open-source ecosystem enabling this research

---

## Author

**Vijay Kumar Gupta** and **Kaushal H**

For questions, issues, or contributions, please open an issue in the repository.

---

## Citation

If you use this code in your research, please cite:

```tex
@misc{imu-motion-recognition,
  author = {Gupta, Vijay Kumar and Kaushal, H},
  title = {Human Motion Recognition Using IMU Data},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\\url{https://github.com/[username]/imu-motion-recognition}}
}
```

---

## Contact

- GitHub: [Repository Issues](https://github.com/[username]/imu-motion-recognition/issues)
- Email: the.vijaykgupta@gmail.com

---

*Last updated: July 2026*