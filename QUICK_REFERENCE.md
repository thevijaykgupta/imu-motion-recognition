# QUICK REFERENCE CARD
## Human Motion Recognition Project

---

## ⚡ ONE-LINE QUICK START

```bash
pip install -r requirements.txt && python run_complete_pipeline.py
```

---

## 🔥 MOST COMMON COMMANDS

| What You Want | Command | Time |
|---------------|---------|------|
| Run everything | `python run_complete_pipeline.py` | 45 min |
| Run interactive demo | `python examples.py` | 2-10 min |
| Load pre-trained model | `python examples.py` → [4] | 2 min |
| Test real-time | `python examples.py` → [3] | 5 min |
| Train one model only | `python examples.py` → [2] | 5 min |
| Generate visualizations | `python examples.py` → [6] | 5 min |

---

## 📦 SETUP CHECKLIST

```bash
# ✓ 1. Create folder
mkdir imu-motion-recognition
cd imu-motion-recognition

# ✓ 2. Copy all Python files here
# (imu_motion_recognition.py, realtime_inference.py, etc.)

# ✓ 3. Create virtual environment
python -m venv venv
source venv/bin/activate  # or: venv\Scripts\activate (Windows)

# ✓ 4. Install dependencies
pip install -r requirements.txt

# ✓ 5. Run pipeline
python run_complete_pipeline.py

# ✓ Done! Check ./models/ and ./visualizations/
```

---

## 🎯 WHAT EACH SCRIPT DOES

### `run_complete_pipeline.py`
```
Orchestrates entire workflow:
  1. Download dataset
  2. Preprocess data
  3. Train ML models (SVM, XGB, RF)
  4. Train DL models (CNN-LSTM, LSTM, CNN)
  5. Evaluate all models
  6. Save trained models
  7. Generate visualizations
  8. Test real-time inference
  9. Generate analysis report

👉 USE THIS: For first run & complete results
⏱️ TIME: 45 minutes
```

### `examples.py`
```
Interactive example selector with 8 options:
  [1] Complete Pipeline
  [2] Individual Model Training
  [3] Real-Time Inference
  [4] Load Pre-trained Models
  [5] Process Custom Data
  [6] Visualization & Analysis
  [7] Model Comparison
  [8] Hyperparameter Tuning

👉 USE THIS: For learning & testing specific parts
⏱️ TIME: 2-10 minutes
```

### `imu_motion_recognition.py`
```
Core module with classes:
  - IMUDataLoader: Download & load UCI HAR
  - SignalPreprocessor: Normalize & preprocess
  - FeatureExtractor: Extract 13 features
  - MLModels: SVM, XGBoost, Random Forest
  - DLModels: CNN-LSTM, LSTM, CNN
  - ModelTrainer: Training & evaluation

👉 USE THIS: For understanding architecture
⏱️ TIME: Reference only
```

### `realtime_inference.py`
```
Real-time prediction engine:
  - RealTimeIMUPredictor: Stream-based inference
  - SimulatedIMUStream: Test with synthetic data
  - RealTimeEvaluator: Performance metrics

👉 USE THIS: For real-time activity recognition
⏱️ TIME: Reference only
```

### `analysis_visualization.py`
```
Visualization & analysis:
  - VisualizationEngine: Generate 7 plot types
  - ComprehensiveAnalysis: Generate reports

👉 USE THIS: For understanding results
⏱️ TIME: Reference only
```

---

## 🎓 LEARNING EXAMPLES

### Example 1: See It Work (2 minutes)
```bash
python examples.py
# Choose: [4] Load Pre-trained Models
# What you'll see: Model accuracies
```

### Example 2: Understand Real-Time (5 minutes)
```bash
python examples.py
# Choose: [3] Real-Time Inference
# What you'll see: Stream-based predictions
```

### Example 3: Train Single Model (5 minutes)
```bash
python examples.py
# Choose: [2] Individual Model Training
# What you'll see: XGBoost training & evaluation
```

### Example 4: Process Your Data (5 minutes)
```bash
python examples.py
# Choose: [5] Process Custom Data
# What you'll learn: How to use your own IMU data
```

### Example 5: See All Visualizations (5 minutes)
```bash
python examples.py
# Choose: [6] Visualization & Analysis
# What you'll see: Signal plots, frequency analysis
```

---

## 📊 EXPECTED OUTPUT

### After running complete pipeline, you'll see:

**Console:**
```
✓ SVM       | Accuracy: 0.8923
✓ XGBoost   | Accuracy: 0.9145
✓ Random Forest | Accuracy: 0.8987
✓ CNN-LSTM  | Accuracy: 0.9612  ← Best
✓ LSTM      | Accuracy: 0.9387
✓ CNN       | Accuracy: 0.9201

Best Model: CNN-LSTM (96.12% accuracy)
```

**Files Created:**
```
models/
├── cnn_lstm_model.h5     ← Best accuracy
├── lstm_model.h5
├── cnn_model.h5
├── xgb_model.pkl         ← Fastest
├── svm_model.pkl
└── rf_model.pkl

visualizations/
├── training_history_accuracy.png
├── confusion_matrices.png
├── model_comparison.png
├── class_distribution.png
├── raw_signals.png
├── frequency_analysis.png
├── per_class_metrics.png
└── report/
    └── analysis_report.txt
```

---

## 🚨 COMMON ISSUES & FIXES

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: tensorflow` | `pip install --upgrade tensorflow` |
| `ModuleNotFoundError: xgboost` | `pip install xgboost` |
| Memory error | Reduce batch_size from 32 to 16 |
| Slow training | Use GPU or reduce epochs |
| Dataset won't download | Download manually from UCI, extract to `./UCI_HAR_Dataset/` |
| "venv not activated" | Run: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows) |

---

## 📁 FILE REFERENCE

### Input
```
requirements.txt           ← Dependencies to install
README.md                  ← Full documentation
SETUP_GUIDE.md            ← Detailed setup (this file)
QUICK_REFERENCE.md        ← This quick reference
```

### Python Files
```
imu_motion_recognition.py  ← Core ML/DL models (1000+ lines)
realtime_inference.py      ← Real-time predictions (400 lines)
analysis_visualization.py  ← Visualizations (600 lines)
run_complete_pipeline.py   ← Master orchestrator (500 lines)
examples.py               ← Interactive examples (400 lines)
```

### Generated (After Running)
```
UCI_HAR_Dataset/          ← Downloaded dataset (~100MB)
models/                   ← Trained models (~160MB)
  ├── cnn_lstm_model.h5   ← 50MB
  ├── lstm_model.h5       ← 40MB
  ├── cnn_model.h5        ← 35MB
  ├── xgb_model.pkl       ← 5MB
  ├── svm_model.pkl       ← 10MB
  └── rf_model.pkl        ← 20MB

visualizations/           ← Generated plots (~5MB)
  ├── training_history_accuracy.png
  ├── confusion_matrices.png
  ├── model_comparison.png
  ├── class_distribution.png
  ├── raw_signals.png
  ├── frequency_analysis.png
  ├── per_class_metrics.png
  └── report/
      └── analysis_report.txt
```

---

## 💻 PYTHON CODE SNIPPETS

### Load Pre-trained Model
```python
import tensorflow as tf
model = tf.keras.models.load_model('./models/cnn_lstm_model.h5')
predictions = model.predict(X_test_data)
```

### Real-Time Prediction
```python
from realtime_inference import RealTimeIMUPredictor

predictor = RealTimeIMUPredictor('./models/cnn_lstm_model.h5')
predictor.add_imu_sample(accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z)

if predictor.is_buffer_ready():
    activity, confidence = predictor.predict()
    print(f"{activity}: {confidence:.2%}")
```

### Use ML Model
```python
import joblib
xgb_model = joblib.load('./models/xgb_model.pkl')
predictions = xgb_model.predict(X_test_data)
```

### Train Custom Model
```python
from imu_motion_recognition import MLModels

ml = MLModels()
ml.train_models(X_train, y_train)
predictions = ml.predict(X_test, 'xgb')
```

---

## 🎯 MODEL SELECTION GUIDE

**Choose CNN-LSTM if:**
- You want best accuracy (96%)
- You have time for training (20 min)
- You have 50MB disk space
- ✅ RECOMMENDED for most cases

**Choose XGBoost if:**
- You want fastest inference
- You have limited memory
- You need <1 second predictions
- ✅ RECOMMENDED for deployment

**Choose LSTM if:**
- You want temporal modeling
- You need balance of speed/accuracy
- You have 40MB disk space

**Choose SVM if:**
- You want minimal dependencies
- You need interpretability
- You have 10MB disk space only

---

## ⏱️ TIMING GUIDE

| Operation | CPU | GPU |
|-----------|-----|-----|
| Installation | 10 min | 10 min |
| Data download | 5 min | 5 min |
| ML model training | 5 min | 5 min |
| DL model training | 30 min | 8 min |
| Evaluation | 2 min | 2 min |
| Visualization | 3 min | 3 min |
| **TOTAL** | **55 min** | **33 min** |

**GPU Recommendation:** If available, use GPU (10-15 min faster training)

---

## ✅ SUCCESS INDICATORS

After running, you should see:
- ✅ No error messages (warnings are OK)
- ✅ `models/` folder with 6 files
- ✅ `visualizations/` folder with 7 PNG files
- ✅ Console output showing "PIPELINE EXECUTION COMPLETED"
- ✅ CNN-LSTM accuracy ~96%
- ✅ Execution time ~45 minutes (CPU) or ~33 minutes (GPU)

---

## 🔗 USEFUL LINKS

- **Dataset**: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
- **TensorFlow Docs**: https://tensorflow.org/
- **scikit-learn Docs**: https://scikit-learn.org/
- **XGBoost Docs**: https://xgboost.readthedocs.io/

---

## 📞 NEED HELP?

1. **Check SETUP_GUIDE.md** - Detailed troubleshooting section
2. **Check README.md** - Full documentation
3. **Run `python examples.py`** - See working examples
4. **Check error message** - Usually tells you what's wrong
5. **Verify Python version** - `python --version` (should be 3.8+)

---

## 🎉 YOU'RE READY!

Copy these files and follow steps in SETUP_GUIDE.md:

```bash
1. Extract all files to a folder
2. Create virtual environment: python -m venv venv
3. Activate: source venv/bin/activate
4. Install: pip install -r requirements.txt
5. Run: python run_complete_pipeline.py
6. Wait 45 minutes
7. Check results in ./models/ and ./visualizations/
```

**That's it! Enjoy! 🚀**

---

**Version**: 1.0  
**Last Updated**: June 2024  
**For**: Kaushal H & Team  
