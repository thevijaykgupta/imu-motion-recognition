# IMU Motion Recognition - Final Evaluation Report

## Project Overview
Human Activity Recognition using IMU data from the UCI HAR Dataset.
Training samples: 7,352 | Testing samples: 2,947 | Features: 561 | Classes: 6

---

## Model Performance Summary

| Model | Accuracy | F1-Score (Weighted) | F1-Score (Macro) | MCC | Cohen's Kappa | Balanced Accuracy |
|-------|----------|---------------------|------------------|-----|---------------|-------------------|
| SVM | 0.9572 | 0.9571 | 0.9563 | 0.9488 | 0.9486 | 0.9554 |
| Random Forest | 0.9274 | 0.9272 | 0.9255 | 0.9129 | 0.9127 | 0.9243 |
| XGBoost | 0.9382 | 0.9381 | 0.9376 | 0.9260 | 0.9258 | 0.9367 |
| CNN | 0.9060 | 0.9065 | 0.9051 | 0.8873 | 0.8870 | 0.9035 |
| LSTM | 0.1822 | 0.0562 | 0.0514 | 0.0000 | 0.0000 | 0.1667 |
| CNN-LSTM | 0.1822 | 0.0562 | 0.0514 | 0.0000 | 0.0000 | 0.1667 |

**Best Model: SVM (Accuracy: 95.72%)**

---

## Generated Figures

### Performance Metrics
- accuracy_comparison.png
- confusion_matrix_normalized.png  
- per_class_accuracy.png
- inference_time_comparison.png
- model_size_comparison.png
- metrics_heatmap.png
- prediction_confidence_histogram.png
- error_distribution.png

### Curves & Analysis
- roc_curves.png
- precision_recall_curves.png

### Dimensionality Reduction
- pca_projection.png
- tsne_projection.png

### Feature Analysis
- feature_correlation_heatmap.png
- feature_importance_rf.png
- feature_importance_xgb.png

---

## Generated Reports

### CSV Files
- evaluation_summary.csv
- classification_report.csv

### Excel
- metrics_summary.xlsx

### Markdown
- evaluation_report.md (this file)

---

## Notes
- ML models (SVM, RF, XGBoost) trained on 561 features achieve high accuracy
- DL models (CNN, LSTM, CNN-LSTM) trained on reshaped data have lower accuracy
- For production, consider retraining DL models with proper sequence data

---

*Generated for publication and GitHub presentation*
