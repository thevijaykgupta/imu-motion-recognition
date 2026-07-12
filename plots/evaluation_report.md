
# IMU Motion Recognition - Evaluation Report

## Model Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score | Inference Time (ms) |
|-------|----------|-----------|--------|----------|---------------------|
| SVM | 0.9572 | 0.9579 | 0.9572 | 0.9571 | 0.58 |
| XGBoost | 0.9382 | 0.9392 | 0.9382 | 0.9381 | 0.005 |
| Random Forest | 0.9274 | 0.9285 | 0.9274 | 0.9272 | 0.060 |
| CNN | 0.9060 | 0.9087 | 0.9060 | 0.9065 | 0.51 |
| CNN-LSTM | 0.1822 | 0.0332 | 0.1822 | 0.0562 | 3.04 |
| LSTM | 0.1822 | 0.0332 | 0.1822 | 0.0562 | 9.79 |

## Best Model: SVM
- Accuracy: 95.72%
- F1-Score (weighted): 95.71%
- F1-Score (macro): 95.63%

## Dataset
- Training samples: 7,352
- Testing samples: 2,947
- Features: 561
- Classes: 6 (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING)

## Generated Visualizations
- accuracy_comparison.png
- confusion_matrix_normalized.png
- per_class_accuracy.png
- inference_time_comparison.png
- metrics_heatmap.png
- feature_importance_rf.png
- feature_importance_xgb.png
