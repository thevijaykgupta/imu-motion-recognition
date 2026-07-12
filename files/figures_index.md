# Figures Index — Human Motion Recognition Using IMUs

All figures below are sourced directly from the supplied project archive (`plots/` and `visualizations/` folders). File paths are relative to `report_assets/figures/`.

## Dataset Figures

**Figure 12 — `class_distribution.png`**
Caption: Sample count per activity class.
Explanation: Bar chart of the number of samples belonging to each of the six activity classes.
Key observation: Classes are reasonably balanced; test-set support ranges from 420 (WALKING_DOWNSTAIRS) to 537 (LAYING) samples, confirmed numerically in `classification_report.csv`.
Why it matters: Confirms the reported accuracy figures are not artifacts of severe class imbalance.

**Figure 20 — `raw_signals.png`**
Caption: Raw accelerometer/gyroscope time-series signals per activity.
Explanation: Time-domain plots of the motion signal for representative activities.
Key observation: Dynamic activities (walking variants) show clear periodic oscillation; static activities (sitting/standing/laying) show near-flat signals distinguished mainly by gravity-axis orientation.
Why it matters: Visually motivates why static activities are harder to separate from one another than from dynamic ones (see Figure 7, Figure 13).

**Figure 14 — `frequency_analysis.png`**
Caption: Frequency-domain (FFT-style) analysis of motion signals.
Explanation: Spectral view of the sensor signal, showing energy distribution across frequency bands per activity.
Key observation: Dynamic activities show energy concentrated at gait-frequency bands; static activities show energy concentrated near zero frequency.
Why it matters: Justifies frequency-domain features as part of the 561-feature representation used for modeling.

## Dimensionality Reduction

**Figure 10 — `pca_projection.png`**
Caption: 2D PCA projection of the 561-dimensional feature space.
Explanation: Principal Component Analysis projection colored by activity class.
Key observation: Dynamic and static activity groups occupy broadly distinct regions; SITTING, STANDING, and LAYING cluster closer to one another than to any walking variant.
Why it matters: Provides a global, linear view of class separability consistent with the confusion patterns in Section 21 of the report.

**Figure 11 — `tsne_projection.png`**
Caption: 2D t-SNE projection of the 561-dimensional feature space.
Explanation: Non-linear dimensionality reduction, colored by activity class.
Key observation: Produces tighter, more distinctly separated clusters than PCA; the SITTING/STANDING boundary remains the least distinct pairing.
Why it matters: Corroborates PCA's finding with a non-linear method, strengthening confidence that SITTING/STANDING overlap is a genuine data-level property, not a linear-projection artifact.

## Feature Analysis

**Figure 17 — `feature_correlation_heatmap.png`**
Caption: Correlation heatmap across engineered features.
Explanation: Pairwise correlation matrix of (a subset of) the 561 engineered features.
Key observation: Substantial correlation exists among feature subsets, expected since many features are derived statistics of the same underlying signal axis.
Why it matters: Explains why tree-based models can achieve strong accuracy while relying on a relatively compact subset of informative features.

**Figure 18 — `feature_importance_rf.png`**
Caption: Random Forest feature importance ranking.
Explanation: Bar chart of the top contributing features by Gini/impurity-based importance for the Random Forest model.
Key observation: Top-ranked features are drawn from gravity- and body-acceleration-derived statistics.
Why it matters: Identifies which physical signal properties are most discriminative for activity classification.

**Figure 19 — `feature_importance_xgb.png`**
Caption: XGBoost feature importance ranking.
Explanation: Bar chart of the top contributing features by gain-based importance for the XGBoost model.
Key observation: Broadly overlapping top features with the Random Forest ranking (Figure 18), reinforcing that both tree ensembles converge on a similar informative feature subset.
Why it matters: Cross-validates feature importance findings across two independent tree-based models.

*Note: No SHAP plots were included in the supplied artifacts; this category is explicitly incomplete (see report Section 22, 26).*

## Model Comparison Figures

**Figure 1 — `accuracy_comparison.png`**
Caption: Accuracy comparison across all six models.
Explanation: Bar chart of test-set accuracy for SVM, XGBoost, Random Forest, CNN, LSTM, and CNN-LSTM.
Key observation: Two clear tiers — SVM/XGBoost/Random Forest/CNN cluster between 90.6%–95.7%, while LSTM and CNN-LSTM collapse to 18.22%.
Why it matters: Immediately surfaces the headline finding of the evaluation — the tiered performance split.

**Figure 21 — `model_comparison.png`**
Caption: Multi-model performance comparison (visualizations folder).
Explanation: Alternate/expanded comparison view across models.
Key observation: Confirms the same ranking as Figure 1 and the metrics heatmap (Figure 5).
Why it matters: Cross-validates the ranking using an independently generated chart.

**Figure 5 — `metrics_heatmap.png`**
Caption: Heatmap of all evaluation metrics across all models.
Explanation: Matrix view of accuracy, precision, recall, F1, MCC, Kappa, and Balanced Accuracy per model.
Key observation: SVM is uniformly the strongest across every metric column, not just accuracy; LSTM/CNN-LSTM are uniformly at or near zero on MCC/Kappa.
Why it matters: Confirms SVM's superiority is robust across multiple, differently-behaved metrics (not an accuracy-only artifact).

**Figure 6 — `model_size_comparison.png`**
Caption: Model size comparison across all six models.
Explanation: Bar chart comparing model footprint size.
Key observation: Chart-only; no accompanying numeric table was supplied, so exact size values are not restated as numbers in this report (see Section 16, 26).
Why it matters: Model size is a deployment-relevant factor alongside accuracy and inference time.

**Figure 3 — `inference_time_comparison.png`**
Caption: Per-sample inference time comparison across all six models.
Explanation: Bar chart of inference latency, cross-validated numerically against `model_metrics.csv` (Table 3 in report).
Key observation: XGBoost is fastest (≈0.005 ms/sample); LSTM is slowest (≈9.79 ms/sample).
Why it matters: Establishes the accuracy/latency trade-off — XGBoost is an attractive choice when both accuracy and speed matter.

## Per-Class & Error Analysis

**Figure 4 — `per_class_accuracy.png`**
Caption: Per-class accuracy across models.
Explanation: Grouped bar chart of classification accuracy broken down by activity class, across models.
Key observation: LAYING is consistently the easiest class across all models; SITTING is consistently one of the harder classes.
Why it matters: Localizes model weaknesses to a specific class pairing rather than describing weakness only in aggregate.

**Figure 2 — `confusion_matrix_normalized.png`**
Caption: Normalized confusion matrix (best model).
Explanation: Row-normalized confusion matrix showing the proportion of each true class predicted as each class.
Key observation: Off-diagonal mass concentrates almost entirely in the SITTING↔STANDING cell pair.
Why it matters: Directly visualizes the dominant error mode discussed in Section 21 of the report.

**Figure 13 — `confusion_matrices.png`**
Caption: Confusion matrices for all six models, side-by-side.
Explanation: Grid of per-model confusion matrices for direct visual comparison.
Key observation: The SITTING/STANDING confusion appears in every model's matrix, including the best model, confirming it is a dataset-level rather than model-specific limitation.
Why it matters: Rules out the possibility that the error pattern is specific to one algorithm's decision boundary.

**Figure 7 — `error_distribution.png`**
Caption: Distribution of misclassifications by true class.
Explanation: Bar chart of the count of misclassified samples per true activity class.
Key observation: Errors concentrate in SITTING and STANDING classes, consistent with Figures 2, 4, and 13.
Why it matters: Quantifies where error-reduction effort would have the greatest impact (Section 27, Future Work item 2).

**Figure 8 — `prediction_confidence_histogram.png`**
Caption: Histogram of prediction confidence for the best model.
Explanation: Distribution of the model's predicted-class probability/confidence values.
Key observation: Distribution is right-skewed toward high confidence, consistent with a well-separated classifier, with a secondary lower-confidence mass plausibly corresponding to the SITTING/STANDING boundary region.
Why it matters: Confidence calibration is relevant for any downstream decision-thresholding use of the model.

**Figure 16 — `classification_report_heatmap.png`**
Caption: Heatmap of precision/recall/F1 by class (best model).
Explanation: Color-coded matrix of per-class precision, recall, and F1-score.
Key observation: Mirrors Table 4 in the report (SVM per-class metrics) — LAYING highest, SITTING lowest recall.
Why it matters: Visual companion to the numeric per-class table, useful for at-a-glance review.

## Curves

**Figure 9 — `roc_curves.png`**
Caption: One-vs-Rest ROC curves per class (best model).
Explanation: ROC curve for each of the six activity classes.
Key observation: Curves for LAYING and the walking variants sit closest to the top-left (ideal) corner; SITTING/STANDING curves show relatively more overlap with each other.
Why it matters: Provides a threshold-independent view of per-class separability, complementing the fixed-threshold confusion matrix.

**Figure 15 — `precision_recall_curves.png`**
Caption: Precision-Recall curves per class (best model).
Explanation: PR curve for each activity class, more informative than ROC under class-support differences.
Key observation: Consistent with the ROC findings — SITTING/STANDING show the least favorable PR trade-off among the six classes.
Why it matters: PR curves are a more sensitive diagnostic than ROC/accuracy alone for the harder class pair.

## Training

**Figure 22 — `training_history_accuracy.png`**
Caption: Accuracy vs. training epoch for the deep learning model(s).
Explanation: Line plot of accuracy over training epochs.
Key observation: Shows an upward training-accuracy trend; given the deep learning models' final test accuracy of 18.22% (LSTM/CNN-LSTM), this curve most likely reflects training-set (not held-out test) performance and should not be read as evidence of generalization.
Why it matters: Highlights the train/test generalization gap that is the central limitation of the deep learning track in this run.

*Note: No training/validation loss curve was included in the supplied artifacts — only accuracy vs. epoch. This is explicitly flagged as missing (see report Section 15, 26).*

---

**Total figures indexed: 22** (matches the count of PNG files supplied in `plots/` and `visualizations/`).
