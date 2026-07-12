# Tables Index — Human Motion Recognition Using IMUs

All tables are reproduced in the main report and also saved as standalone CSVs in `report_assets/tables/`.

**Table 1 — Dataset Summary**
File: `report_assets/tables/dataset_summary.csv`
Source: `evaluation_report.md`, `final_evaluation_report.md`, `classification_report.csv` (class support totals).

**Table 2 — Full Model Comparison (8 metrics × 6 models)**
File: `report_assets/tables/model_comparison_metrics.csv`
Source: `evaluation_summary.csv`, `final_evaluation_report.md`.
Columns: model, accuracy, precision_weighted, recall_weighted, f1_weighted, f1_macro, mcc, kappa, balanced_accuracy.

**Table 3 — Inference Time Comparison**
File: `report_assets/tables/inference_time_raw.csv`
Source: `model_metrics.csv` (raw seconds), cross-validated against `evaluation_report.md` (milliseconds).

**Table 4 — Best Model (SVM) Per-Class Performance**
File: `report_assets/tables/per_class_performance.csv` (filtered to Model = SVM)
Source: `classification_report.csv`.

**Table 5 — Cross-Model Per-Class Performance (all 3 ML models × 6 classes)**
File: `report_assets/tables/per_class_performance.csv` (full file — SVM, XGBoost, Random Forest)
Source: `classification_report.csv`.
Note: Per-class metrics are only available in the supplied artifacts for the three classical ML models (SVM, XGBoost, Random Forest). No per-class breakdown was supplied for CNN, LSTM, or CNN-LSTM.

**Excel Workbook — Metrics Summary**
File: `report_assets/csv/metrics_summary.xlsx`
Contents: Single "Summary" sheet with model, accuracy, f1_weighted for SVM, XGBoost, Random Forest (subset of Table 2).

---

## Output Files Generated (this report package)

| File | Description |
|---|---|
| `Comprehensive_Project_Report.md` | Full narrative report (Markdown) |
| `Comprehensive_Project_Report.docx` | Full narrative report (Word) |
| `Comprehensive_Project_Report.pdf` | Full narrative report (PDF) |
| `figures_index.md` | Captioned index of all 22 figures |
| `tables_index.md` | This file — index of all tables and source CSVs |
| `report_assets/figures/` | All 22 original PNG figures from the project archive |
| `report_assets/tables/` | Derived summary CSVs (dataset summary, per-class performance, full metrics, inference time) |
| `report_assets/csv/` | Original CSV/XLSX metric files as supplied |
| `report_assets/appendix/` | Original `evaluation_report.md` and `final_evaluation_report.md` as supplied |
| `report_assets/models/` | Empty — no saved model files were included in the supplied project archive |
