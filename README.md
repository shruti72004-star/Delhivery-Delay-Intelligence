# Delhivery Last-Mile Delay Intelligence Pipeline

A reproducible synthetic-data capstone covering data preparation, delivery-time regression, delay classification, hub clustering and batch scoring.

## How to Run

Use Python 3.11 and the pinned dependencies:

```bash
python -m venv venv
# Windows PowerShell: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python generate_data.py
python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
python part_d_segments.py
python package_model.py
python predict_delay.py
```

`package_model.py` regenerates `models/delay_model.pkl`. A scikit-learn pickle should be loaded with the same scikit-learn version used to create it.

## Data Findings

After running Part A, use `outputs/part_a_checks.csv`, `outputs/part_a_delay_by_group.csv` and `outputs/part_a_correlations.csv` to report measured findings. Do not copy reference values in place of your own output.

## Model Results (Regression)

Use `outputs/part_b_metrics.csv`, `part_b_tree_depth.csv`, `part_b_gb_learning_rate.csv` and `part_b_cv.csv`. Report the validation winner, its RMSE, baseline R², depth behaviour, learning-rate comparison and final test score.

## Model Results (Classification)

Use the Part C CSV files to report baseline limitations, feature rankings, feature-count comparison, tuning, threshold trade-off and final test precision/recall/F1.

## Operations Recommendation

Use `outputs/segment_profile.csv`, `outputs/tier_vs_segment_mismatch.csv`, `outputs/part_c_final_test.csv` and `outputs/new_shipment_actions.csv`. Summarize Escalate hubs, mismatches, threshold performance and the actions generated for the 48 new shipments. K=3 is retained because the business requires three playbooks and K=4/5 create small clusters.

## Production Plan

Map repository components to reproducibility, consistency, maintainability, reliability, performance and monitorability. Propose numeric monitoring triggers, a retraining cadence and sale-event comparisons against comparable sale periods rather than ordinary weeks.

## Limitations

- The dataset is synthetic and does not represent real Delhivery shipments.
- The test set contains a limited number of delayed parcels.
- The alert threshold is selected from one validation split.
- Real traffic, weather severity, hub staffing and other operational variables may be absent.
