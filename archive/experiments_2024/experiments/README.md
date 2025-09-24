# XGBoost Ranker Experiments

This folder contains stepwise application of fixes/enhancements from MODEL_FIXES.md.

Workflow:
- step0_baseline: unmodified baseline scripts.
- stepN_*: apply exactly one change group, then evaluate.

How to test quickly:
1) Activate env and run: python train_xgboost_ranking_model.py --start-year 2022 --end-year 2022 --trials 0
2) Evaluate OOS on 2025: python evaluate_model.py --model-file model_output/xgboost_ranker_2022_2022_*joblib --eval-data-file year_2025_data.csv --config-file config.yaml --sharpe-edges-file model_output/sharpe_qcut_edges_2022_2022_*.pkl --feature-list-file model_output/xgb_feature_names_2022_2022_*.pkl

After each step, record metrics and notes here.
