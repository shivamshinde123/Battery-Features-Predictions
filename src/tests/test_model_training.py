import os
import json

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))

def test_check_xgb_model_fit():

    xgb_metrics_file_path = os.path.join(root_dir, 'Metrics', 'xgb_metrics.json')

    with open(xgb_metrics_file_path, 'r') as f:
        data = json.load(f)

    assert data['r2_adjusted_score_train'] >= 0.85, "Underfit: Train R² adjusted score is below threshold"
    assert data['r2_adjusted_score_test'] >= 0.85, "Underfit: Test R² adjusted score is below threshold"
    assert data['r2_adjusted_score_val'] >= 0.85, "Underfit: Validation R² adjusted score is below threshold"

    assert abs(data['r2_adjusted_score_train'] - data['r2_adjusted_score_test']) <= 0.5, "Overfit: Train vs Test R² adjusted score gap too high"
    assert abs(data['r2_adjusted_score_train'] - data['r2_adjusted_score_val']) <= 0.5, "Overfit: Train vs Validation R² adjusted score gap too high"


def test_check_nn_model_fit():

    nn_metrics_file_path = os.path.join(root_dir, 'Metrics', 'nn_metrics.json')

    with open(nn_metrics_file_path, 'r') as f:
        data = json.load(f)

    assert data['r2_adj_train'] >= 0.85, "Underfit: Train R² adjusted score is below threshold"
    assert data['r2_adj_test'] >= 0.85, "Underfit: Test R² adjusted score is below threshold"
    assert data['r2_adj_val'] >= 0.85, "Underfit: Validation R² adjusted score is below threshold"

    assert abs(data['r2_adj_train'] - data['r2_adj_test']) <= 0.5, "Overfit: Train vs Test R² adjusted score gap too high"
    assert abs(data['r2_adj_train'] - data['r2_adj_val']) <= 0.5, "Overfit: Train vs Validation R² adjusted score gap too high"


