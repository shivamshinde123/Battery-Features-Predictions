
# 🔋Battery State of Charge (SoC) Prediction 🔋

A comprehensive machine learning pipeline for **predicting and analyzing battery state of charge (SoC)**. This project leverages modern ML and experiment tracking workflows to extract insights from raw battery data, build predictive models, and ensure robust, reproducible results.

## Detailed Documentation

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/shivamshinde123/Battery-Features-Predictions)

## Overview

This repository covers a full ML workflow:
- Organizing, preprocessing, and validating raw battery datasets
- Engineering advanced features for modeling
- Training and evaluating models (Neural Networks, XGBoost, etc.)
- Experiment tracking using DVC and MLflow
- Automated testing and CI with GitHub Actions

The project is modular, scalable, and ready for collaborative or production-level work.

## Project Flow

https://gitdiagram.com/shivamshinde123/Battery-Features-Predictions

## Folder Structure

```
Battery-Feature-Project/
├── .dvc/                # DVC cache and metadata
├── .github/
│   └── workflows/
│       └── ci.yml       # GitHub Actions CI pipeline
├── Data/                # Raw and processed input data
├── Logs/                # Training logs and outputs
├── Metrics/             # Model evaluation metrics (JSON/CSV/etc.)
├── mlruns/              # MLflow experiment runs tracking directory
├── Models/              # Saved/trained models
├── Notebooks/           # Jupyter notebooks for exploration
├── src/
│   ├── frontend/        # Frontend code
│   ├── tests/           # Automated tests (pytest syntax)
│   │   ├── __init__.py
│   │   ├── test_data_validation.py
│   │   ├── test_loader.py
│   │   ├── test_model_training.py
│   │   └── test_preprocessing.py
│   ├── __init__.py
│   ├── loader.py        # Data loading utilities
│   ├── model_training_nn.py     # Neural Network models
│   ├── model_training_xgboost.py # XGBoost models
│   ├── preprocessing.py # Data cleaning & feature engineering
│   ├── utils.py         # Helper utilities
│   └── validation.py    # Data validation utilities
├── .dvcignore
├── .gitignore
├── dvc.lock
├── dvc.yaml
├── params.yaml          # Model and pipeline parameters
├── requirements.txt     # Python dependencies
```

## Getting Started

### Prerequisites

- Python 3.10+
- [DVC](https://dvc.org/)
- [Git](https://git-scm.com/)
- [MLflow](https://mlflow.org/) (recommended for experiment tracking)

### Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/Battery-Feature-Project.git
    cd Battery-Feature-Project
    ```

2. **Install Python requirements:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

- **Jupyter Notebooks:**  
  Explore example workflows in `Notebooks/`.  
  Launch with:
  ```bash
  jupyter notebook Notebooks/
  ```
- **Running the Pipeline with DVC:**  
  To run the full pipeline as defined in `dvc.yaml`:
  ```bash
  dvc repro
  ```

- **Training Custom Models:**  
  Use scripts in `src/` (e.g., `model_training_nn.py`, `model_training_xgboost.py`) to train or retrain models.

- **Experiment Tracking:**  
  MLflow tracking directories can be found in `mlruns/`.

- **Logs & Metrics:**  
  All logs are saved in `Logs/`, model files in `Models/`, and evaluation metrics in `Metrics/`.

## Project Workflow

1. **Data Collection & Validation:**  
   Place or sync raw data in `Data/`. Use `src/loader.py` and `src/validation.py` to inspect or clean data.
2. **Feature Engineering:**  
   `src/preprocessing.py` to perform advanced transformations.
3. **Model Training:**  
   Both classical ML (XGBoost) and deep learning (NN) pipelines are available.
4. **Evaluation & Metrics:**  
   Metrics saved in `Metrics/`, tracked in MLflow, and auto-logged via scripts.
5. **Experiment Tracking:**  
   All parameter changes and results can be logged to MLflow for review.
6. **Continuous Integration:**  
   Each push or PR runs checks via `.github/workflows/ci.yml`.

## Dataset

[Data](https://www.kaggle.com/datasets/atechnohazard/battery-and-heating-data-in-real-driving-cycles/data)

## Testing

Automated tests are in `src/tests/`.  
To run all tests:
```bash
pytest src/tests/
```

## Requirements

- All Python dependencies are listed in `requirements.txt`.
- DVC and MLflow installation may require additional system dependencies.

## Contributing

Contributions, feature ideas, or bug fixes are welcome!  
- Fork the repo and open a Pull Request with your proposed changes.
- For major changes, open an issue first to discuss plans.

## License

This project is licensed under the MIT License.
