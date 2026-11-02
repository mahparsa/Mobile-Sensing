# Nested Cross-Validation Pipeline

Binary classification with nested cross-validation, per-fold preprocessing,
automatic feature selection, and hyperparameter tuning.

## Pipeline (per outer fold)
1. **Scaling**: RobustScaler fitted separately for each platform (train fold only)
2. **Outlier removal**: samples flagged by LOF for every `n_neighbors` in 5–30
3. **Oversampling**: SVMSMOTE to balance classes
4. **Feature selection**: several methods × CV sizes 2–10, best by macro-F1
5. **Tuning**: GridSearchCV over the chosen classifier (inner CV)
6. **Evaluation**: accuracy, weighted F1, AUC, balanced accuracy, sensitivity, specificity

## Setup
```bash
pip install -r requirements.txt
```

## Usage
1. Put your dataset at `data/dataset.csv` (this folder is git-ignored).
2. Edit `config.py`: set `TARGET_COLUMN` and `FEATURE_COLUMNS`.
3. Run:
```bash
python main.py                    # uses MODEL_NAME from config.py
python main.py --model xgboost    # catboost | xgboost | random_forest | gradient_boosting | extra_trees
```
Results (per-fold CSV and ROC plot) are saved to `results/`.

## Project structure
| File | Purpose |
|------|---------|
| `config.py` | All settings |
| `main.py` | Entry point |
| `src/data.py` | Load dataset |
| `src/preprocessing.py` | Scaling, LOF, SVMSMOTE |
| `src/feature_selection.py` | Feature selection |
| `src/models.py` | Classifiers and grids |
| `src/nested_cv.py` | Nested CV loop |
| `src/evaluation.py` | Metrics and plots |

## Note on runtime
Feature selection runs 8 methods × 9 CV values × a 600-combination RF grid
in every outer fold, so this is slow. To speed it up, shrink `FS_METHODS`,
`FS_CV_RANGE` or `FS_RF_PARAM_GRID` in `config.py`.
