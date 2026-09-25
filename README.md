# Heart Disease Prediction

A binary classification project predicting whether a patient has heart disease from 13 clinical parameters, using the [Cleveland Heart Disease dataset](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci) (297 patients).

## Approach

1. **EDA** — class balance, heart-disease rate by sex, age vs. max heart rate, chest-pain type vs. diagnosis, and a full feature correlation matrix.
2. **Modelling** — five classifiers compared on a held-out test split: Logistic Regression, KNN, Random Forest, XGBoost Random Forest, and Extra Trees.
3. **Tuning** — `RandomizedSearchCV` and `GridSearchCV` over Logistic Regression, Random Forest, and XGBoost Random Forest hyperparameters.
4. **Evaluation** — ROC/AUC, confusion matrix, classification report, and 10-fold cross-validated accuracy/precision/recall/F1 on the best model.

## Results

| Model | Test accuracy |
|---|---|
| Extra Trees Classifier | **86.7%** |
| Logistic Regression (tuned) | 85.0% |
| XGBoost Random Forest | 76.7% |
| Random Forest | 75.0% |
| KNN | 60.0% |

Best model (Extra Trees), 10-fold cross-validated: **82.1% accuracy**, 84.2% precision, 77.6% recall, 78.2% F1.

## Repository layout

- `Heart_Disease_prediction.ipynb` — full pipeline: EDA, modelling, tuning, evaluation.
- `heart_cleveland_upload.csv` — the dataset.

## Running it

```bash
pip install -r requirements.txt
jupyter lab
```
Open `Heart_Disease_prediction.ipynb` and run all cells.

## Tech stack

Python, pandas, NumPy, scikit-learn, XGBoost, matplotlib, seaborn, klib

## Author

Rajat Satonkar — rajatsatonkar@gmail.com
