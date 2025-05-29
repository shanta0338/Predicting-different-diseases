# Disease Classification Using Stacking Ensemble Learning

This project applies a machine learning pipeline to classify disease categories based on a wide range of blood test and physiological features. A stacking ensemble architecture with hyperparameter tuning is used to improve classification performance.

## 🔬 Problem Statement

Given a dataset of patients' blood and health metrics, the goal is to predict the disease category associated with each patient. The target variable is `Disease`, which is a multi-class label.

## 🧪 Features Used

The model uses the following 24 features:

- Glucose
- Cholesterol
- Hemoglobin
- Platelets
- White Blood Cells
- Red Blood Cells
- Hematocrit
- Mean Corpuscular Volume
- Mean Corpuscular Hemoglobin
- Mean Corpuscular Hemoglobin Concentration
- Insulin
- BMI
- Systolic Blood Pressure
- Diastolic Blood Pressure
- Triglycerides
- HbA1c
- LDL Cholesterol
- HDL Cholesterol
- ALT
- AST
- Heart Rate
- Creatinine
- Troponin
- C-reactive Protein

## 🧰 Technologies Used

- **Label Encoding** – for converting categorical data into numeric format.
- **Pipeline** – to streamline preprocessing and modeling steps.
- **StackingClassifier** – for combining multiple base learners with a meta learner.
- **RandomizedSearchCV** – for hyperparameter tuning with cross-validation.

## ⚙️ Model Architecture

```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

base_model = [
    ('gb', GradientBoostingClassifier()),
    ('lr', LogisticRegression())
]

stack = StackingClassifier(
    estimators=base_model,
    final_estimator=RandomForestClassifier()
)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', stack)
])
