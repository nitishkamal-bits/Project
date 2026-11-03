# 30-Day Hospital Readmission Prediction (Diabetes 130-US Hospitals)

## a. Problem statement
Hospital readmissions within 30 days of discharge are costly and are widely used as a
quality-of-care indicator. Given the clinical and administrative details of a diabetic
patient's hospital stay, the goal is to **predict whether the patient is readmitted in
fewer than 30 days** (binary classification: `1` = readmitted <30 days, `0` = not
readmitted within 30 days, or never readmitted). Five classifiers are trained on the same
data, compared on six metrics, and served through an interactive Streamlit app.

## b. Dataset description
- **Source:** UCI Machine Learning Repository - *Diabetes 130-US Hospitals for Years 1999-2008*
  (https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
- **Raw size:** 101,766 encounters x 50 columns from 130 US hospitals.
- **Target:** original `readmitted` (`<30`, `>30`, `NO`) converted to binary
  `readmitted_30d` (`<30` -> 1, otherwise 0).
- **Cleaning / preprocessing**
  - Kept only the first encounter per patient (`patient_nbr`) so no patient appears in both
    train and test, and removed patients who died or went to hospice (they cannot be
    readmitted) -> 69,973 rows.
  - Dropped identifiers and columns with heavy missingness or very high cardinality
    (`weight`, `payer_code`, `medical_specialty`, `diag_2`, `diag_3`).
  - `diag_1` (ICD-9) grouped into broad clinical categories (`diag_1_group`).
  - Missing values: median (numeric) / most-frequent (categorical) imputation; numeric
    features standardised; categorical features one-hot encoded. All steps sit inside one
    scikit-learn `Pipeline` per model, so raw test rows can be scored directly.
- **Features used (25; minimum required 12):** 8 numeric - `time_in_hospital`,
  `num_lab_procedures`, `num_procedures`, `num_medications`, `number_outpatient`,
  `number_emergency`, `number_inpatient`, `number_diagnoses`; 17 categorical - `race`,
  `gender`, `age`, `admission_type_id`, `discharge_disposition_id`, `admission_source_id`,
  `diag_1_group`, `max_glu_serum`, `A1Cresult`, `metformin`, `glipizide`, `glyburide`,
  `pioglitazone`, `rosiglitazone`, `insulin`, `change`, `diabetesMed`.
- **Split:** stratified 80/20 (`random_state=42`): 55,978 training rows, 13,995 test rows.
  The classes are strongly imbalanced - only about **9%** of encounters are positive.
  `test_data.csv` holds the held-out rows (raw feature values + `readmitted_30d`).

## c. GitHub repository link
https://github.com/<your-username>/<your-repo>   *(replace with your repo URL)*

```
project-folder/
|-- app.py            single file: preprocessing, training, evaluation and Streamlit UI
|-- requirements.txt
|-- README.md
|-- test_data.csv     held-out test set (features + target)
|-- train_data.csv    training split (only used to refit models if saved ones cannot load)
|-- results.csv       metric table below
|-- model/            saved pipelines: logistic_regression, decision_tree, knn,
|                     naive_bayes, support_vector_machine (.joblib)
```
Run the app: `pip install -r requirements.txt`, then `python3 -m streamlit run app.py`.
Retrain: `python3 app.py train diabetic_data.csv` (original UCI file).

## d. Models used

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|:--|--:|--:|--:|--:|--:|--:|
| Logistic Regression | 0.6700 | 0.6440 | 0.1378 | 0.5100 | 0.2170 | 0.1188 |
| Decision Tree | 0.6011 | 0.6291 | 0.1252 | 0.5761 | 0.2057 | 0.1043 |
| KNN | 0.6110 | 0.5669 | 0.1086 | 0.4629 | 0.1759 | 0.0521 |
| Naive Bayes | 0.7452 | 0.6295 | 0.1493 | 0.3920 | 0.2163 | 0.1158 |
| Support Vector Machine | 0.6556 | 0.6298 | 0.1351 | 0.5259 | 0.2150 | 0.1164 |

Precision, recall and F1 are for the positive class (readmitted <30 days).
Settings: Logistic Regression (`class_weight=balanced`); Decision Tree (`max_depth=6`,
`min_samples_leaf=50`, balanced); KNN (`k=25`); Gaussian Naive Bayes
(`var_smoothing=0.01`, chosen on a validation split of the training data); SVM (RBF
kernel, `C=1`). KNN, Naive Bayes and SVM are fitted on a class-balanced training
subsample (5,000 rows per class) because they cannot weight classes the same way, and
SVM/KNN are expensive on the full set. All five models are scored on the same full test set.

### Observations

| ML Model Name | Observation about model performance |
|:--|:--|
| Logistic Regression | Best overall: highest AUC (0.644) and MCC (0.119), with an F1 of 0.217. Class weighting lets it find about half of the true early readmissions (recall 0.51), at the price of many false alarms (precision 0.14). Fast, stable and easy to interpret. |
| Decision Tree | AUC 0.629 and the highest recall (0.576), but the lowest accuracy (0.60) because balanced weighting makes it flag many patients. A shallow depth limits overfitting, and the rules are easy to explain to clinicians. |
| KNN | Weakest ranking ability (AUC 0.567, MCC 0.052). Distances in a one-hot, 100-plus-dimension space are not very informative, so neighbours are only slightly more likely to share the label. Training on a balanced subsample stops it from predicting "not readmitted" for everyone. |
| Naive Bayes | Competitive despite its simplicity: AUC 0.630, F1 0.216 and the highest precision (0.149) and accuracy (0.745) among the models. With default smoothing it labelled nearly every patient positive; raising `var_smoothing` fixed this. The independence assumption is only roughly true for clinical features. |
| Support Vector Machine | Very close to Logistic Regression (AUC 0.630, F1 0.215, MCC 0.116), which suggests the boundary is close to linear on this noisy data. Trained on only 10,000 rows and by far the slowest model to fit, so it adds cost without a gain. |
| **Overall winner** | **Logistic Regression** - best AUC and MCC, joint-best F1, and negligible training cost. Predicting 30-day readmission from these features is intrinsically hard (every AUC is below 0.65), so the models are screening aids rather than reliable predictors. Accuracy alone is misleading on this 9% positive-class problem. |

## e. Streamlit app features
CSV upload (test data only; the bundled `test_data.csv` is used when nothing is uploaded),
model dropdown, the six metrics, confusion matrix, ROC curve, classification report, and an
expandable comparison of all five models on the same data. Saved models are loaded and
never retrained on the uploaded data.
Live app: https://<your-app>.streamlit.app   *(replace with your deployed URL)*
