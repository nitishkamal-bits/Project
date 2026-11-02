import glob
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

TARGET = "early_readmit"
MODEL_DIR = "model"

st.set_page_config(page_title="Readmission Risk - Model Lab", layout="wide")
st.title("30-day Hospital Readmission: Model Lab")
st.caption("Five classifiers trained on the UCI Diabetes 130-US Hospitals data. "
           "Upload a test CSV and compare how each saved model performs. "
           "No retraining happens here.")


@st.cache_resource
def load_models():
    models = {}
    for path in sorted(glob.glob(os.path.join(MODEL_DIR, "*.joblib"))):
        label = os.path.basename(path)[:-7].replace("_", " ").title()
        models[label] = joblib.load(path)
    return models


def scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def metrics(model, X, y):
    pred = model.predict(X)
    return pred, {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, scores(model, X)) if y.nunique() > 1 else float("nan"),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }


models = load_models()
if not models:
    st.error("No saved models found in the model/ folder.")
    st.stop()

with st.sidebar:
    st.header("Controls")
    upload = st.file_uploader("Upload test CSV", type="csv")
    choice = st.selectbox("Choose a model", list(models.keys()))
    st.markdown("The CSV needs every feature column plus the target column "
                f"`{TARGET}` (1 = readmitted within 30 days). "
                "You can download a sample from the GitHub repo (`test_data.csv`).")

if upload is None:
    st.info("Upload a test CSV from the sidebar to begin.")
    st.stop()

data = pd.read_csv(upload)
if TARGET not in data.columns:
    st.error(f"Target column `{TARGET}` is missing from the uploaded file.")
    st.stop()

required = list(models[choice].feature_names_in_)
missing = [c for c in required if c not in data.columns]
if missing:
    st.error("Missing feature columns: " + ", ".join(missing))
    st.stop()

X, y = data[required], data[TARGET]
st.write(f"Loaded **{len(data):,}** rows. Positive rate: **{y.mean():.1%}**")

tab_single, tab_all = st.tabs(["Selected model", "All models side by side"])

with tab_single:
    pred, m = metrics(models[choice], X, y)
    cols = st.columns(6)
    for col, (k, v) in zip(cols, m.items()):
        col.metric(k, f"{v:.3f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        fig, ax = plt.subplots(figsize=(4, 3.4))
        sns.heatmap(confusion_matrix(y, pred), annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Not <30d", "<30d"], yticklabels=["Not <30d", "<30d"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
    with right:
        st.subheader("Classification report")
        report = classification_report(y, pred, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(3))

with tab_all:
    table = pd.DataFrame({name: metrics(mdl, X, y)[1] for name, mdl in models.items()}).T
    table.index.name = "ML Model Name"
    st.dataframe(table.round(4), use_container_width=True)
    st.bar_chart(table[["AUC", "Recall", "F1", "MCC"]])
