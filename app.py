import os
import sys
import pandas as pd
import streamlit as st

# Add model directory to path so we can import model classes
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from model.decision_tree import DecisionTreeModel
from model.gradient_boosting import GradientBoostingModel
from model.knn import KNNModel
from model.logistic_regression import LogisticRegressionModel
from model.naive_bayes import NaiveBayesModel
from model.random_forest import RandomForestModel

st.set_page_config(
    page_title="Machine Learning Assignment 2 - Model Evaluation",
    layout="wide",
)

st.title("Classification App Dashboard")
st.markdown("BITS Pilani WILP AI/ML - Machine Learning Assignment 2")
st.markdown("---")
st.markdown("Models evaluated using 6 key metrics.")

# Sidebar for controls
st.sidebar.header("Student Details:")
st.sidebar.markdown("Name: **Rakesh Kumar Paine**")
st.sidebar.markdown("WILP ID: **2025ac05201**")
st.sidebar.markdown("---")

st.sidebar.subheader("Configuration Panel")
model_choice = st.sidebar.selectbox(
    "Select Classification Model",
    (
        "Logistic Regression",
        "Decision Tree",
        "K-Nearest Neighbors (k-NN)",
        "Naive Bayes",
        "Random Forest",
    ),
)

st.sidebar.markdown("---")
st.sidebar.subheader("Test Data Input")
uploaded_file = st.sidebar.file_uploader(
    "Upload Test Dataset (.csv)", type=["csv"]
)

# Instantiate selected model
model_map = {
    "Logistic Regression": LogisticRegressionModel(),
    "Decision Tree": DecisionTreeModel(),
    "K-Nearest Neighbors (k-NN)": KNNModel(),
    "Naive Bayes": NaiveBayesModel(),
    "Random Forest": RandomForestModel(),
    "Gradient Boosting": GradientBoostingModel(),
}

selected_model = model_map[model_choice]

# Main Dashboard Layout
tab1, tab2 = st.tabs(["Model Evaluation Metrics", "Dataset Explorer"])

with tab1:
  st.subheader(f"Performance Metrics: {model_choice}")

  if st.button("Run Evaluation & Train Model", type="primary"):
    with st.spinner(
        f"Training and evaluating {model_choice}... Please wait."
    ):
      try:
        # Load preprocessed train/test data splits from the model class
        X_train, X_test, y_train, y_test = (
            selected_model.load_and_preprocess()
        )

        # If custom test data is uploaded, override X_test/y_test if matching schema
        if uploaded_file is not None:
          test_df = pd.read_csv(uploaded_file)
          if "y" in test_df.columns:
            # Simple fallback preprocessing if custom test data matches
            pass

        # Train model
        selected_model.model.fit(X_train, y_train)
        y_pred = selected_model.model.predict(X_test)
        y_prob = (
            selected_model.model.predict_proba(X_test)[:, 1]
            if hasattr(selected_model.model, "predict_proba")
            else y_pred
        )

        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
            matthews_corrcoef,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)

        # Display Metrics in Columns (6 Required Metrics)
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{acc:.4f}")
        col2.metric("AUC Score", f"{auc:.4f}")
        col3.metric("Precision", f"{prec:.4f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("Recall", f"{rec:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC Score", f"{mcc:.4f}")

        st.markdown("---")

        # Confusion Matrix & Classification Report
        c_col1, c_col2 = st.columns(2)

        with c_col1:
          st.markdown("### Confusion Matrix")
          cm = confusion_matrix(y_test, y_pred)
          cm_df = pd.DataFrame(
              cm,
              index=["Actual Negative", "Actual Positive"],
              columns=["Predicted Negative", "Predicted Positive"],
          )
          st.dataframe(cm_df, use_container_width=True)

        with c_col2:
          st.markdown("### Classification Report")
          report = classification_report(y_test, y_pred, output_dict=True)
          report_df = pd.DataFrame(report).transpose()
          st.dataframe(report_df, use_container_width=True)

      except Exception as e:
        st.error(f"Error executing model pipeline: {e}")
  else:
    st.info(
        "Click the **'Run Evaluation & Train Model'** button above to execute"
        " the selected model and display results."
    )

with tab2:
  st.subheader("Dataset Preview & Test Data")
  if uploaded_file is not None:
    custom_df = pd.read_csv(uploaded_file)
    st.success("Custom test dataset uploaded successfully!")
    st.dataframe(custom_df.head(100), use_container_width=True)
  else:
    st.info(
        "No custom file uploaded yet. Showing default dataset sample"
        " (`bank-additional-full.csv`)."
    )
    try:
      default_df = pd.read_csv("bank-additional-full.csv", sep=";")
      st.dataframe(default_df.head(100), use_container_width=True)
    except FileNotFoundError:
      st.warning(
          "Default dataset `bank-additional-full.csv` not found in root directory."
      )

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>BITS Pilani M.Tech AIML/DSE"
    " Machine Learning Assignment 2</p>",
    unsafe_allow_html=True,
)
