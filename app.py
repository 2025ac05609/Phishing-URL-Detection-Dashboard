import streamlit as st
import pandas as pd
import joblib
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns

# Model Accuracy Dictionary
model_accuracy = {
    "Logistic Regression": 97.84,
    "Decision Tree": 96.15,
    "K-Nearest Neighbors": 97.22,
    "Naive Bayes": 94.81,
    "Random Forest": 98.76
}

# Page Configuration

st.set_page_config(
    page_title="Phishing URL Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>

/* Main Background */
.stApp {
    background-color: #F4F8FB;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #EAF2F8;
}

/* Headers */
h1, h2, h3 {
    color: #1F3C88;
}

</style>
""", unsafe_allow_html=True)

# Dashboard Header

st.markdown("""
<h1 style="color:#1F3C88;">
🛡️ Phishing URL Detection Dashboard
</h1>
""", unsafe_allow_html=True)

st.markdown("""
This dashboard classifies URLs as **Legitimate** or **Phishing**
using five supervised Machine Learning models trained on the
PhiUSIIL Phishing URL Dataset.
""")

st.caption(
    "Developed by Sasanka Vutukuru | BITS Pilani M.Tech (AIML) | Machine Learning Assignment 2"
)

st.divider()

# Sidebar Navigation

st.sidebar.markdown(
    """
    <h2 style="color:#1F3C88;">
        🧭 Navigation
    </h2>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    """
    ### Instructions

    1. Upload the test CSV file.
    2. Select a Machine Learning model.
    3. View the prediction results.
    """
)

st.sidebar.divider()

# Upload Test Dataset

uploaded_test_data = st.file_uploader(
    "📂 Upload Test CSV File",
    type=["csv"]
)

# Available Machine Learning Models

available_models = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "Decision Tree": "models/decision_tree.pkl",
    "K-Nearest Neighbors": "models/knn.pkl",
    "Naive Bayes": "models/naive_bayes.pkl",
    "Random Forest": "models/random_forest.pkl"
}

st.sidebar.subheader("Classification Model")

selected_model = st.sidebar.selectbox(
    "Choose a Machine Learning Model",
    list(available_models.keys())
)

# Load Selected Model

loaded_model = joblib.load(
    available_models[selected_model]
)

# Read Uploaded Dataset

if uploaded_test_data is not None:

    uploaded_dataset = pd.read_csv(uploaded_test_data)

    st.subheader("📄 Dataset Preview")
    st.dataframe(uploaded_dataset.head())

    # Dataset Information
    st.write("### Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", uploaded_dataset.shape[0])

    with col2:
        st.metric("Columns", uploaded_dataset.shape[1])

# Separate Features and Target
    test_features = uploaded_dataset.drop("label", axis=1)
    test_labels = uploaded_dataset["label"]

    # Apply Feature Scaling
    if selected_model in [
        "Logistic Regression",
        "K-Nearest Neighbors"
    ]:
        scaler = joblib.load("models/scaler.pkl")
        prediction_features = scaler.transform(test_features)

    else:
        prediction_features = test_features

# Prediction Pipeline

    start_time = time.perf_counter()

    predicted_labels = loaded_model.predict(
        prediction_features
    )

    prediction_probabilities = loaded_model.predict_proba(
        prediction_features
    )[:, 1]

    end_time = time.perf_counter()

    prediction_time = end_time - start_time

# Model Evaluation

    accuracy = accuracy_score(
        test_labels,
        predicted_labels
    )

    precision = precision_score(
        test_labels,
        predicted_labels
    )

    recall = recall_score(
        test_labels,
        predicted_labels
    )

    f1 = f1_score(
        test_labels,
        predicted_labels
    )

    auc = roc_auc_score(
        test_labels,
        prediction_probabilities
    )

    mcc = matthews_corrcoef(
        test_labels,
        predicted_labels
    )

# Quick Facts

    st.subheader("📈 Quick Facts")

    quick_col1, quick_col2, quick_col3, quick_col4 = st.columns(4)

    with quick_col1:
        st.metric(
            "Dataset Size",
            f"{uploaded_dataset.shape[0]:,}"
        )

    with quick_col2:
        st.metric(
            "Features",
            test_features.shape[1]
        )

    with quick_col3:
        st.metric(
            "Selected Model",
            selected_model
        )

    with quick_col4:
        st.metric(
            "Prediction Time",
            f"{prediction_time:.4f} sec"
        )


# Hero Metrics

    total_urls = len(predicted_labels)
    phishing = (predicted_labels == 1).sum()
    legitimate = (predicted_labels == 0).sum()

    st.subheader("🚀 Prediction Dashboard")

    hero1, hero2, hero3, hero4 = st.columns(4)

    with hero1:
        st.metric("📂 URLs Uploaded", total_urls)

    with hero2:
        st.metric("🟢 Legitimate URLs", legitimate)

    with hero3:
        st.metric("🔴 Phishing URLs", phishing)

    with hero4:
        st.metric("🎯 Accuracy", f"{accuracy*100:.2f}%")

# Display Model Performance

    st.success(f"Current Model: {selected_model}")

    st.subheader("📊 Model Performance")

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:
        st.metric("Accuracy", f"{accuracy * 100:.2f}%")

    with metric_col2:
        st.metric("Precision", f"{precision * 100:.2f}%")

    with metric_col3:
        st.metric("Recall", f"{recall * 100:.2f}%")

    metric_col4, metric_col5, metric_col6 = st.columns(3)

    with metric_col4:
        st.metric("F1 Score", f"{f1 * 100:.2f}%")

    with metric_col5:
        st.metric("AUC Score", f"{auc * 100:.2f}%")

    with metric_col6:
        st.metric("MCC Score", f"{mcc:.4f}")

# Performance Summary

    st.subheader("📈 Performance Summary")

    if accuracy >= 0.99:
        st.success(
            "🎯 Excellent model performance! This classifier achieves outstanding predictive accuracy."
        )

    elif accuracy >= 0.95:
        st.info(
            "👍 Good model performance. The classifier produces reliable predictions."
        )

    elif accuracy >= 0.90:
        st.warning(
            "⚠️ Acceptable performance. Further tuning may improve the model."
        )

    else:
        st.error(
            "❌ Poor performance. Consider feature engineering or model optimisation."
        )

# Confusion Matrix

    st.subheader("📊 Confusion Matrix")

    conf_matrix = confusion_matrix(
        test_labels,
        predicted_labels
    )

    fig, ax = plt.subplots(figsize=(4.5, 4))

    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        linewidths=0.5,
        linecolor="white",
        cbar=True,
        annot_kws={"size": 10},
        ax=ax
    )

    ax.set_xlabel(
        "Predicted Label",
        fontsize=10
    )

    ax.set_ylabel(
        "Actual Label",
        fontsize=10
    )
    ax.set_title(
        f"{selected_model} Confusion Matrix",
        fontsize=12
    )

    st.pyplot(fig)

# Classification Report

    st.subheader("📋 Classification Report")

    report = classification_report(
        test_labels,
        predicted_labels,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

# Rename class labels for better readability

    report_df.rename(
        index={
            "0": "Legitimate URL",
            "1": "Phishing URL",
            "accuracy": "Overall Accuracy",
            "macro avg": "Macro Average",
            "weighted avg": "Weighted Average"
        },
        inplace=True
    )
    report_df.index.name = "Class"

    styled_report = report_df.style.format({
        "precision": "{:.2%}",
        "recall": "{:.2%}",
        "f1-score": "{:.2%}",
        "support": "{:,.0f}"
    })

    st.dataframe(
        styled_report,
        use_container_width=True
    )

# Model Comparison

    st.subheader("🏆 Model Comparison")
    comparison_df = pd.read_csv("model_comparison_results.csv")

    comparison_display = comparison_df.copy()

# Convert metrics to percentages

    percentage_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "AUC"
    ]

    for column in percentage_columns:
        comparison_display[column] = (
            comparison_display[column] * 100
        ).map(lambda x: f"{x:.2f}%")

    comparison_display["MCC"] = comparison_display["MCC"].map(
        lambda x: f"{x:.4f}"
    )
    st.dataframe(
        comparison_display.reset_index(drop=True),
        use_container_width=True
    )

# Best Performing Model

    best_accuracy = comparison_df["Accuracy"].max()
    best_models = comparison_df[
        comparison_df["Accuracy"] == best_accuracy
    ]

    st.subheader("🥇 Best Performing Model")

    if len(best_models) == 1:

        best_model = best_models.iloc[0]

        st.success(
            f"""
    ### 🏆 {best_model['Model']}

    **Accuracy:** {best_model['Accuracy']*100:.2f}%  
    **Precision:** {best_model['Precision']*100:.2f}%  
    **Recall:** {best_model['Recall']*100:.2f}%  
    **F1 Score:** {best_model['F1 Score']*100:.2f}%  
    **AUC:** {best_model['AUC']*100:.2f}%  
    **MCC:** {best_model['MCC']:.4f}
    """
        )

    else:

        st.success(
            f"🏆 **{len(best_models)} models achieved the highest accuracy of {best_accuracy*100:.2f}%**"
        )

# Format the best model(s) for display
        best_models_display = best_models.copy()

        percentage_columns = [
            "Accuracy",
            "AUC",
            "Precision",
            "Recall",
            "F1 Score"
        ]

        for column in percentage_columns:
            best_models_display[column] = (
                best_models_display[column] * 100
            ).map(lambda x: f"{x:.2f}%")

        best_models_display["MCC"] = best_models_display["MCC"].map(
            lambda x: f"{x:.4f}"
        )

        st.dataframe(
            best_models_display.reset_index(drop=True),
            use_container_width=True
        )

# Prediction Summary

    total_urls = len(predicted_labels)
    phishing = (predicted_labels == 1).sum()
    legitimate = (predicted_labels == 0).sum()

    st.subheader("🚀 Prediction Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "📂 URLs Uploaded",
            total_urls
        )

    with col2:
        st.metric(
            "🟢 Legitimate",
            legitimate
        )

    with col3:
        st.metric(
            "🔴 Phishing",
            phishing
        )

    with col4:
        st.metric(
            "🎯 Accuracy",
            f"{accuracy*100:.2f}%"
        )

    summary_df = pd.DataFrame({
        "Prediction": ["Legitimate", "Phishing"],
        "Count": [legitimate, phishing]
    })

    fig = px.pie(
        summary_df,
        names="Prediction",
        values="Count",
        hole=0.5
    )

    fig.update_layout(
        title="Prediction Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# Footer

    st.divider()

    st.markdown(
        """
    ### 📌 Project Information

    **Course:** Machine Learning (BITS Pilani WILP)

    **Programme:** M.Tech Artificial Intelligence & Machine Learning

    **Assignment:** Assignment 2 – Machine Learning Model Deployment

    **Dataset:** PhiUSIIL Phishing URL Dataset (UCI Machine Learning Repository)

    **Developed By:** Sasanka Vutukuru
    """
    )

    current_year = datetime.now().year

    st.caption(
        f"© {current_year} | Phishing URL Detection Dashboard | Built using Python, Scikit-learn and Streamlit"
    )

# Download Prediction Results

    st.subheader("📥 Download Prediction Results")

# Create results dataframe
    results_df = uploaded_dataset.copy()

    results_df["Predicted Label"] = predicted_labels

# Convert numeric labels into readable text
    results_df["Actual Label"] = results_df["label"].map({
        0: "Legitimate URL",
        1: "Phishing URL"
    })

    results_df["Predicted Label"] = results_df["Predicted Label"].map({
        0: "Legitimate URL",
        1: "Phishing URL"
    })

# Rearrange columns so labels appear first
    results_df.drop(columns=["label"], inplace=True)
    prediction_results = results_df[
        ["Actual Label", "Predicted Label"] +
        [col for col in results_df.columns
         if col not in ["Actual Label", "Predicted Label"]]
    ]

    st.dataframe(
        prediction_results.head(),
        use_container_width=True
    )

    csv = prediction_results.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Prediction Results (CSV)",
        data=csv,
        file_name=f"{selected_model.lower().replace(' ', '_')}_predictions.csv",
        mime="text/csv"
    )