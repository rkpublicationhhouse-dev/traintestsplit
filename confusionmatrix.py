import streamlit as st
import pandas as pd
from sklearn.metrics import confusion_matrix

st.set_page_config(page_title="Confusion Matrix & Metrics", layout="centered")

st.title("Confusion Matrix & Evaluation Metrics")
st.write("Labels supported: **Cat** and **Dog**")

# --- User Input ---
true_labels = st.text_input(
    "Enter **Actual Labels** (comma-separated: Cat, Dog)",
    " "
)

pred_labels = st.text_input(
    "Enter **Predicted Labels** (comma-separated: Cat, Dog)",
    " "
)

if st.button("Calculate"):
    try:
        y_true = [x.strip().capitalize() for x in true_labels.split(",")]
        y_pred = [x.strip().capitalize() for x in pred_labels.split(",")]

        if len(y_true) != len(y_pred):
            st.error("❌ Number of actual and predicted labels must match.")
            st.stop()

        # --- Confusion Matrix ---
        labels = ["Cat", "Dog"]
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        cm_df = pd.DataFrame(
            cm,
            index=["Actual Cat", "Actual Dog"],
            columns=["Predicted Cat", "Predicted Dog"]
        )

        # --- Display Confusion Matrix ---
        st.subheader("Confusion Matrix")

        styled_cm = cm_df.style.set_properties(
            **{
                "text-align": "center",
                "font-size": "20px"
            }
        ).set_table_styles([
            {
                "selector": "th",
                "props": [("text-align", "center"), ("font-size", "20px")]
            },
            {
                "selector": "td",
                "props": [("padding", "10px")]
            }
        ])

        st.dataframe(styled_cm, use_container_width=True)

        # --- Extract TP, FP, TN, FN ---
        # Cat is Positive class
        TP = cm[0, 0]
        FN = cm[0, 1]
        FP = cm[1, 0]
        TN = cm[1, 1]

        st.subheader("Step-by-Step Values")
        st.markdown(
            f"""
            **True Positive (TP)** = {TP}  
            **False Positive (FP)** = {FP}  
            **True Negative (TN)** = {TN}  
            **False Negative (FN)** = {FN}
            """
        )

        # --- Metrics Calculation ---
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) != 0 else 0
        )

        # --- Metrics Table ---
        metrics_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
            "Formula": [
                "(TP + TN) / (TP + TN + FP + FN)",
                "TP / (TP + FP)",
                "TP / (TP + FN)",
                "2 × (Precision × Recall) / (Precision + Recall)"
            ],
            "Value": [
                f"{accuracy:.2f}",
                f"{precision:.2f}",
                f"{recall:.2f}",
                f"{f1:.2f}"
            ]
        })

        styled_metrics = metrics_df.style.set_properties(
            **{
                "text-align": "center",
                "font-size": "2018px"
            }
        ).set_table_styles([
            {
                "selector": "th",
                "props": [("text-align", "center"), ("font-size", "20px")]
            },
            {
                "selector": "td",
                "props": [("padding", "10px")]
            }
        ])

        st.subheader("Evaluation Metrics")
        st.dataframe(styled_metrics, use_container_width=False)

    except Exception as e:
        st.error("⚠️ Please enter labels correctly using Cat or Dog only.")

