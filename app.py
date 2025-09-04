import gradio as gr
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)
import numpy as np
import io

# -------------------------------
# Load model + label encoder
# -------------------------------
try:
    bundle = joblib.load("models/xgb_bundle.joblib")
    xgb_model = bundle["pipeline"]
    le_target = bundle["label_encoder"]
    cv_results = bundle.get("cv_results", None)
    best_params = bundle.get("best_params", None)
    stored_class_report = bundle.get("classification_report", None)
    roc_data = bundle.get("roc_curve", None)
except:
   
    try:
        xgb_model = joblib.load("models/xgb_pipeline.joblib")
        le_target = joblib.load("models/label_encoder.joblib")
    except:
        class DummyModel:
            def predict(self, X):
                return np.zeros(len(X))
        xgb_model = DummyModel()
        le_target = type('DummyEncoder', (), {
            'classes_': ['Accepted', 'Documentation Error', 'Authorization Required'],
            'transform': lambda self, y: [0 if x == 'Accepted' else 1 for x in y],
            'inverse_transform': lambda self, y: ['Accepted' if x == 0 else 'Denied' for x in y]
        })()
    cv_results = None
    best_params = None
    stored_class_report = None
    roc_data = None

required_cols = ["CPT Code", "Insurance Company", "Physician Name", "Payment Amount", "Balance"]

# -------------------------------
# Visualization Functions
# -------------------------------
# -------------------------------
def generate_plots(df):
    figs = []
    plt.style.use('default')  
    
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10

    fig1, ax1 = plt.subplots(figsize=(8, 8))
    insurance_counts = df["Insurance Company"].value_counts()
    if len(insurance_counts) > 0:
        wedges, texts, autotexts = ax1.pie(
            insurance_counts.values, 
            labels=insurance_counts.index, 
            autopct='%1.1f%%', 
            startangle=140, 
            colors=plt.cm.Set3(np.linspace(0, 1, len(insurance_counts))),
            textprops={'fontsize': 9}
        )
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(9)
    else:
        ax1.text(0.5, 0.5, "No insurance data available", 
                ha='center', va='center', transform=ax1.transAxes, fontsize=11)
    ax1.set_title("Claims Distribution by Insurance Company", fontweight='bold', pad=20)
    ax1.set_ylabel("")
    fig1.tight_layout()
    figs.append(fig1)

    fig2, ax2 = plt.subplots(figsize=(8, 8))
    denial_counts = df["Predicted Denial Reason"].value_counts()
    if len(denial_counts) > 0:
        wedges, texts, autotexts = ax2.pie(
            denial_counts.values, 
            labels=denial_counts.index, 
            autopct='%1.1f%%', 
            startangle=140, 
            colors=plt.cm.Set2(np.linspace(0, 1, len(denial_counts))),
            textprops={'fontsize': 9}
        )

        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(9)

        for text in texts:
            text.set_fontsize(9)
    else:
        ax2.text(0.5, 0.5, "No denial predictions available", 
                ha='center', va='center', transform=ax2.transAxes, fontsize=11)
    ax2.set_title("Predicted Denial Reasons Distribution", fontweight='bold', pad=20)
    ax2.set_ylabel("")
    fig2.tight_layout()
    figs.append(fig2)

    fig3, ax3 = plt.subplots(figsize=(10, 6))

    if len(denial_counts) > 0:
        # Use a clean seaborn color palette (pastel / Set2 / tab10 are good)
        colors = sns.color_palette("Set2", n_colors=len(denial_counts))
        
        bars = ax3.bar(
            range(len(denial_counts)),
            denial_counts.values,
            color=colors, alpha=0.85
        )
        
        ax3.set_xticks(range(len(denial_counts)))
        ax3.set_xticklabels(denial_counts.index, rotation=45, ha='right', fontsize=9)
        
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        ax3.set_ylabel("Count", fontweight='bold')
        ax3.grid(axis='y', alpha=0.3, linestyle="--")
    else:
        ax3.text(0.5, 0.5, "No denial data available for bar chart", 
                ha='center', va='center', transform=ax3.transAxes, fontsize=11)

    ax3.set_title("Denial Reasons Frequency Analysis", fontweight='bold', pad=20)
    fig3.tight_layout()
    figs.append(fig3)

    fig4, ax4 = plt.subplots(figsize=(10, 6))
    if len(df) > 1:
        scatter = ax4.scatter(df["Payment Amount"], df["Balance"], alpha=0.6, s=50)
        
        if len(df) > 2:
            z = np.polyfit(df["Payment Amount"], df["Balance"], 1)
            p = np.poly1d(z)
            x_line = np.linspace(df["Payment Amount"].min(), df["Payment Amount"].max(), 100)
            ax4.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label='Trend line')
            ax4.legend()
        
        ax4.set_xlabel("Payment Amount ($)", fontweight='bold')
        ax4.set_ylabel("Balance Amount ($)", fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f'${y:,.0f}'))
        
    else:
        ax4.text(0.5, 0.5, "Insufficient data for payment vs balance analysis\n(Minimum 2 records required)", 
                ha='center', va='center', transform=ax4.transAxes, fontsize=11, linespacing=1.5)
    ax4.set_title("Payment Amount vs Balance Correlation", fontweight='bold', pad=20)
    fig4.tight_layout()
    figs.append(fig4)

    return figs

# -------------------------------
# Helper: Plot ROC Curve
# -------------------------------
def plot_roc_curve(roc_data):
    if roc_data is None:
        fig, ax = plt.subplots(figsize=(4,3))  # smaller default
        ax.text(0.5, 0.5, "ROC data not available", 
                ha='center', va='center', transform=ax.transAxes, fontsize=8)
        return fig
    
    fpr, tpr, _ = roc_data
    fig, ax = plt.subplots(figsize=(0,0))  # smaller size
    ax.plot(fpr, tpr, color="blue", lw=1.5, label="ROC curve")
    ax.plot([0,1],[0,1], color="gray", linestyle="--", lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("FPR", fontsize=8)
    ax.set_ylabel("TPR", fontsize=8)
    ax.set_title("ROC", fontsize=9)
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    return fig


# -------------------------------
# Prediction + Analysis
# -------------------------------
def load_excel_dynamic(file):
    df_raw = pd.read_excel(file, header=None)
    # Assume CPT Code is always in column B (index 1)
    header_row = df_raw.index[df_raw.iloc[:,1] == "CPT Code"][0]
    df = pd.read_excel(file, header=header_row)
    return df



def predict_from_file(file):
    try:
        if hasattr(file, 'name') and file.name.endswith(".csv"):
            df = pd.read_csv(file)
        elif hasattr(file, 'name') and (file.name.endswith(".xlsx") or file.name.endswith(".xls")):
            df = load_excel_dynamic(file)
        else:
            if isinstance(file, str):
                if file.endswith(".csv"):
                    df = pd.read_csv(file)
                else:
                    df = load_excel_dynamic(file)

            else:
                return "Invalid file format", None, None, None, None, None, "N/A", "N/A", "N/A", "N/A", None, None, None, None, pd.DataFrame()

        df.columns = df.columns.str.strip()
        df = df.dropna(how="all").reset_index(drop=True)

        column_mapping = {
            "cpt_code": "CPT Code",
            "insurance": "Insurance Company",
            "doctor": "Physician Name",
            "payment": "Payment Amount",
            "balance_amt": "Balance"
        }
        df.rename(columns=column_mapping, inplace=True)

        for col in ["Payment Amount", "Balance"]:
            if col in df.columns:
                df[col] = (
                    df[col].astype(str)
                    .str.replace(r"[\$,]", "", regex=True)
                    .str.extract(r"([-+]?\d*\.?\d+)")[0]
                )
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        for col in ["CPT Code", "Insurance Company", "Physician Name"]:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna("Unknown").str.strip()

        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            return (f"Missing columns: {missing}", None, None, None, None, None, "N/A", "N/A", "N/A", "N/A", None, None, None, None, pd.DataFrame())

        # Predict
        preds = xgb_model.predict(df[required_cols])
        preds_decoded = le_target.inverse_transform(preds)
        df["Predicted Denial Reason"] = preds_decoded

        # Denial summaries
        cpt_denials = (
            df[df["Predicted Denial Reason"] != "Accepted"]
            .groupby("CPT Code")
            .size()
            .sort_values(ascending=False)
            .head(5)
        )

        insurer_denials = (
            df[df["Predicted Denial Reason"] != "Accepted"]
            .groupby("Insurance Company")
            .size()
            .sort_values(ascending=False)
            .head(5)
        )

        reason_denials = df["Predicted Denial Reason"].value_counts().head(5)

        # Payment efficiency
        df["Total Charge"] = df["Payment Amount"] + df["Balance"]
        df["Payment Efficiency"] = df["Payment Amount"] / df["Total Charge"].replace(0, 1)
        inefficiencies = df.groupby("CPT Code")["Payment Efficiency"].mean().sort_values().head(5)

        # Plots
        figs = generate_plots(df)

        # Metrics
        metrics_text = "Ground truth not provided"
        class_report_df = None
        cm_fig = None
        roc_fig = plot_roc_curve(roc_data)  

        # If bundle already stored a classification report (from training CV)
        if stored_class_report and "Denial Reason" not in df.columns:
            class_report_df = pd.DataFrame(stored_class_report).T
            metrics_text = " Showing stored cross-validation results (no ground truth in input)"

        if "Denial Reason" in df.columns:
            y_true_raw = df["Denial Reason"].astype(str).fillna("Accepted").str.strip()
            valid_idx = y_true_raw.isin(le_target.classes_)
            
            if valid_idx.any():
                try:
                    y_true = le_target.transform(y_true_raw[valid_idx])
                    y_pred = preds[valid_idx]

                    acc = accuracy_score(y_true, y_pred)
                    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
                    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
                    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

                    metrics_text = f"""
                     Model Performance:
                    - Accuracy : {acc:.2%}
                    - Precision: {prec:.2%}
                    - Recall   : {rec:.2%}
                    - F1 Score : {f1:.2%}
                    """

                    # Classification Report - handle class mismatch
                    try:
                        class_report_dict = classification_report(
                            y_true, y_pred, 
                            target_names=le_target.classes_[:len(np.unique(y_true))], 
                            output_dict=True
                        )
                        class_report_df = pd.DataFrame(class_report_dict).T
                    except:
                        class_report_df = pd.DataFrame({"Error": ["Class mismatch in evaluation"]})

                    # Confusion Matrix
                    try:
                        fig_cm, ax_cm = plt.subplots(figsize=(12, 8))

                        cm = confusion_matrix(y_true, y_pred)

                        sns.heatmap(
                            cm, annot=True, fmt="d", cmap="YlGnBu",  # nicer gradient colors
                            xticklabels=le_target.classes_[:len(np.unique(y_true))],
                            yticklabels=le_target.classes_[:len(np.unique(y_true))],
                            cbar=True, linewidths=0.5, linecolor="gray",  # grid lines for clarity
                            ax=ax_cm, annot_kws={"size": 12, "weight": "bold"}  # bigger numbers
                        )

                        ax_cm.set_xlabel("Predicted", fontsize=12, fontweight='bold')
                        ax_cm.set_ylabel("Actual", fontsize=12, fontweight='bold')
                        ax_cm.set_title("Confusion Matrix", fontsize=14, fontweight='bold', pad=20)

                        # Rotate labels for readability
                        ax_cm.set_xticklabels(ax_cm.get_xticklabels(), rotation=45, ha="right")
                        ax_cm.set_yticklabels(ax_cm.get_yticklabels(), rotation=0)

                        fig_cm.tight_layout()
                        cm_fig = fig_cm
                    except:
                        fig_cm, ax_cm = plt.subplots(figsize=(7,6))
                        ax_cm.text(0.5, 0.5, "Could not generate confusion matrix", 
                                  ha='center', va='center', transform=ax_cm.transAxes)
                        cm_fig = fig_cm

                except Exception as e:
                    metrics_text = f"Error in evaluation: {str(e)}"

        # Cross-validation + Best Params
        cv_summary = "N/A"
        if cv_results:
            cv_scores = cv_results.get("mean_test_score", [0])
            cv_summary = f"Cross-Validation Accuracy (avg): {np.mean(cv_scores):.2%}"
        
        best_params_text = str(best_params) if best_params else "N/A"

        # Ensure all outputs are returned
        return (
            metrics_text,
            class_report_df if class_report_df is not None else pd.DataFrame({"Info": ["No classification report available"]}),
            cm_fig if cm_fig is not None else plot_roc_curve(None),  # Reuse ROC function for placeholder
            cv_summary,
            best_params_text,
            roc_fig if roc_fig is not None else plot_roc_curve(None),
            cpt_denials.to_string() if len(cpt_denials) > 0 else "No denials found",
            insurer_denials.to_string() if len(insurer_denials) > 0 else "No insurer denials found",
            reason_denials.to_string() if len(reason_denials) > 0 else "No denial reasons found",
            inefficiencies.to_string() if len(inefficiencies) > 0 else "No efficiency data found",
            figs[0], figs[1], figs[2], figs[3],
            df[required_cols + ["Predicted Denial Reason"]]
        )

    except Exception as e:
        # Return all expected outputs with error messages
        error_msg = f"Error: {str(e)}"
        empty_df = pd.DataFrame()
        empty_plot = plot_roc_curve(None)  # Create empty plot
        
        return (
            error_msg, 
            pd.DataFrame({"Error": [error_msg]}), 
            empty_plot, 
            "N/A", 
            "N/A", 
            empty_plot,
            "N/A", "N/A", "N/A", "N/A",
            empty_plot, empty_plot, empty_plot, empty_plot,
            empty_df
        ) 



# -------------------------------
# Gradio App (Organized UI)
# -------------------------------
with gr.Blocks(title="Insurance Claim Denial Prediction & Analysis") as demo:
    gr.Markdown("##  Insurance Claim Denial Prediction & Analysis")
    gr.Markdown("Upload claims file (.csv or .xlsx). Predict denial reasons, get insights, and visualize claim patterns.")

    with gr.Row():
        file_input = gr.File(label="Upload CSV/Excel File", file_types=[".csv", ".xlsx"])

    with gr.Tab(" Model Performance"):
        metrics_out = gr.Textbox(label=" Accuracy / Precision / Recall / F1")
        with gr.Row():
            report_out = gr.Dataframe(label=" Classification Report")
            cm_out = gr.Plot(label="Confusion Matrix")
        with gr.Row():
            cv_out = gr.Textbox(label=" Cross Validation Score")
            params_out = gr.Textbox(label="⚙ Best Hyperparameters")
        roc_out = gr.Plot(label="ROC Curve")

    with gr.Tab(" Insights"):
        with gr.Row():
            cpt_out = gr.Textbox(label=" Top Denied CPT Codes")
            insurer_out = gr.Textbox(label=" Top Denying Insurers")
        with gr.Row():
            reason_out = gr.Textbox(label=" Top Denial Reasons")
            efficiency_out = gr.Textbox(label=" Lowest Payment Efficiency CPTs")

    with gr.Tab(" Visualizations"):
        with gr.Row():
            fig1 = gr.Plot(label="Claims by Insurance Company")
            fig2 = gr.Plot(label="Predicted Denial Reasons")
        with gr.Row():
            fig3 = gr.Plot(label="Denial Reasons Frequency")
            fig4 = gr.Plot(label="Payments vs Balances")

    with gr.Tab(" Predictions Table"):
        table_out = gr.Dataframe(label="Predictions Table")

    # Connect function
    file_input.change(
        fn=predict_from_file,
        inputs=file_input,
        outputs=[
            metrics_out,
            report_out,
            cm_out,
            cv_out,
            params_out,
            roc_out,
            cpt_out, 
            insurer_out, 
            reason_out, 
            efficiency_out,
            fig1, fig2, fig3, fig4,
            table_out
        ]
    )

if __name__ == "__main__":
    demo.launch(share = True)