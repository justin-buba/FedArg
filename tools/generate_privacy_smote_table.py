from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "results" / "experiment_summary.csv"
COMPARISON_DIR = ROOT / "results" / "experiments" / "comparisons"
OUTPUT_PATH = ROOT / "results" / "global" / "privacy_smote_results.png"


def format_metric(summary, metric):
    row = summary.loc[summary["Metric"] == metric].iloc[0]
    if pd.isna(row["Std"]):
        return f"{row['Mean']:.2f}"
    return f"{row['Mean']:.2f} +/- {row['Std']:.2f}"


def add_table(axis, title, columns, rows, bbox):
    axis.text(0.02, bbox[1] + bbox[3] + 0.035, title, fontsize=14, weight="bold")
    table = axis.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        loc="center",
        bbox=bbox,
        colWidths=[0.20, 0.1333, 0.1333, 0.1333, 0.1333, 0.1333, 0.1333],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#333333")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#e9eef2")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("white")


def generate_table():
    metrics = ["Accuracy", "Precision", "Recall", "F1", "FPR", "FNR"]
    result_files = {
        "baseline": COMPARISON_DIR / "baseline_smote.csv",
        "dp_only": COMPARISON_DIR / "dp_only_smote.csv",
        "dp_smpc_smote": COMPARISON_DIR / "dp_smpc_smote.csv",
    }

    summaries = {}
    for name, path in result_files.items():
        data = pd.read_csv(path)
        if "Metric" in data.columns:
            summaries[name] = data
        else:
            summaries[name] = pd.DataFrame({
                "Metric": ["Accuracy", "Precision", "Recall", "Specificity", "F1", "FPR", "FNR"],
                "Mean": [data[column].mean() for column in ["Accuracy", "Precision", "Recall", "Specificity", "F1", "FPR", "FNR"]],
                "Std": [data[column].std() for column in ["Accuracy", "Precision", "Recall", "Specificity", "F1", "FPR", "FNR"]],
            })

    values = {
        name: [format_metric(summary, metric) for metric in metrics]
        for name, summary in summaries.items()
    }

    figure, axis = plt.subplots(figsize=(13, 5))
    axis.axis("off")
    figure.text(
        0.05,
        0.96,
        "Table of Results on Privacy and SMOTE",
        ha="left",
        va="top",
        fontsize=17,
        weight="bold",
    )
    figure.text(
        0.05,
        0.915,
        "Comparison runs completed one by one; values are percentages and use one run unless +/- std is shown",
        ha="left",
        va="top",
        fontsize=10,
    )

    add_table(
        axis,
        "Privacy configuration",
        ["Setup", "Accuracy", "Precision", "Recall", "F1", "FPR", "FNR"],
        [
            ["Baseline FL", *values["baseline"]],
            ["FL + DP only", *values["dp_only"]],
            ["FL + DP + SMPC + SMOTE", *values["dp_smpc_smote"]],
        ],
        [0.03, 0.28, 0.94, 0.48],
    )

    figure.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    generate_table()
