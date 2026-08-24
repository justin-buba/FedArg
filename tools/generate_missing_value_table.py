from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_PATH = ROOT / "results" / "global" / "missing_value_analysis.png"


def generate_table():
    files = sorted(PROCESSED_DIR.glob("*_processed.csv"))
    frames = [pd.read_csv(path) for path in files]
    data = pd.concat(frames, ignore_index=True)
    total_records = len(data)

    rows = [
        ("Age", "Numerical", "Median"),
        ("Pulse", "Numerical", "Median / default 70"),
        ("Resp", "Numerical", "Median / default 16"),
        ("Temp", "Numerical", "Median / default 37"),
        ("Sys", "Numerical", "Median / default 120"),
        ("Dia", "Numerical", "Median / default 80"),
        ("Sponsor", "Categorical", "No missing values"),
    ]

    table_data = []
    for variable, data_type, method in rows:
        missing_count = int(data[variable].isna().sum())
        missing_percent = missing_count / total_records * 100
        table_data.append([
            variable,
            data_type,
            f"{missing_count:,}",
            f"{missing_percent:.2f}",
            method,
        ])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(12, 5.2))
    axis.axis("off")
    figure.text(
        0.08,
        0.94,
        "Missing value analysis and preprocessing strategy",
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.08,
        0.885,
        f"Processed hospital records analyzed: {total_records:,}",
        ha="left",
        va="top",
        fontsize=10,
    )

    table = axis.table(
        cellText=table_data,
        colLabels=[
            "Variable",
            "Data Type",
            "Missing\nCount",
            "Missing (%)",
            "Imputation\nMethod",
        ],
        colWidths=[0.16, 0.18, 0.18, 0.18, 0.30],
        cellLoc="center",
        loc="center",
        bbox=[0.08, 0.12, 0.84, 0.67],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)

    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#333333")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#e9eef2")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("white")

    figure.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    generate_table()