# client/clean_app.py
from pathlib import Path
import pandas as pd

def clean_all_data():
    """
    Reads all uncleaned CSVs from data/processed, fills missing vital signs,
    bins Age into categories, cleans Outcome column (including unknowns),
    and saves cleaned files to data/cleaned.
    """
    
    ROOT = Path(__file__).resolve().parents[1]
    PROCESSED_DIR = ROOT / "data" / "processed"
    CLEANED_DIR   = ROOT / "data" / "cleaned"
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    # Pick only uncleaned CSVs
    csv_files = [f for f in PROCESSED_DIR.glob("*.csv") if "cleaned_" not in f.name]
    csv_files += [f for f in PROCESSED_DIR.glob("*.CSV") if "cleaned_" not in f.name]
    csv_files = sorted(list(set(csv_files)))  # remove duplicates

    if not csv_files:
        print("❌ No uncleaned CSV files found in data/processed/")
        return

    columns_to_drop = ["Id", "Name", "Gender", "Date of birth", "Ward", "District", "Diagnoses"]
    vital_columns = ["Pulse", "Resp", "Temp", "Sys", "Dia"]

    # Correct mapping according to actual raw CSV + Unknown category
    # Verified from source data: Home, Referral, Death present (not Admitted/Referred)
    outcome_map = {
        "Home": 0,
        "Referral": 1,  # Patient referred to another facility
        "Death": 2      # Patient died
    }
    UNKNOWN_OUTCOME = 3  # numeric code for missing/unknown/NaN

    for input_path in csv_files:
        print(f"→ Processing: {input_path.name}")
        try:
            df = pd.read_csv(input_path, low_memory=False, dtype_backend="numpy_nullable", on_bad_lines="warn")
            df_clean = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors="ignore")

            # ---- Vital signs cleaning with clinical range validation ----
            # Clinically justified ranges for vital signs (adult patients)
            vital_ranges = {
                'Pulse': (30, 200),       # beats per minute
                'Resp': (5, 50),          # breaths per minute
                'Temp': (33, 43),         # degrees Celsius
                'Sys': (50, 250),         # mmHg systolic
                'Dia': (30, 150)          # mmHg diastolic
            }

            for col in vital_columns:
                if col in df_clean.columns:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

                    if df_clean[col].dropna().empty:
                        default_values = {"Pulse": 70, "Resp": 16, "Temp": 37, "Sys": 120, "Dia": 80}
                        fill_val = default_values[col]
                    else:
                        fill_val = df_clean[col].median()

                    # Replace missing values with median
                    df_clean[col] = df_clean[col].fillna(fill_val)
                    
                    # Replace implausible values (out of clinical range) with median
                    lo, hi = vital_ranges.get(col, (-float('inf'), float('inf')))
                    implausible_mask = (df_clean[col] < lo) | (df_clean[col] > hi)
                    implausible_count = implausible_mask.sum()
                    if implausible_count > 0:
                        print(f"    ⚠️  {col}: {implausible_count} implausible values (outside range {lo}-{hi}) replaced with median {fill_val}")
                        df_clean.loc[implausible_mask, col] = fill_val
                    
                    df_clean[f"{col}_was_missing"] = df_clean[col].isna()

            # ---- Age binning with explicit Unknown category ----
            if "Age" in df_clean.columns:
                df_clean["Age"] = pd.to_numeric(df_clean["Age"], errors="coerce")

                age_bins = [0, 19, 25, 30, 35, 40, 50, 120]
                age_labels = [
                    "0-19",
                    "20-25",
                    "26-30",
                    "31-35",
                    "36-40",
                    "41-50",
                    "51+"
                ]

                df_clean["Age_bin"] = pd.cut(
                    df_clean["Age"],
                    bins=age_bins,
                    labels=age_labels,
                    right=True,
                    include_lowest=True
                )

                df_clean["Age_bin"] = df_clean["Age_bin"].astype("object")
                df_clean.loc[df_clean["Age"].isna(), "Age_bin"] = "Unknown"

            # ---- Outcome cleaning with unknowns ----
            if "Outcome" in df_clean.columns:
                df_clean["Outcome"] = df_clean["Outcome"].astype(str).str.strip()
                df_clean["Outcome"] = df_clean["Outcome"].replace({"nan": pd.NA})

                # Map known outcomes
                df_clean["Outcome"] = df_clean["Outcome"].map(outcome_map)

                # Assign unknown numeric value to missing/unknown outcomes
                df_clean["Outcome"] = df_clean["Outcome"].fillna(UNKNOWN_OUTCOME).astype(int)

            # Save cleaned CSV
            base_name = input_path.stem
            if base_name.endswith("_processed"):
                base_name = base_name.replace("_processed", "")

            output_filename = f"{base_name}.csv"
            output_path = CLEANED_DIR / output_filename

            df_clean.to_csv(output_path, index=False)
            print(f"  ✓ Saved successfully to: {output_path.relative_to(ROOT)}\n")
        except Exception as e:
            print(f"❌ Error processing {input_path.name}: {e}\n")


if __name__ == "__main__":
    clean_all_data()
