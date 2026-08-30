import os
import pandas as pd
import hashlib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import re
import warnings


# ========================================================================
# HASHING SALTS: Must be set via environment variables in production
# ========================================================================
# For production deployment, set these environment variables:
#   export MWAKATOBE_SALT_ID=<random-uuid-or-strong-random-string>
#   export MWAKATOBE_SALT_NAME=<random-uuid-or-strong-random-string>
#   export MWAKATOBE_SALT_REGION=<random-uuid-or-strong-random-string>
#   export MWAKATOBE_SALT_DISTRICT=<random-uuid-or-strong-random-string>
#   export MWAKATOBE_SALT_WARD=<random-uuid-or-strong-random-string>
#   export MWAKATOBE_SALT_DIAGNOSES=<random-uuid-or-strong-random-string>
#
# Or use a secret manager (e.g., AWS Secrets Manager, HashiCorp Vault, Azure Key Vault)
# ========================================================================

def _load_salts():
    """Load hashing salts from environment variables with fallback defaults."""
    salts = {
        'id': os.getenv('MWAKATOBE_SALT_ID', 'dev-salt-placeholder-do-not-use-in-production'),
        'name': os.getenv('MWAKATOBE_SALT_NAME', 'dev-salt-placeholder-do-not-use-in-production'),
        'region': os.getenv('MWAKATOBE_SALT_REGION', 'dev-salt-placeholder-do-not-use-in-production'),
        'district': os.getenv('MWAKATOBE_SALT_DISTRICT', 'dev-salt-placeholder-do-not-use-in-production'),
        'ward': os.getenv('MWAKATOBE_SALT_WARD', 'dev-salt-placeholder-do-not-use-in-production'),
        'diagnoses': os.getenv('MWAKATOBE_SALT_DIAGNOSES', 'dev-salt-placeholder-do-not-use-in-production'),
    }
    
    # Warn if using development defaults
    if any('dev-salt-placeholder' in v for v in salts.values()):
        warnings.warn(
            "⚠️  SECURITY WARNING: Using development salt placeholders. "
            "For production, set environment variables: MWAKATOBE_SALT_* "
            "with strong random values. Static salts provide only pseudonymisation, not anonymisation.",
            category=UserWarning
        )
    
    return salts

SALTS = _load_salts()

# Reference date for age calculation
REFERENCE_DATE = datetime(2025, 12, 30)


def safe_hash(value, salt_key='id'):
    if pd.isna(value) or str(value).strip() == '':
        return np.nan
    clean = str(value).strip()
    salt = SALTS.get(salt_key, "")
    return hashlib.sha256((salt + clean).encode('utf-8')).hexdigest()


def generalize_clinical(value):
    if pd.isna(value) or str(value).strip().lower() in ['', 'none', 'no', '-', 'null', 'not recorded']:
        return "Not recorded"
    return "Recorded"


def categorize_sponsor(value):
    """Categorize sponsor into GOVERNMENT, CASH or PRIVATE"""
    if pd.isna(value):
        return "Unknown"
    
    val = str(value).strip().upper()
    
    if 'NHIF' in val:
        return "GOVERNMENT"
    elif 'CASH' in val:
        return "CASH"
    else:
        return "PRIVATE"


def parse_age_to_years(value):
    if pd.isna(value) or str(value).strip() == '':
        return np.nan

    value_str = str(value).strip().lower()

    simple_match = re.match(r'^(\d{1,3})\s*(years?|yrs?|y|old)?$', value_str)
    if simple_match:
        return int(simple_match.group(1))

    years_match = re.search(r'(\d+)\s*years?', value_str)
    if years_match:
        return int(years_match.group(1))

    date_formats = [
        '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y',
        '%m-%d-%Y', '%Y/%m/%d', '%d.%m.%Y', '%Y.%m.%d',
        '%b %d, %Y', '%d %b %Y',
    ]

    for fmt in date_formats:
        try:
            birth_date = datetime.strptime(value_str, fmt)
            age = REFERENCE_DATE.year - birth_date.year
            if (REFERENCE_DATE.month, REFERENCE_DATE.day) < (birth_date.month, birth_date.day):
                age -= 1
            if 0 <= age <= 120:
                return age
        except ValueError:
            continue

    number_match = re.search(r'\d{1,3}', value_str)
    if number_match:
        try:
            age = int(number_match.group(0))
            if 0 <= age <= 120:
                return age
        except:
            pass

    return np.nan


def find_column(df, possible_names):
    cols_lower = df.columns.str.lower().str.replace(r'[\s_]', '', regex=True)
    for name in possible_names:
        mask = cols_lower.str.contains(name, na=False)
        if mask.any():
            return df.columns[mask].tolist()[0]
    return None


def preprocess_single_file(raw_file_path, output_dir='./data/processed/'):
    print(f"\nProcessing: {os.path.basename(raw_file_path)}")

    try:
        df = pd.read_csv(raw_file_path)
    except Exception as e:
        print(f"   Failed to read file: {e}")
        raise

    # Column detection
    id_col          = find_column(df, ['id', 'patientid', 'patient_id', 'patientno', 'ptid', 'patientnumber'])
    name_col        = find_column(df, ['name', 'patientname', 'fullname', 'patient_name'])
    gender_col      = find_column(df, ['gender', 'sex'])
    dob_col         = find_column(df, ['dob', 'dateofbirth', 'birthdate', 'date_of_birth', 'birth_date'])
    age_col         = find_column(df, ['age', 'Age', 'years'])
    sponsor_col     = find_column(df, ['sponsor', 'sponsors', 'payment', 'insurance', 'payer'])
    region_col      = find_column(df, ['region'])
    district_col    = find_column(df, ['district', 'dist'])
    ward_col        = find_column(df, ['ward'])
    diagnoses_col   = find_column(df, ['diagnos', 'diagnosis', 'diag', 'icd', 'dx'])
    procedures_col  = find_column(df, ['procedure', 'procedures'])
    medications_col = find_column(df, ['medication', 'medications', 'medic', 'drug'])
    outcome_col     = find_column(df, ['outcome', 'discharge', 'result', 'disposition'])

    # Vitals
    pulse_col = find_column(df, ['pulse', 'heartrate', 'hr'])
    resp_col  = find_column(df, ['resp', 'respiratory', 'rr', 'resprate', 'breaths'])
    temp_col  = find_column(df, ['temp', 'temperature', 'tempc'])
    sys_col   = find_column(df, ['sys', 'systolic', 'sbp'])
    dia_col   = find_column(df, ['dia', 'diastolic', 'dbp'])

    # Final standardized columns
    final_columns = [
        "Id", "Name", "Gender", "Date of birth", "Age", "Sponsor",
        "Region", "District", "Ward",
        "Pulse", "Resp", "Temp", "Sys", "Dia",
        "Diagnoses", "Procedures", "Medications", "Outcome"
    ]

    final_df = pd.DataFrame(columns=final_columns, index=df.index)

    # Fill values
    final_df["Id"]          = df[id_col].apply(safe_hash, args=('id',)) if id_col else np.nan
    final_df["Name"]        = df[name_col].apply(safe_hash, args=('name',)) if name_col else np.nan
    final_df["Gender"]      = df[gender_col].fillna("Unknown") if gender_col else "Unknown"
    final_df["Date of birth"] = df[dob_col].fillna("Unknown") if dob_col else "Unknown"

    # Age calculation
    age_source = dob_col if dob_col else age_col
    if age_source:
        final_df["Age"] = df[age_source].apply(parse_age_to_years)
    else:
        final_df["Age"] = np.nan

    # Sponsor categorization
    if sponsor_col:
        final_df["Sponsor"] = df[sponsor_col].apply(categorize_sponsor)
    else:
        final_df["Sponsor"] = "Unknown"

    final_df["Region"] = df[region_col].fillna("Unknown") if region_col else "Unknown"
    # final_df["Region"]      = df[region_col].apply(safe_hash, args=('region',)) if region_col else np.nan
    final_df["District"]    = df[district_col].apply(safe_hash, args=('district',)) if district_col else np.nan
    final_df["Ward"]        = df[ward_col].apply(safe_hash, args=('ward',)) if ward_col else np.nan

    # Vitals
    for target, source in [
        ("Pulse", pulse_col), ("Resp", resp_col), ("Temp", temp_col),
        ("Sys", sys_col), ("Dia", dia_col)
    ]:
        if source:
            final_df[target] = pd.to_numeric(df[source], errors='coerce')

    # Clinical fields
    final_df["Diagnoses"]   = df[diagnoses_col].apply(safe_hash, args=('diagnoses',)) if diagnoses_col else "Not recorded"
    final_df["Procedures"]  = df[procedures_col].apply(generalize_clinical) if procedures_col else "Not recorded"
    final_df["Medications"] = (df[medications_col].apply(keep_or_not_recorded) if medications_col else "Not recorded")
    final_df["Outcome"]     = df[outcome_col].fillna("Unknown") if outcome_col else "Unknown"

    # Normalization - vitals only
    # vitals_cols = ["Pulse", "Resp", "Temp", "Sys", "Dia"]
    # vitals_present = [col for col in vitals_cols if final_df[col].notna().any()]

    # if vitals_present:
    #     scaler = MinMaxScaler()
    #     final_df[vitals_present] = scaler.fit_transform(final_df[vitals_present].fillna(final_df[vitals_present].median()))
    #     final_df[vitals_present] = final_df[vitals_present].round(3)

    # Fill missing ages with median
    if final_df["Age"].notna().any():
        median_age = final_df["Age"].median()
        final_df["Age"] = final_df["Age"].fillna(median_age).round(0).astype("Int64")

    # Deduplication
    final_df = final_df.drop_duplicates(subset=['Id'], keep='first')

    print(f"   Final shape: {final_df.shape}")
    print(f"   Final columns: {final_df.columns.tolist()}")

    return final_df


def keep_or_not_recorded(value):
    if pd.isna(value) or str(value).strip().lower() in ['', 'none', 'no', '-', 'null', 'not recorded']:
        return "Not recorded"
    return str(value).strip()



def preprocess_all_data():
    raw_dir = './data/raw/'
    processed_dir = './data/processed/'
    os.makedirs(processed_dir, exist_ok=True)

    csv_files = [f for f in os.listdir(raw_dir) if f.lower().endswith('.csv')]

    if not csv_files:
        print("No CSV files found in ./data/raw/")
        return

    successful = []
    failed = []

    for filename in csv_files:
        try:
            processed_df = preprocess_single_file(os.path.join(raw_dir, filename))
            out_name = f"{os.path.splitext(filename)[0]}_processed.csv"
            out_path = os.path.join(processed_dir, out_name)
            processed_df.to_csv(out_path, index=False)
            print(f"   Saved: {out_name}")
            successful.append(filename)
        except Exception as e:
            print(f"   FAILED {filename}: {str(e)}")
            failed.append(filename)

    print("\n" + "="*60)
    print("SUMMARY")
    print(f"Success: {len(successful)} | Failed: {len(failed)}")
    print("="*60)


if __name__ == "__main__":
    preprocess_all_data()