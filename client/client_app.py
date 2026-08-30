import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU

# Use non-GUI backend for Matplotlib
import matplotlib
matplotlib.use("Agg")

import torch
import atexit
import flwr as fl
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from opacus import PrivacyEngine
from opacus.validators import ModuleValidator


from pathlib import Path
from sklearn import metrics
from collections import Counter
from imblearn.over_sampling import SMOTE # type: ignore
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def env_flag(name, default):
    value = os.getenv(name)
    return default if value is None else value.lower() in {"1", "true", "yes"}


USE_DP = env_flag("MWAKATOBE_USE_DP", True)
USE_SMPC = env_flag("MWAKATOBE_USE_SMPC", True)
USE_SMOTE = env_flag("MWAKATOBE_USE_SMOTE", True)
RESULTS_ONLY = env_flag("MWAKATOBE_RESULTS_ONLY", False)

# experiment parameter
DP_SIGMA = 0.002  # noise scale for differential privacy

# =====================================================
# GLOBAL ENCODERS (CRITICAL FOR FEDERATED LEARNING)
# =====================================================
GLOBAL_ENCODERS = None
CAT_COLS = ["Sponsor", "Region", "Procedures", "Medications", "Age_bin"]

# =====================================================
# GLOBAL CONFUSION MATRIX COLLECTOR
# =====================================================
EVALUATION_DONE = set()

GLOBAL_Y_TRUE = []
GLOBAL_Y_PRED = []

# =====================================================
# CONFUSION MATRICES (GENERATE PNG AT FINALIZE)
# =====================================================
CONFUSION_MATRICES = {}

# =====================================================
# DATASET & FEDERATED WEIGHT COLLECTORS
# =====================================================
HOSPITAL_STATS = {}

# =====================================================
# EPOCH / ROUND-WISE HOSPITAL METRICS
# =====================================================
HOSPITAL_HISTORY = {}
HOSPITAL_PREDICTIONS = {}

# =====================================================
# MULTIPLE EXPERIMENT SUPPORT
# =====================================================

NUM_RUNS = int(os.getenv("MWAKATOBE_NUM_RUNS", "10"))

EXPERIMENT_RESULTS = []

CURRENT_RUN = 1

# =====================================================
# LOCAL FEDERATED ROUND COUNTER (CLIENT-SIDE)
# =====================================================
LOCAL_ROUND_COUNTER = {}

# =====================================================
# LOCAL EPOCH-WISE METRICS (PER HOSPITAL)
# =====================================================
HOSPITAL_EPOCH_HISTORY = {}

# =====================================================
# DEFERRED TRAINING LOGS (FOR CLEAN FINAL OUTPUT)
# =====================================================
TRAINING_LOG_BUFFER = []


# =====================================================
# RESET EXPERIMENT STATE
# =====================================================
def reset_experiment():
    """
    Reset all global variables before starting a new
    independent federated learning experiment.
    """

    global GLOBAL_ENCODERS
    global EVALUATION_DONE
    global GLOBAL_Y_TRUE
    global GLOBAL_Y_PRED
    global CONFUSION_MATRICES
    global HOSPITAL_STATS
    global HOSPITAL_HISTORY
    global HOSPITAL_PREDICTIONS
    global LOCAL_ROUND_COUNTER
    global HOSPITAL_EPOCH_HISTORY
    global TRAINING_LOG_BUFFER

    # Re-fit encoders for a fresh experiment
    GLOBAL_ENCODERS = None

    # Evaluation tracking
    EVALUATION_DONE.clear()

    GLOBAL_Y_TRUE.clear()
    GLOBAL_Y_PRED.clear()

    # Confusion matrices
    CONFUSION_MATRICES.clear()

    # Dataset statistics
    HOSPITAL_STATS.clear()

    # Hospital summaries
    HOSPITAL_HISTORY.clear()
    HOSPITAL_PREDICTIONS.clear()

    # Training history
    LOCAL_ROUND_COUNTER.clear()
    HOSPITAL_EPOCH_HISTORY.clear()

    # Console logs
    TRAINING_LOG_BUFFER.clear()

    print("\n" + "=" * 60)
    print("Starting a NEW Federated Learning Experiment")
    print("Previous experiment state cleared successfully.")
    print("=" * 60 + "\n")


# =====================================================
# EXPERIMENT RESULTS
# =====================================================

def set_current_run(run_number: int):
    """
    Store the current experiment number.
    """
    global CURRENT_RUN
    CURRENT_RUN = run_number


def add_experiment_result(metrics: dict):
    """
    Save metrics from one completed experiment.
    """

    global EXPERIMENT_RESULTS

    EXPERIMENT_RESULTS.append({
        "Run": CURRENT_RUN,
        "Accuracy": metrics["Accuracy"],
        "Precision": metrics["Precision"],
        "Recall": metrics["Recall/Sensitivity"],
        "Specificity": metrics["Specificity"],
        "F1": metrics["F1-Score"],
        "FPR": metrics["FPR"],
        "FNR": metrics["FNR"],
    })


def save_experiment_summary(output_dir):
    """
    Save all experiment runs together with
    mean and standard deviation.
    """

    if len(EXPERIMENT_RESULTS) == 0:
        return

    df = pd.DataFrame(EXPERIMENT_RESULTS)

    os.makedirs(output_dir, exist_ok=True)

    # Individual runs
    runs_csv = os.path.join(output_dir, "experiment_runs.csv")
    df.to_csv(runs_csv, index=False)

    # Summary statistics
    metric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "Specificity",
        "F1",
        "FPR",
        "FNR",
    ]

    summary = pd.DataFrame({
        "Metric": metric_columns,
        "Mean": [df[c].mean() for c in metric_columns],
        "Std": [df[c].std() for c in metric_columns],
        "Min": [df[c].min() for c in metric_columns],
        "Max": [df[c].max() for c in metric_columns],
    })

    summary_csv = os.path.join(output_dir, "experiment_summary.csv")
    summary.to_csv(summary_csv, index=False)

    print("\n" + "=" * 60)
    print("MULTIPLE EXPERIMENT SUMMARY")
    print("=" * 60)
    print(summary.round(4))
    print("=" * 60)


# -------------------------------
# Model definition
# -------------------------------
class HospitalModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )


    def forward(self, x):
        if isinstance(x, pd.DataFrame):
            x = torch.tensor(x.values, dtype=torch.float32)
        return self.net(x.float())



def compute_metrics(y_true, y_pred):
    """
    Returns all metrics: Accuracy, Precision, Recall/Sensitivity, Specificity, F1, FPR, FNR
    """

    cm = confusion_matrix(y_true, y_pred)

    if cm.shape != (2, 2):
        return {
            "Accuracy": 0.0,
            "Precision": 0.0,
            "Recall/Sensitivity": 0.0,
            "Specificity": 0.0,
            "F1-Score": 0.0,
            "FPR": 0.0,
            "FNR": 0.0
        }

    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "Accuracy": accuracy * 100,
        "Precision": precision * 100,
        "Recall/Sensitivity": recall * 100,
        "Specificity": specificity * 100,
        "F1-Score": f1 * 100,
        "FPR": fpr * 100,
        "FNR": fnr * 100
    }


# ==============================
# DIFFERENTIAL PRIVACY UTILITIES
# ==============================
def clip_and_add_noise(params, C=5.0, sigma=DP_SIGMA):
    """
    Implements:
    Δ_i^t = clip(Δ_i^t, C) + N(0, σ²I)
    """
    dp_params = []

    for p in params:
        norm = np.linalg.norm(p)
        clipped = p * min(1.0, C / (norm + 1e-8))
        noise = np.random.normal(0, sigma, clipped.shape)
        dp_params.append(clipped + noise)

    return dp_params


# ==============================
# SMPC UTILITIES (SECURE AGG)
# ==============================
def generate_mask(params, seed, scale=1e-3):
    rng = np.random.default_rng(seed)
    return [rng.normal(0, scale, p.shape) for p in params]


def apply_mask(params, masks):
    return [p + m for p, m in zip(params, masks)]


# -------------------------------
# Data loader (FEDERATED SAFE)
# -------------------------------
def load_data(csv_path):
    global GLOBAL_ENCODERS

    df = pd.read_csv(csv_path)

    features = [
        "Age", "Sponsor", "Region", "Pulse", "Resp", "Temp",
        "Sys", "Dia", "Procedures", "Medications", "Age_bin"
    ]
    df = df[features]

    # ---- Fit encoders ONCE globally ----
    if GLOBAL_ENCODERS is None:
        GLOBAL_ENCODERS = {}
        for col in CAT_COLS:
            enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            enc.fit(df[[col]])
            GLOBAL_ENCODERS[col] = enc

    # ---- Apply encoders consistently ----
    for col in CAT_COLS:
        encoded = GLOBAL_ENCODERS[col].transform(df[[col]])
        encoder_df = pd.DataFrame(
            encoded,
            columns=[f"{col}_{i}" for i in range(encoded.shape[1])],
            index=df.index
        )
        df = pd.concat([df.drop(columns=[col]), encoder_df], axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.values)
    X = torch.tensor(X_scaled, dtype=torch.float32)

    # Label
    y = pd.read_csv(csv_path)["Outcome"].astype(int).values
    y = torch.tensor(y, dtype=torch.long)

    return X, y

# -------------------------------
# Global outcome mapping
# -------------------------------
def get_global_outcome_classes(cleaned_dir="data/cleaned"):
    all_classes = set()
    for csv in Path(cleaned_dir).glob("*.csv"):
        if csv.name.endswith("_mapped.csv"):
            continue
        df = pd.read_csv(csv)
        all_classes.update(df["Outcome"].unique())
    return {cls: i for i, cls in enumerate(sorted(all_classes))}

def map_outcomes_to_global(csv_path, class_to_index):
    df = pd.read_csv(csv_path)
    df = df.copy()

    df["Outcome"] = df["Outcome"].astype(str).str.strip()

    mapping = {str(k): int(v) for k, v in class_to_index.items()}
    df["Outcome"] = df["Outcome"].replace(mapping)

    df["Outcome"] = pd.to_numeric(df["Outcome"], errors="coerce").fillna(-1).astype(int)

    mapped_path = csv_path.replace(".csv", "_mapped.csv")
    df.to_csv(mapped_path, index=False)
    return mapped_path

# -------------------------------
# Binary clinical mapping
# -------------------------------
def to_binary(y):
    """
    Binary medical evaluation

    1 = Referral/Death (referred or poor outcome)
    0 = Home (discharged home)
    """
    
    return (y >= 1).astype(int)


def plot_dataset_sizes_by_hospital():
    out = Path("results/global")
    out.mkdir(parents=True, exist_ok=True)

    hospitals = list(HOSPITAL_STATS.keys())
    totals = [
        HOSPITAL_STATS[h]["train"] + HOSPITAL_STATS[h]["test"]
        for h in hospitals
    ]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(hospitals, totals)
    for bar, total in zip(bars, totals):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"n={total:,}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.xlabel("Hospitals")
    plt.ylabel("Number of Samples")
    plt.title("Dataset Size by Hospital")
    plt.ylim(0, max(totals) * 1.12)
    plt.tight_layout()
    plt.savefig(out / "dataset_sizes_by_hospital.png", dpi=300)
    plt.close()


def plot_federated_weights():
    out = Path("results/global")
    out.mkdir(parents=True, exist_ok=True)

    hospitals = list(HOSPITAL_STATS.keys())
    totals = np.array([
        HOSPITAL_STATS[h]["train"] + HOSPITAL_STATS[h]["test"]
        for h in hospitals
    ])

    weights = totals / totals.sum()

    plt.figure(figsize=(6, 6))
    plt.pie(weights, labels=hospitals, autopct="%.2f%%", startangle=90)
    plt.title("Federated Learning Weights (FedAvg)")
    plt.tight_layout()
    plt.savefig(out / "federated_weights.png", dpi=300)
    plt.close()


def plot_sample_distribution():
    out = Path("results/global")
    out.mkdir(parents=True, exist_ok=True)

    if HOSPITAL_STATS:
        train_total = sum(v["train"] for v in HOSPITAL_STATS.values())
        test_total = sum(v["test"] for v in HOSPITAL_STATS.values())
    else:
        cleaned_dir = Path("data/cleaned")
        hospital_counts = []
        for csv_path in sorted(cleaned_dir.glob("Hospital*.csv")):
            df = pd.read_csv(csv_path)
            if df.empty:
                continue
            n = len(df)
            hospital_counts.append((n, int(n * 0.8), n - int(n * 0.8)))

        if not hospital_counts:
            train_total = 0
            test_total = 0
        else:
            train_total = sum(train for _, train, _ in hospital_counts)
            test_total = sum(test for _, _, test in hospital_counts)

    total = train_total + test_total

    labels = ["Training", "Testing", "Total"]
    values = [train_total, test_total, total]

    plt.figure(figsize=(6, 5))
    plt.bar(labels, values, color="tab:blue")
    plt.xlabel("Sample Type")
    plt.ylabel("Number of Samples")
    plt.title("Sample Distribution Comparison Across Hospitals")
    plt.tight_layout()
    plt.savefig(out / "sample_distribution.png", dpi=300)
    plt.close()


def plot_contribution_matrix():
    out = Path("results/global")
    out.mkdir(parents=True, exist_ok=True)

    hospitals = list(HOSPITAL_STATS.keys())
    totals = np.array([
        HOSPITAL_STATS[h]["train"] + HOSPITAL_STATS[h]["test"]
        for h in hospitals
    ])

    weights = totals / totals.sum()
    matrix = weights.reshape(-1, 1)

    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, aspect="auto")
    plt.colorbar(label="Federated Weight")
    plt.yticks(range(len(hospitals)), hospitals)
    plt.xticks([0], ["Weight"])

    for i, w in enumerate(weights):
        plt.text(0, i, f"{w:.3f}", ha="center", va="center")

    plt.title("Hospital Contribution Matrix")
    plt.tight_layout()
    plt.savefig(out / "contribution_matrix.png", dpi=300)
    plt.close()


def plot_epoch_accuracy_per_hospital():
    out = Path("results/epoch/epoch_accuracy")
    out.mkdir(parents=True, exist_ok=True)

    for hospital, data in HOSPITAL_EPOCH_HISTORY.items():
        df = pd.DataFrame(data)

        # Average accuracy per epoch across rounds
        avg = df.groupby("epoch")["accuracy"].mean()

        plt.figure(figsize=(7, 5))
        plt.plot(
            avg.index,
            avg.values,
            marker="o",
            linewidth=1.5,
            alpha=0.9
        )
        plt.fill_between(
            avg.index,
            avg.values,
            alpha=0.25
        )

        plt.xlabel("Training Epochs (Round-Epochs)")
        plt.ylabel("Accuracy (%)")
        plt.ylim(0, 100)
        plt.title(f"Epoch-wise Accuracy (Avg) – {hospital}")
        plt.xticks(avg.index)
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(out / f"{hospital}_epoch_accuracy_avg.png", dpi=300)
        plt.close()


def plot_epoch_loss_per_hospital():
    out = Path("results/epoch/epoch_loss")
    out.mkdir(parents=True, exist_ok=True)

    for hospital, data in HOSPITAL_EPOCH_HISTORY.items():
        df = pd.DataFrame(data)

        # Average loss per epoch across federated rounds
        avg = df.groupby("epoch")["loss"].mean()

        plt.figure(figsize=(7, 5))
        plt.plot(
            avg.index,
            avg.values,
            marker="o",
            linewidth=1.5,
            alpha=0.9
        )
        plt.fill_between(
            avg.index,
            avg.values,
            alpha=0.25
        )

        plt.xlabel("Training Epochs (Round-Epochs)")
        plt.ylabel("Loss")
        plt.title(f"Epoch-wise Loss (Avg) – {hospital}")
        plt.xticks(avg.index)
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(out / f"{hospital}_epoch_loss_avg.png", dpi=300)
        plt.close()

# -------------------------------
# Local training
# -------------------------------
def train_local_model(csv_path, num_classes, epochs=40, lr=0.001):
    X, y = load_data(csv_path)

    print("\n===== CLASS DISTRIBUTION (RAW) =====")
    for cls, cnt in Counter(y.tolist()).items():
        print(f"Class {cls}: {cnt} samples")

    # =========================
    # SPLIT FIRST (SAFER METHODOLOGY)
    # =========================
    # Split BEFORE resampling to ensure evaluation is unbiased
    # Synthetic samples are created only from training data

    X_np = X.numpy()  # convert torch tensor to numpy for SMOTE
    y_np = y.numpy()

    input_size = X.shape[1]

    class_counts = Counter(y_np)
    stratify = y if min(class_counts.values()) >= 2 else None

    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X_np, y_np, test_size=0.2, random_state=42, stratify=stratify
    )

    # =========================
    # APPLY SMOTE ONLY TO TRAINING SET
    # =========================
    # This ensures test set evaluation is unbiased and includes no synthetic samples

    counter = Counter(y_train_np)
    min_count = min(counter.values())

    if USE_SMOTE and min_count > 1:
        k_neighbors = min(min_count - 1, 5)
        smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=k_neighbors)
        X_train_res, y_train_res = smote.fit_resample(X_train_np, y_train_np)
        print("\n===== AFTER SMOTE (TRAINING SET ONLY) =====")
        print(Counter(y_train_res))
    else:
        print("⚠️ SMOTE skipped due to very small class sizes")
        X_train_res, y_train_res = X_train_np, y_train_np

    # Test set remains unmodified (original data, no synthetic samples)
    
    # Convert back to torch
    X_train = torch.tensor(X_train_res, dtype=torch.float32)
    y_train = torch.tensor(y_train_res, dtype=torch.long)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    y_test = torch.tensor(y_test_np, dtype=torch.long)

    print("\n===== TRAIN SET DISTRIBUTION (AFTER SMOTE) =====")
    print(Counter(y_train.tolist()))

    print("\n===== TEST SET DISTRIBUTION (ORIGINAL, NO SYNTHETIC) =====")
    print(Counter(y_test.tolist()))

    model = HospitalModel(input_size, num_classes)

    counts = torch.bincount(y_train, minlength=num_classes).float()
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes

    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()

        pred = out.argmax(dim=1)
        acc = (pred == y_train).float().mean().item()

        print(f"[LOCAL] {epoch+1}/{epochs} Loss={loss.item():.4f} Acc={acc:.4f}")

    return model.state_dict(), X_train, y_train, X_test, y_test

# -------------------------------
# Flower Client
# -------------------------------
class FlowerHospitalClient(fl.client.NumPyClient): # type: ignore
    def __init__(self, csv_path, class_to_index):
        self.hospital_name = Path(csv_path).stem
        LOCAL_ROUND_COUNTER[self.hospital_name] = 0

        if self.hospital_name not in HOSPITAL_EPOCH_HISTORY:
            HOSPITAL_EPOCH_HISTORY[self.hospital_name] = {
                "epoch": [],
                "loss": [],
                "accuracy": []
            }

        if self.hospital_name not in HOSPITAL_HISTORY:
            HOSPITAL_HISTORY[self.hospital_name] = {
                "round": [],
                "loss": [],
                "accuracy": []
            }

        # ---- Outcome mapping FIRST ----
        self.csv_path = map_outcomes_to_global(csv_path, class_to_index)
        self.num_classes = len(class_to_index)

        # ---- Load & split data ----
        state, self.X_train, self.y_train, self.X_test, self.y_test = \
            train_local_model(self.csv_path, self.num_classes)

        # ---- CLASS IMBALANCE FIX (NOW SAFE) ----
        counts = torch.bincount(self.y_train, minlength=self.num_classes).float()
        self.class_weights = 1.0 / (counts + 1e-6)
        self.class_weights = (
            self.class_weights / self.class_weights.sum()
        ) * self.num_classes

        # ---- Dataset stats ----
        HOSPITAL_STATS[self.hospital_name] = {
            "train": len(self.X_train),
            "test": len(self.X_test),
        }

        # ---- Model ----
        self.model = HospitalModel(self.X_train.shape[1], self.num_classes)
        self.model.load_state_dict(state)

    def get_parameters(self, config=None):
        return [v.cpu().numpy() for v in self.model.state_dict().values()]

    def set_parameters(self, params):
        self.model.load_state_dict({
            k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), params)
        }, strict=True)

    def fit(self, params, config):
        self.set_parameters(params)

        criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Federated round number (from server if available)
        round_num = config.get("server_round", "?")

        for epoch in range(1, 31):  # LOCAL EPOCHS
            self.model.train()
            optimizer.zero_grad()

            out = self.model(self.X_train)
            loss = criterion(out, self.y_train)
            loss.backward()
            optimizer.step()

            _, pred = torch.max(out, 1)
            probs = torch.softmax(out, dim=1)
            acc = (pred == self.y_train).float().mean().item() * 100

            # ---- EPOCH-WISE LOGGING ----
            hist = HOSPITAL_EPOCH_HISTORY[self.hospital_name]
            hist["epoch"].append(epoch)
            hist["loss"].append(loss.item())
            hist["accuracy"].append(acc)

            # ---- 🔥 CONSOLE OUTPUT (THIS IS WHAT YOU WANT) ----
            TRAINING_LOG_BUFFER.append({
                "hospital": self.hospital_name,
                "round": round_num,
                "epoch": epoch,
                "accuracy": acc,
                "loss": loss.item()
            })
    
        params = self.get_parameters()

        if USE_DP:
            params = clip_and_add_noise(params, C=5.0, sigma=DP_SIGMA)

        if USE_SMPC:
            round_seed = int(config.get("server_round", 0))
            masks = generate_mask(params, seed=round_seed)
            params = apply_mask(params, masks)

        return params, len(self.X_train), {}


    def evaluate(self, params, config):
        self.set_parameters(params)
        self.model.eval()

        server_round = config.get("server_round", 0)

        with torch.no_grad():

            # ---------------------------------
            # Forward pass
            # ---------------------------------
            out = self.model(self.X_test)

            pred = torch.argmax(out, dim=1)

            # ---------------------------------
            # Binary evaluation
            # ---------------------------------
            y_true_binary = to_binary(
                self.y_test.cpu().numpy()
            )

            y_pred_binary = to_binary(
                pred.cpu().numpy()
            )

            print("\n===== BINARY LABEL DEBUG =====")
            print("Hospital:", self.hospital_name)
            print("Round:", server_round)

            print("Unique y_true:",
                np.unique(y_true_binary))

            print("Unique y_pred:",
                np.unique(y_pred_binary))

            print("y_true counts:")
            print(np.bincount(y_true_binary))

            print("y_pred counts:")
            print(np.bincount(y_pred_binary))

            # ---------------------------------
            # Store predictions
            # ---------------------------------
            HOSPITAL_PREDICTIONS[self.hospital_name] = {
                "y_true": y_true_binary.copy(),
                "y_pred": y_pred_binary.copy()
            }

            # ---------------------------------
            # Confusion matrix
            # ---------------------------------
            cm = confusion_matrix(
                y_true_binary,
                y_pred_binary,
                labels=[0, 1]
            )

            print("\n===== FINAL CONFUSION MATRIX =====")
            print(cm)

            # ---------------------------------
            # Save CSV only
            # ---------------------------------
            cm_dir = Path(
                "results/confusion_matrices"
            )

            cm_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            csv_path = (
                cm_dir /
                f"{self.hospital_name}_round_{server_round}.csv"
            )

            pd.DataFrame(
                cm,
                index=[
                    "Actual Home",
                    "Actual Admitted"
                ],
                columns=[
                    "Pred Home",
                    "Pred Admitted"
                ]
            ).to_csv(csv_path)

            print("Saved CSV:", csv_path)

            # ---------------------------------
            # STORE MATRIX FOR FINALIZE()
            # ---------------------------------
            CONFUSION_MATRICES[self.hospital_name] = cm

        # ---------------------------------
        # Collect global predictions
        # ---------------------------------
        if self.hospital_name not in EVALUATION_DONE:

            GLOBAL_Y_TRUE.extend(
                y_true_binary.tolist()
            )

            GLOBAL_Y_PRED.extend(
                y_pred_binary.tolist()
            )

            EVALUATION_DONE.add(
                self.hospital_name
            )

        # ---------------------------------
        # Loss
        # ---------------------------------
        loss = nn.CrossEntropyLoss()(
            out,
            self.y_test
        ).item()

        # ---------------------------------
        # Accuracy
        # ---------------------------------
        acc = (
            y_true_binary ==
            y_pred_binary
        ).mean() * 100

        # ---------------------------------
        # History
        # ---------------------------------
        hist = HOSPITAL_HISTORY[
            self.hospital_name
        ]

        hist["round"].append(
            server_round
        )

        hist["loss"].append(
            loss
        )

        hist["accuracy"].append(
            acc
        )

        return (
            float(loss),
            len(self.X_test),
            {"accuracy": acc}
        )
    
# -------------------------------
# Start client
# -------------------------------
def start_flower_client(csv_path, class_to_index):
    fl.client.start_numpy_client( # type: ignore
        server_address="127.0.0.1:9090",
        client=FlowerHospitalClient(csv_path, class_to_index)
    )

# -------------------------------
# Final global plot (runs once)
# -------------------------------
@atexit.register
def generate_confusion_matrix_images():
    if RESULTS_ONLY:
        return

    cm_dir = Path(
        "results/confusion_matrices"
    )

    cm_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for hospital, cm in CONFUSION_MATRICES.items():

        plt.figure(figsize=(8, 6))

        plt.imshow(
            cm,
            cmap="Blues"
        )

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):

                plt.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    fontsize=12
                )

        plt.colorbar()

        plt.xticks(
            [0, 1],
            [
                "Pred Home",
                "Pred Admitted"
            ]
        )

        plt.yticks(
            [0, 1],
            [
                "Actual Home",
                "Actual Admitted"
            ]
        )

        plt.xlabel(
            "Predicted Class"
        )

        plt.ylabel(
            "Actual Class"
        )

        plt.title(
            f"Confusion Matrix\n{hospital}"
        )

        plt.tight_layout()

        img_path = (
            cm_dir /
            f"{hospital}_confusion_matrix.png"
        )

        plt.savefig(
            img_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"Saved confusion matrix image: "
            f"{img_path}"
        )

def plot_experiment_comparison():
    global_out = Path("results/global")
    experiments_out = Path("results/experiments")
    for out_dir in (global_out, experiments_out):
        out_dir.mkdir(parents=True, exist_ok=True)

    if not EXPERIMENT_RESULTS:
        return

    df = pd.DataFrame(EXPERIMENT_RESULTS)
    metric_cols = [
        "Accuracy",
        "Precision",
        "Recall",
        "Specificity",
        "F1",
        "FPR",
        "FNR",
    ]

    comparison_csv = experiments_out / "experiment_comparison.csv"
    df[["Run", *metric_cols]].to_csv(comparison_csv, index=False)

    plot_df = df[["Run", *metric_cols]].melt(
        id_vars="Run",
        var_name="Metric",
        value_name="Value",
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, x="Run", y="Value", hue="Metric")
    plt.title("Experiment Comparison")
    plt.xlabel("Experiment Run")
    plt.ylabel("Metric Value")
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(experiments_out / "experiment_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    summary_table = df[["Run", *metric_cols]].copy()
    summary_table.index = [f"Run {r}" for r in summary_table["Run"]]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")
    table = ax.table(
        cellText=summary_table.round(2).values,
        colLabels=summary_table.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.2)
    plt.title("Experiment Summary Table")
    plt.tight_layout()
    plt.savefig(global_out / "experiment_summary_table.png", dpi=300, bbox_inches="tight")
    plt.close()




def finalize():
    assert len(GLOBAL_Y_TRUE) == sum(
        v["test"] for v in HOSPITAL_STATS.values()
    ), "❌ Sample count mismatch – evaluation bug detected"

    if not RESULTS_ONLY:
        plot_dataset_sizes_by_hospital()
        plot_federated_weights()
        plot_sample_distribution()
        plot_contribution_matrix()
        plot_epoch_accuracy_per_hospital()
        plot_epoch_loss_per_hospital()
    print_hospital_summary()
    print_global_summary()
    save_experiment_results()
    if not RESULTS_ONLY:
        plot_experiment_comparison()

    # ================================
    # FINAL CHRONOLOGICAL TRAINING LOG
    # ================================
    print("\n================ TRAINING LOG (CHRONOLOGICAL) ================\n")

    df = pd.DataFrame(TRAINING_LOG_BUFFER)

    if not df.empty:
        df = df.sort_values(by=["round", "hospital", "epoch"])

        for (rnd, hosp), group in df.groupby(["round", "hospital"]):
            print(f"Training at {hosp} (Federated Round {rnd})")
            for _, row in group.iterrows():
                print(
                    f"- {hosp} Epoch {int(row.epoch)}: "
                    f"Acc={row.accuracy:.2f}%, Loss={row.loss:.4f}"
                )
            print()
    else:
        print("⚠️ No training logs collected.")

    print("==============================================================")


def print_hospital_summary():
    summary = []
    
    for hosp in HOSPITAL_HISTORY.keys():
        # Use global predictions for hospital-specific metrics
        mask = [log['hospital']==hosp for log in TRAINING_LOG_BUFFER]
        if not any(mask):
            continue

        # Filter y_true and y_pred for this hospital
        y_true = np.array(GLOBAL_Y_TRUE[:HOSPITAL_STATS[hosp]["test"]])
        y_pred = np.array(GLOBAL_Y_PRED[:HOSPITAL_STATS[hosp]["test"]])

        metrics = compute_metrics(y_true, y_pred)
        loss = np.mean([log["loss"] for log in TRAINING_LOG_BUFFER if log["hospital"]==hosp])

        summary.append({
            "Hospital": hosp,
            **metrics,
            "Loss": loss
        })

    df_summary = pd.DataFrame(summary)
    df_summary = df_summary[[
        "Hospital", "Accuracy", "Precision", "Recall/Sensitivity",
        "Specificity", "F1-Score", "FPR", "FNR", "Loss"
    ]]
    print("\n================ HOSPITAL PERFORMANCE SUMMARY ================\n")
    print(df_summary.to_string(index=False))
    print("==============================================================\n")


def print_global_summary():
    metrics = compute_metrics(GLOBAL_Y_TRUE, GLOBAL_Y_PRED)
    loss = np.mean([log["loss"] for log in TRAINING_LOG_BUFFER])

    df_global = pd.DataFrame([{
        "Hospital": "GLOBAL",
        **metrics,
        "Loss": loss
    }])
    print("\n================ GLOBAL PERFORMANCE SUMMARY ================\n")
    print(df_global.to_string(index=False))
    print("==============================================================\n")


# Experiment Logging CSV
def save_experiment_results():
    out = Path("results/experiments")
    out.mkdir(parents=True, exist_ok=True)

    metrics = compute_metrics(
        GLOBAL_Y_TRUE,
        GLOBAL_Y_PRED
    )

    add_experiment_result(metrics)

    df = pd.DataFrame([{
        "Run": CURRENT_RUN,
        "sigma": DP_SIGMA,
        **metrics
    }])

    df.to_csv(
        out / f"experiment_run_{CURRENT_RUN}_sigma_{DP_SIGMA}.csv",
        index=False
    )



# -------------------------------
# CLI
# -------------------------------
if __name__ == "__main__":
    import sys
    cleaned = Path("data/cleaned")
    class_map = get_global_outcome_classes(cleaned)
    start_flower_client(sys.argv[1], class_map)
