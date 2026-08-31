# Mwakatobe Federated Learning and Privacy-Preserving Healthcare Architecture

## Executive Summary

Mwakatobe is a multi-institutional healthcare machine-learning prototype in which Hospital A, Hospital B, and Hospital C train a shared predictive model without placing their patient-level CSV records on a central server. The system uses Flower to coordinate federated learning, PyTorch for local neural-network training, local preprocessing and anonymisation, weighted Federated Averaging (FedAvg), and experimental client-side parameter perturbation and masking.

The architecture is best described as a **central-coordinator federated learning system with privacy-oriented controls**. Patient rows remain within each hospital's local data boundary during training. The coordinating server distributes the current global model and receives client model parameters, sample counts, and evaluation results. Local training can use Opacus DP-SGD with per-sample gradient clipping, Gaussian noise, and accountant-reported privacy budgets. Cryptographic secure aggregation is not yet enabled and remains a planned hardening activity.

## 1. System Objectives and Boundaries

### Objectives

The proposed system aims to:

- Enable several hospitals to learn a common clinical outcome model.
- Avoid centralising raw patient records.
- Harmonise differently structured hospital datasets.
- Reduce exposure of direct identifiers and detailed clinical fields.
- Address local class imbalance.
- Measure performance separately for each hospital.
- Provide a foundation for fairness and privacy analysis.

### Trust boundaries

There are three important boundaries:

1. **Hospital boundary:** each institution owns and processes its local patient data.
2. **Communication boundary:** model parameters move between clients and the Flower server.
3. **Coordinator boundary:** the server coordinates rounds and aggregates updates but is not assumed to have access to raw patient rows.

A production deployment would also need authenticated, encrypted client-server communication, access control, audit logs, secret management, and an explicit threat model. The current local configuration uses `127.0.0.1:9090`, which is suitable for experimentation but not a distributed clinical deployment.

## 2. High-Level Architecture

```mermaid
flowchart LR
	subgraph H[Hospital client repeated at each institution]
		A[Private patient records] --> B[Preprocess and harmonise]
		B --> C[Encode features and split data]
		C --> D[Train local PyTorch model]
		D --> E[Clip, add noise, and mask update]
		D --> F[Evaluate on local test data]
	    E -->|Protected model update only| S[Flower coordinator]
	    S -->|FedAvg global model| D
	    F -->|Aggregate metrics only| R[Experiment results]
	    K[Other hospitals use the same client workflow] --> S
    end
```

This is a representative view: every participating hospital uses the same client workflow shown inside the hospital boundary. The coordinator receives protected model updates and selected aggregate metrics, never the private patient records. `K` represents the total number of participating institutions.

**Plain-text flow:** Private patient records -> local preprocessing -> local training -> update protection -> Flower/FedAvg coordinator -> global model returned to the hospital. Raw patient rows remain inside each hospital.

The architecture has two complementary pipelines:

- **Data pipeline:** raw hospital files are transformed into cleaned, harmonised local datasets.
- **Model pipeline:** local clients train from the current global model, transform outgoing parameters, and participate in server aggregation.

## 3. End-to-End Execution Flow

The entry point is [main.py](main.py). Its execution sequence is:

1. Disable CUDA for the experiment and force CPU execution.
2. Preprocess raw hospital files with `preprocess_all_data()`.
3. Clean and harmonise the processed files with `clean_all_data()`.
4. Discover the shared outcome-class mapping.
5. Start a Flower server subprocess.
6. Start one client thread for each cleaned hospital CSV.
7. Run the configured federated rounds.
8. Wait for all clients to finish.
9. Generate per-run results, figures, and summaries.
10. Stop the server and begin the next experiment run.

The main experiment runner repeats this process for `NUM_RUNS` runs. The generated outputs are placed under `results/`.

### Federated round sequence

```mermaid
sequenceDiagram
	participant S as Flower server
	participant A as Hospital A client
	participant B as Hospital B client
	participant C as Hospital C client
	S->>A: Send global model parameters
	S->>B: Send global model parameters
	S->>C: Send global model parameters
	A->>A: Train locally for 30 epochs
	B->>B: Train locally for 30 epochs
	C->>C: Train locally for 30 epochs
	A->>S: Parameters + local sample count
	B->>S: Parameters + local sample count
	C->>S: Parameters + local sample count
	S->>S: Weighted FedAvg aggregation
	S-->>A: Updated global model
	S-->>B: Updated global model
	S-->>C: Updated global model
	A->>S: Local evaluation metrics
	B->>S: Local evaluation metrics
	C->>S: Local evaluation metrics
```

For client $k$ with $n_k$ training examples, the intended FedAvg update is:

$$
w_{t+1} = \sum_{k=1}^{K}\frac{n_k}{\sum_{j=1}^{K}n_j}w_{t+1}^{(k)}
$$

where $w_{t+1}^{(k)}$ is the locally trained parameter set returned by client $k$.

## 4. Data Engineering and Anonymisation

### 4.1 Raw-to-cleaned transformation

The preprocessing layer in [client/data_utils.py](client/data_utils.py) detects and standardises source columns, parses ages, categorises sponsors, generalises clinical fields, hashes selected values, and removes duplicate records. The cleaning layer in [client/clean_app.py](client/clean_app.py) removes direct identity and diagnosis columns, imputes vital signs, creates age bins, and maps outcomes.

The resulting feature set used by clients includes:

- `Age`
- `Sponsor`
- `Region`
- `Pulse`
- `Resp`
- `Temp`
- `Sys`
- `Dia`
- `Procedures`
- `Medications`
- `Age_bin`

The current source inventory contains 15,208 raw records and 13,329 records after identifier-based deduplication: Hospital A contributes 10,569 records, Hospital B 990, and Hospital C 1,770. The final record contribution is therefore 79.29%, 7.43%, and 13.28%, respectively. The values 16,212 and 14,818 mentioned in an earlier manuscript draft do not occur in the current repository and cannot be verified against this data version.

### 4.2 Data-quality and missingness heterogeneity

The raw hospital exports have different schemas and different levels of feature availability. Hospitals A and B contain 17 source columns, including five vital-sign fields, while Hospital C contains 12 source columns and does not provide those vital-sign fields. The following feature-level missingness results were calculated directly from the raw CSV files. Blank values and explicit missing-value markers (`NA`, `None`, `Unknown`, `Not recorded`, and `-`) are counted as missing.

| Hospital | Records | Source columns | Overall missing cells | Features with missing values |
|---|---:|---:|---:|---:|
| Hospital A | 12,000 | 17 | 5.76% | Pulse, Resp, Temp, Sys, Dia, Diagnoses, Procedures, Medications |
| Hospital B | 1,379 | 17 | 2.59% | Pulse, Resp, Temp, Sys, Dia, Diagnoses, Procedures, Outcome |
| Hospital C | 1,829 | 12 | 7.95% | Ordered Tests, Procedures, Outcome |

The largest raw missingness source is `Diagnoses` for Hospital A (84.29%), `Diagnoses` for Hospital B (30.82%), and `Ordered Tests` for Hospital C (94.59%). Hospital C's missing vital-sign fields are structural missingness caused by absent source columns rather than blank cells. This demonstrates that the manuscript's feature-availability claim should distinguish unavailable columns from missing cells; the reported 14-feature/17.65% and 9-feature/47.06% values cannot be reproduced from the current CSV schemas using the cell-level definition above.

### 4.2.1 Detailed path from raw data to local training

The complete hospital-side path is local. A hospital starts with its own raw CSV and does not upload that file to the coordinator. The file is read by [client/data_utils.py](client/data_utils.py), transformed into a standard schema, written to `data/processed/`, cleaned by [client/clean_app.py](client/clean_app.py), and then loaded by [client/client_app.py](client/client_app.py) for model training.

```mermaid
flowchart TD
    A[Hospital raw CSV] --> B[Detect source columns]
    B --> C[Standardise schema]
    C --> D[Hash or generalise sensitive fields]
    D --> E[Parse age and categorise sponsor]
    E --> F[Save processed CSV locally]
    F --> G[Drop unused identifiers and diagnosis]
    G --> H[Impute missing vital signs]
    H --> I[Create Age_bin and map Outcome]
    I --> J[Encode categorical features]
    J --> K[Standardise feature matrix]
    K --> L[Apply local balancing]
    L --> M[Split local train and test data]
    M --> N[Train local PyTorch model]
```

#### Step 1: discover non-uniform source columns

Hospital exports may use names such as `patient_id`, `patientno`, `insurance`, `heartrate`, `systolic`, or `discharge`. The implementation normalises column names by removing spaces and underscores, then searches for known aliases:

```python
def find_column(df, possible_names):
	cols_lower = df.columns.str.lower().str.replace(
		r'[\s_]', '', regex=True
	)
	for name in possible_names:
		mask = cols_lower.str.contains(name, na=False)
		if mask.any():
			return df.columns[mask].tolist()[0]
	return None
```

This allows different hospital schemas to be mapped into the common fields `Id`, `Age`, `Sponsor`, `Region`, the five vital signs, clinical fields, and `Outcome`. Missing source columns are assigned documented defaults such as `Unknown`, `Not recorded`, or an empty numeric column rather than causing the entire federation to stop.

#### Step 2: transform age values

The `parse_age_to_years()` function accepts an existing age, a value containing years, or a date of birth. For a birth date $b$ and reference date $r$, the intended calculation is:

$$
\mathrm{age} = r_{year} - b_{year} - \mathbf{1}[(r_{month},r_{day}) < (b_{month},b_{day})]
$$

Values outside the accepted range of 0 to 120 become missing. If valid ages exist, missing ages are filled with the local median:

$$
x_{\text{age}}^{*} = \begin{cases}
x_{\text{age}}, & \text{if age is present}\\
\mathrm{median}(\mathrm{Age}_{\text{local}}), & \text{otherwise}
\end{cases}
$$

The exact age is also grouped into `Age_bin`, using ranges `0-19`, `20-25`, `26-30`, `31-35`, `36-40`, `41-50`, `51+`, and `Unknown`. Age binning reduces precision before categorical encoding.

#### Step 3: pseudonymise and generalise fields

The preprocessing code hashes identifiers and selected sensitive values before saving the processed file:

```python
final_df["Id"] = (
	df[id_col].apply(safe_hash, args=("id",))
	if id_col else np.nan
)
final_df["Name"] = (
	df[name_col].apply(safe_hash, args=("name",))
	if name_col else np.nan
)
final_df["Diagnoses"] = (
	df[diagnoses_col].apply(safe_hash, args=("diagnoses",))
	if diagnoses_col else "Not recorded"
)
```

Procedures are reduced to whether a value was recorded. Medication values are retained only in the local processed file and are later selected or generalised according to the client feature pipeline. The cleaning stage subsequently removes `Id`, `Name`, `Gender`, date of birth, ward, district, and diagnoses from the model input.

#### Step 4: impute clinical measurements

For each available vital-sign column, the cleaning code converts values to numeric form. Missing values are replaced by the hospital's local median. If an entire column is missing, a documented clinical default is used:

| Variable | Default if the complete column is missing |
|---|---:|
| Pulse | 70 |
| Resp | 16 |
| Temp | 37 |
| Sys | 120 |
| Dia | 80 |

For a vital-sign value $x_i$ in hospital $h$, the imputation rule is:

$$
x_{i,h}^{*} = \begin{cases}
x_{i,h}, & x_{i,h}\text{ is observed}\\
\mathrm{median}(X_h), & X_h\text{ contains observed values}\\
d, & X_h\text{ is completely missing}
\end{cases}
$$

The implementation also creates `*_was_missing` columns during cleaning. The active `load_data()` feature list selects the clinical variables, categorical variables, and `Age_bin` used by the model.

#### Step 5: create the common outcome labels

The cleaning layer maps local outcome strings into the shared label space:

```python
outcome_map = {
	"Home": 0,                  # Patient discharged home
	"Referral": 1,             # Patient referred to another facility
	"Death": 2                 # Patient died
}
unknown_outcome = 3            # Missing or unrecorded outcome

df_clean["Outcome"] = (
	df_clean["Outcome"].astype(str).str.strip().map(outcome_map)
	.fillna(unknown_outcome).astype(int)
)
```

The client then scans the cleaned hospital files and creates a global mapping so every client uses the same output positions. If hospital $k$ has local labels $y_k$, the global mapping applies one shared function $g(y_k)$ before training:

$$
y_{k}^{global} = g(y_k) \in \{0,1,2,3\}
$$

This is essential for FedAvg because the fourth output of a model must represent the same class at every hospital.

#### Step 6: encode categorical features locally

The client selects the exact input columns:

```python
features = [
	"Age", "Sponsor", "Region", "Pulse", "Resp", "Temp",
	"Sys", "Dia", "Procedures", "Medications", "Age_bin"
]
df = df[features]
```

The categorical columns are `Sponsor`, `Region`, `Procedures`, `Medications`, and `Age_bin`. Each is converted into one-hot indicator columns using `OneHotEncoder(handle_unknown="ignore")`. For category value $c$ in field $j$, one-hot encoding produces:

$$
\phi_j(c) = [\mathbf{1}(c=c_1),\mathbf{1}(c=c_2),\ldots,\mathbf{1}(c=c_m)]
$$

The encoded categorical columns are concatenated with numeric age and vital-sign columns. Unknown categories are ignored rather than raising an exception. 

✅ **FIXED:** The encoders are now fitted deterministically on combined data from all hospitals (in sorted order) and cached to disk (`data/encoders.pkl`). This ensures reproducible feature dimensions across all experimental runs. The feature dimension is **3,794 total features**:
- Medications: 3,749 categories (one-hot)
- Region: 23 categories
- Sponsor: 3 categories
- Procedures: 2 categories
- Age_bin: 7 categories
- Numeric: 6 (Age + 5 vital signs)

#### Step 7: standardise the model matrix

After encoding, the client applies `StandardScaler` to the complete feature matrix:

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.values)
X = torch.tensor(X_scaled, dtype=torch.float32)
```

For feature $j$, standardisation is:

$$
z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j + \varepsilon}
$$

where $\mu_j$ and $\sigma_j$ are calculated from the hospital's local matrix. This helps the neural network optimise across variables with different units, although local fitting means the same clinical value may receive different scales at different hospitals. A production federation should agree on a safe global or population-independent scaling policy.

#### Step 8: split the local dataset and perform class balancing

The implementation now correctly uses an 80/20 split FIRST, then applies SMOTE only to the training partition for unbiased evaluation:

```python
# Split FIRST (CORRECT ORDER for unbiased evaluation)
X_train, X_test, y_train, y_test = train_test_split(
	X, y,
	test_size=0.2,
	round_state=42,
	stratify=stratify if min(class_counts.values()) >= 2 else None,
)

# Apply SMOTE ONLY to training partition
counter = Counter(y_train_np)
min_count = min(counter.values())

if USE_SMOTE and min_count > 1:
	k_neighbors = min(min_count - 1, 5)
	smote = SMOTE(
		sampling_strategy="auto",
		random_state=42,
		k_neighbors=k_neighbors,
	)
	X_train_res, y_train_res = smote.fit_resample(X_train_np, y_train_np)
else:
	X_train_res, y_train_res = X_train_np, y_train_np
```

This methodology ensures:
1. Test set remains completely unbiased (contains only original data)
2. Synthetic samples are created only from training data
3. No information leakage from test set to training
4. Evaluation metrics reflect real-world generalization

SMOTE synthesises a minority observation between a sample $x_i$ and one of its minority neighbours $x_{nn}$:

$$
x_{new} = x_i + \lambda(x_{nn} - x_i), \qquad \lambda \sim U(0,1)
$$

The operation is performed locally, so synthetic patient-like records are not uploaded.

The test partition remains at the hospital and is used for local evaluation after federated rounds. It is not sent to the server.

#### Step 10: compute class weights and train locally

For training counts $n_c$ in class $c$, the client computes inverse-frequency weights:

$$
\hat{w}_c = \frac{1}{n_c + 10^{-6}}, \qquad
w_c = \hat{w}_c\frac{C}{\sum_{r=1}^{C}\hat{w}_r}
$$

where $C$ is the number of output classes. The initial local model uses weighted cross-entropy with label smoothing:

$$
\mathcal{L} = -\sum_{i=1}^{N}w_{y_i}\sum_{c=1}^{C}q_{ic}\log p_{ic}
$$

where $q_{ic}$ is the smoothed target and $p_{ic}$ is the model softmax probability. The code then performs full-batch Adam optimisation:

```python
counts = torch.bincount(y_train, minlength=num_classes).float()
weights = 1.0 / (counts + 1e-6)
weights = weights / weights.sum() * num_classes

criterion = nn.CrossEntropyLoss(
	weight=weights,
	label_smoothing=0.05,
)
optimizer = optim.Adam(model.parameters(), lr=0.0005)

for epoch in range(epochs):
	model.train()
	optimizer.zero_grad()
	out = model(X_train)
	loss = criterion(out, y_train)
	loss.backward()
	optimizer.step()
```

This produces the locally trained state dictionary used to initialise the hospital client. In each later Flower round, the client receives the current global parameters, trains for 30 local epochs using weighted cross-entropy, and returns the resulting model parameters after the configured privacy transformations.

### 4.3 Effects of preprocessing

The following before-versus-after results compare each raw hospital CSV with its corresponding cleaned CSV. Duplicate records before cleaning are repeated source identifiers; the after value is zero because those duplicates are removed during preprocessing. Missingness before cleaning is the percentage of missing cells across the raw source columns. Missingness after cleaning is calculated across the 11 active model features and treats `Not recorded` as missing information. Invalid values are nonnumeric or implausible age/vital-sign values. Inconsistent categories are distinct raw sponsor and outcome labels outside the canonical categories used by the cleaner.

| Metric | A before | A after | B before | B after | C before | C after |
|---|---:|---:|---:|---:|---:|---:|
| Records | 12,000 | 10,569 | 1,379 | 990 | 1,829 | 1,770 |
| Duplicate records removed | 1,431 | 0 | 389 | 0 | 59 | 0 |
| Missing values (%) | 5.76% | 0.06% | 2.59% | 0.02% | 7.95% | 9.10% |
| Usable features | 10 | 11 | 10 | 11 | 5 | 11 |
| Invalid values | 82 | 0 | 10 | 0 | 0 | 0 |
| Inconsistent categories | 23 | 0 | 11 | 0 | 5 | 0 |

Preprocessing removes 1,431 records from Hospital A, 389 from Hospital B, and 59 from Hospital C because their source identifiers are duplicated. It standardises the usable model input to 11 features: `Age`, `Sponsor`, `Region`, five vital signs, `Procedures`, `Medications`, and `Age_bin`. Missing or nonnumeric vital-sign values are coerced and imputed. Implausible numeric vital-sign values (outside clinical ranges) are also replaced with the local median:

- **Pulse:** 30–200 bpm
- **Respiratory rate:** 5–50 breaths/min
- **Temperature:** 33–43°C
- **Systolic BP:** 50–250 mmHg
- **Diastolic BP:** 30–150 mmHg

This range-filtering process removed 74 implausible values from Hospital A (7 Pulse, 28 Resp, 25 Temp, 14 Dia) and 6 from Hospital B (2 Pulse, 2 Resp, 2 Dia), resulting in 0 invalid values in all cleaned datasets. Sponsor labels are mapped to `GOVERNMENT`, `CASH`, or `PRIVATE`. Hospital C remains at 9.10% semantic missingness after cleaning because `Medications` is absent from its source schema and is represented as `Not recorded` for all cleaned records; this is structural missingness, not missing-cell recovery.

After preprocessing with the corrected outcome mapping, the outcome distributions are:

- **Hospital A** (10,569 total): Home/class 0 = 10,553 (99.85%), Referral/class 1 = 8 (0.08%), Death/class 2 = 8 (0.08%), Unknown/class 3 = 0
- **Hospital B** (990 total): Home/class 0 = 978 (98.79%), Referral/class 1 = 1 (0.10%), Death/class 2 = 3 (0.30%), Unknown/class 3 = 8 (0.81%)
- **Hospital C** (1,770 total): Home/class 0 = 1,750 (98.87%), Referral/class 1 = 2 (0.11%), Death/class 2 = 6 (0.34%), Unknown/class 3 = 12 (0.68%)

All source outcome values (`Home`, `Referral`, `Death`, `NaN`) are now correctly mapped to their clinical classes.

### 4.4 Outcome harmonisation

All institutions use the same label space:

| Outcome | Numeric class | Clinical meaning |
|---|---:|---|
| Home | 0 | Patient discharged home |
| Referral | 1 | Patient referred to another facility |
| Death | 2 | Patient died during stay |
| Missing or unknown | 3 | Unrecorded or NaN outcome |


The shared mapping is important because model outputs from different hospitals must represent the same clinical categories.

### 4.4.1 Common target and encoded input

The intended common target definition is a four-class integer outcome: `Home=0`, `Referral=1`, `Death=2`, and missing or unknown=`3`. The current cleaned datasets contain all four classes as verified above. The final common model feature list contains 11 logical features. Numeric inputs contribute six dimensions (`Age` and five vital signs); categorical inputs are one-hot encoded for `Sponsor`, `Region`, `Procedures`, `Medications`, and `Age_bin`.

✅ **FIXED:** The encoder implementation is now deterministic. Encoders are fitted on combined data from all hospitals in a fixed order and cached to disk (`data/encoders.pkl`). All experimental runs use the same encoder vocabulary and produce **3,794 encoded input dimensions** reproducibly.

### 4.5 Hashing and generalisation code

The project's current salted hashing routine is:

```python
def safe_hash(value, salt_key='id'):
	if pd.isna(value) or str(value).strip() == '':
		return np.nan
	clean = str(value).strip()
	salt = SALTS.get(salt_key, "")
	return hashlib.sha256((salt + clean).encode('utf-8')).hexdigest()
```

Clinical detail is also coarsened where possible:

```python
def generalize_clinical(value):
	if pd.isna(value) or str(value).strip().lower() in [
		'', 'none', 'no', '-', 'null', 'not recorded'
	]:
		return "Not recorded"
	return "Recorded"
```

This reduces direct exposure, but hashing is pseudonymisation rather than guaranteed anonymisation. Static salts remain in the repository and should be moved to protected, institution-specific environment variables or a secret manager before deployment.

## 5. Federated Client Architecture

Each hospital client is implemented by `FlowerHospitalClient` in [client/client_app.py](client/client_app.py). A client reads its own hospital file, selects and encodes features, scales data, handles local imbalance, constructs the model, receives global parameters, trains locally, applies privacy transformations, returns parameters and sample count, and evaluates on its local test set.

### Model code

```python
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
```

During federated training, the client uses weighted cross-entropy and performs local optimisation:

```python
criterion = nn.CrossEntropyLoss(weight=self.class_weights)
optimizer = optim.Adam(self.model.parameters(), lr=0.001)

for epoch in range(1, 31):
	self.model.train()
	optimizer.zero_grad()
	output = self.model(self.X_train)
	loss = criterion(output, self.y_train)
	loss.backward()
	optimizer.step()
```

Local SMOTE and class weighting are intended to reduce local class imbalance. SMOTE should ideally be applied only to the training partition; applying it before the train/test split can allow synthetic information to influence evaluation.

### Class imbalance and privacy-preserving learning roadmap

Healthcare outcomes are often imbalanced: a small minority class can be clinically important even when it represents relatively few records. The project should evaluate balancing methods inside each hospital client, after the local train/test split, so that the test partition remains an untouched estimate of generalisation.

The recommended methods are:

1. **SMOTE:** ✅ **FIXED** — now applied only to training partition after split, ensuring unbiased test evaluation.
2. **ADASYN:** create more synthetic examples near difficult or sparsely represented minority observations. ADASYN is a candidate alternative to SMOTE and should be evaluated locally without transmitting the synthetic records.
3. **Class-weighted optimisation:** assign larger loss weights to minority classes. The current client uses weighted cross-entropy, so this is the currently implemented algorithm-level approach.
4. **Focal loss:** reduce the contribution of easy examples and focus optimisation on difficult or misclassified cases. This is a future alternative to weighted cross-entropy, not currently active in the pipeline.

Balancing is compatible with federated learning because the resampling or loss calculation occurs inside each hospital. Synthetic records must still remain local; they must never be uploaded to the coordinator. Experiments should compare no balancing, SMOTE, ADASYN, class weighting, and focal loss using macro-F1, minority recall, balanced accuracy, and calibration, in addition to ordinary accuracy.

Local training uses Opacus DP-SGD by default. Opacus applies per-sample gradient clipping and calibrated Gaussian noise during optimisation, then reports the cumulative $(\epsilon, \delta)$ budget for each hospital. The resulting values are saved under `results/privacy/`. Cryptographic secure aggregation is separate from DP-SGD and is not enabled in the current legacy Flower `NumPyClient` deployment; Flower SecAgg/SecAgg+ must be configured before making that claim.

## 6. Server and Aggregation Architecture

The active server is in [server/server_app.py](server/server_app.py):

```python
import flwr as fl

class FederatedAveraging(fl.server.strategy.FedAvg):
	"""Standard weighted FedAvg; secure aggregation is not enabled here."""

if __name__ == "__main__":
	fl.server.start_server(
		server_address="127.0.0.1:9090",
		config=fl.server.ServerConfig(num_rounds=10),
		strategy=FederatedAveraging(
			on_fit_config_fn=lambda server_round: {"server_round": server_round}
		),
	)
```

The server performs model coordination and FedAvg aggregation. It does not receive raw patient rows. The separate [server/strategy.py](server/strategy.py) defines another FedAvg configuration, but it is not imported by the active `server_app.py`.

## 7. Privacy-Preserving Design

Privacy is implemented as a layered design rather than a single algorithm. Data minimisation and generalisation reduce sensitive detail before training, federated learning keeps patient rows at the institution, and optional DP-SGD limits the influence of individual training records. Cryptographic secure aggregation is not enabled in the active server configuration.

| Algorithm or control | Operation | Intended protection | Current status |
|---|---|---|---|
| Data minimisation | Remove fields not needed by the model | Reduces sensitive information processed | Implemented |
| Salted SHA-256 | Hash `salt + value` | Prevents direct identifier transmission | Implemented; static salts need replacement |
| Generalisation | Replace detailed fields with broad categories | Reduces re-identification risk | Partially implemented |
| Age binning | Convert exact ages into ranges | Reduces quasi-identifier precision | Implemented |
| Federated learning | Train at each hospital and exchange parameters | Prevents routine central collection of patient rows | Implemented |
| Opacus DP-SGD | Per-sample clipping and Gaussian-noised gradients | Bounds individual-record influence during local training | Implemented; privacy budget logged per hospital |
| Cryptographic secure aggregation | Mask updates so only their aggregate is revealed | Prevents coordinator inspection of an individual update | Not enabled; requires Flower SecAgg/SecAgg+ deployment |
| TLS and authentication | Encrypt and authenticate network traffic | Protects updates in transit | Required for deployment |

### 7.1 Local data protection and raw-data non-sharing

Each hospital client reads its own CSV, performs preprocessing locally, constructs tensors locally, and trains locally. The server receives model-related values rather than the original patient table.

```mermaid
flowchart TD
	H[Hospital data store] -->|Raw rows remain inside hospital| L[Hospital client]
	L -->|Parameters, sample count, selected metrics| S[Flower server]
	S -->|Global model parameters| L
	S -.->|No raw patient CSV transfer| H
```

During a normal federated round, these remain inside the institution:

- Patient identifiers and demographic rows.
- Raw clinical measurements and original categorical text.
- Local feature matrices, labels, train/test partitions, and intermediate gradients.
- Local predictions used for hospital-level evaluation.

The following may leave the institution:

- Model parameter arrays returned by `get_parameters()` and `fit()`.
- The local training sample count used for weighted FedAvg.
- Evaluation values such as loss, accuracy, and other metrics.
- Experiment metadata and logs generated by the orchestration process.

Federated learning prevents direct raw-row sharing, but model updates can still leak information through membership inference, gradient analysis, or model inversion. “Raw data not shared” is therefore a data-flow property, not proof that the trained model reveals nothing about patients.

### 7.2 Algorithm 1: salted hashing and pseudonymisation

For an input value $x$ and secret salt $s$, the client computes:

$$
h = \mathrm{SHA256}(s || \mathrm{normalise}(x))
$$

where $\mathbin{||}$ means concatenation. The same salt and value produce the same digest, which supports local duplicate detection without transmitting the original value. Hashing is not encryption: low-entropy values may still be guessed if the salt is exposed or an attacker can test candidate values.

### 7.3 Algorithm 2: local generalisation and data minimisation

Before training, the pipeline drops direct identifiers, converts age into ranges, groups sponsor values, replaces selected clinical text with `Recorded` or `Not recorded`, imputes missing vitals, and deduplicates available records. This is data minimisation and generalisation, not formal $k$-anonymity. A formal assessment should calculate the size of every equivalence class formed by quasi-identifiers such as age bin, region, and sponsor.

### 7.4 Algorithm 3: federated local training and FedAvg

At round $t$, the server broadcasts global parameters $w_t$. Hospital $k$ trains locally and returns $w_{t+1}^{(k)}$. The server calculates:

$$
w_{t+1} = \sum_{k=1}^{K}\frac{n_k}{\sum_{j=1}^{K}n_j}w_{t+1}^{(k)}
$$

where $n_k$ is the local training count. The server does not need patient rows for this calculation, although model parameters, update sizes, and metrics can still leak information.

### 7.5 Algorithm 4: DP-SGD with privacy accounting

```python
privacy_engine = PrivacyEngine(accountant="rdp")
model, optimizer, train_loader = privacy_engine.make_private(
	module=model,
	optimizer=optimizer,
	data_loader=train_loader,
	noise_multiplier=DP_NOISE_MULTIPLIER,
	max_grad_norm=DP_MAX_GRAD_NORM,
)

epsilon = privacy_engine.get_epsilon(DP_DELTA)
```

For each per-sample gradient $g_i$ and clipping threshold $C$:

$$
\bar{g}_i = g_i\min\left(1, \frac{C}{||g_i||_2 + 10^{-8}}\right)
$$

The noisy batch gradient is:

$$
	ilde{g} = \frac{1}{B}\left(\sum_{i=1}^{B}\bar{g}_i + \mathcal{N}(0,\sigma^2C^2I)\right)
$$

The active path uses Opacus `PrivacyEngine(accountant="rdp")`, Poisson-sampled minibatches, per-sample clipping, and Gaussian noise. Default values are `MWAKATOBE_DP_NOISE_MULTIPLIER=1.1`, `MWAKATOBE_DP_MAX_GRAD_NORM=1.0`, and `MWAKATOBE_DP_DELTA=1e-5`; each client writes its measured epsilon to `results/privacy/<hospital>_privacy_budget.csv`. The reported $(\epsilon, \delta)$ guarantee is valid only under Opacus' sampling and threat-model assumptions and must be reported with the run configuration.

### 7.6 Cryptographic secure aggregation status

The prior custom additive-mask prototype is disabled. It did not provide cryptographic secure aggregation because it did not use authenticated key exchange, pairwise complementary masks, or dropout recovery. The active server therefore performs standard weighted FedAvg and must not be described as secure aggregation.

For a future true additive secure-aggregation deployment, clients need masks $r_k$ satisfying:

$$
\sum_{k=1}^{K}r_k = 0
$$

so that:

$$
\sum_{k=1}^{K}(w_k+r_k)=\sum_{k=1}^{K}w_k
$$

Production secure aggregation requires a vetted protocol such as Flower SecAgg/SecAgg+, authenticated key exchange, dropout recovery, minimum-participant thresholds, and controlled unmasking. Until it is deployed and tested, the coordinator can inspect individual model updates.

### 7.7 Privacy threat model and residual risks

The architecture primarily reduces exposure to a central coordinator that should not receive raw hospital tables. It does not by itself fully protect against malicious clients, a curious server analysing unmasked updates, membership inference, model inversion, re-identification from rare quasi-identifiers, leakage through logs or temporary files, or collusion between the coordinator and hospitals.

Privacy verification should inspect both code and operations: capture network payloads to confirm that no CSV rows are sent, check logs for identifiers and feature values, validate that result files contain only approved aggregates, and test the trained model for membership or inversion leakage where appropriate.

## 8. Evaluation, Results, and Fairness

The system saves numerical and visual outputs under `results/`.

### Dataset and contribution figures

**Table 2: Hospital statistics of the dataset contributed by participating hospitals**

The dataset-size figure and hospital summary use the cleaned local records that are loaded by the clients:

| Hospital | Records | Features | Missing<br>(%) | Classes<br>Outcome |
|---|---:|---:|---:|---:|
| Hospital A | 10,569 | 11 | 0.06 | 2 |
| Hospital B | 990 | 11 | 0.02 | 2 |
| Hospital C | 1,770 | 11 | 9.10 | 2 |

`Features` counts the model input columns before one-hot encoding. `Missing (%)` is measured across those 11 model inputs after local cleaning; `Not recorded` is treated as missing information and identifier fields are removed before training. Hospital C's 9.10% is structural missingness because `Medications` is absent from its source schema.

![Dataset size by hospital](results/global/dataset_sizes_by_hospital.png)

![Federated weights](results/global/federated_weights.png)

![Sample distribution](results/global/sample_distribution.png)

![Contribution matrix](results/global/contribution_matrix.png)

![Missing value analysis and preprocessing strategy](results/global/missing_value_analysis.png)

The missing-value table is generated from all processed hospital records with `tools/generate_missing_value_table.py` before the local cleaning and imputation step.

![Privacy and SMOTE results](results/global/privacy_smote_results.png)

The privacy and SMOTE results table is generated from the recorded experiment summaries with `tools/generate_privacy_smote_table.py`. The comparison configurations were run one by one in isolated copies, so the existing project graphs were not regenerated.

### Hospital-level confusion matrices

The current evaluation converts the four-class outcome into a binary task: any outcome other than Home (class 0) is positive, representing escalated care (referral or death). With the corrected outcome mapping, the cleaned files now contain all four classes. The binary definition is: class 0 (`Home`) vs. classes 1–3 (referral, death, or unknown outcome).

![Hospital A confusion matrix](results/confusion_matrices/HospitalA_confusion_matrix.png)

![Hospital B confusion matrix](results/confusion_matrices/HospitalB_confusion_matrix.png)

![Hospital C confusion matrix](results/confusion_matrices/HospitalC_confusion_matrix.png)

Numeric matrices are available in `results/confusion_matrices/`. **Note:** The stored confusion-matrix images and CSVs were generated with an earlier outcome mapping and may not match the current cleaned data distributions. These are historical artifacts and should be regenerated after a complete federated learning run with the corrected outcome mapping.

### Training dynamics

![Hospital A accuracy](results/epoch/epoch_accuracy/HospitalA_epoch_accuracy_avg.png)

![Hospital B accuracy](results/epoch/epoch_accuracy/HospitalB_epoch_accuracy_avg.png)

![Hospital C accuracy](results/epoch/epoch_accuracy/HospitalC_epoch_accuracy_avg.png)

![Hospital A loss](results/epoch/epoch_loss/HospitalA_epoch_loss_avg.png)

![Hospital B loss](results/epoch/epoch_loss/HospitalB_epoch_loss_avg.png)

![Hospital C loss](results/epoch/epoch_loss/HospitalC_epoch_loss_avg.png)

### Experiment comparison

![Experiment comparison](results/experiments/experiment_comparison.png)

![Experiment summary table](results/global/experiment_summary_table.png)

The primary numerical sources are `results/experiment_runs.csv`, `results/experiment_summary.csv`, and the per-run files under `results/experiments/`.

### Metrics and fairness

The implementation calculates accuracy, precision, recall/sensitivity, specificity, F1 score, false-positive rate, false-negative rate, local training loss and accuracy, and per-hospital confusion matrices. The fairness utilities can support sponsor-stratified analysis, but demographic parity, equal opportunity, subgroup recall, and hospital-level disparity metrics are not yet fully integrated into the main federated run.

## 9. Security and Methodological Caveats

| Area | Current state | Required strengthening |
|---|---|---|
| Raw-data locality | Implemented in the client workflow | Add deployment controls and audit verification |
| Identifier hashing | ✅ **FIXED:** Environment variables (MWAKATOBE_SALT_*) | Migrate to secret manager |
| Data generalisation | Partially implemented | Perform formal re-identification and k-anonymity checks |
| Federated aggregation | Weighted FedAvg implemented | Secure transport, authentication, and coordinator hardening |
| Differential privacy | Opacus DP-SGD with accountant-reported budgets | Validate settings and publish per-run $(\epsilon, \delta)$ |
| Secure aggregation | Not enabled | Deploy Flower SecAgg/SecAgg+ with dropout handling |
| Feature encoding | ✅ **FIXED:** Deterministic, cached (3,794 dims) | Distribute versioned bundle with experiment |
| Evaluation | Binary metrics from four-class labels | Report multiclass metrics and document binary reduction |
| SMOTE methodology | ✅ **FIXED:** Split first, then resample training only | Evaluate alternative resampling methods |
| Vital signs validation | ✅ **FIXED:** Range filtering applied (80 → 0 invalid) | Monitor for implausible values in deployment |
| Fairness | Utilities exist but are not central to the run | Add subgroup and hospital disparity reporting |

## 10. Recommended Production Architecture

1. Run one isolated client service per hospital rather than sharing a filesystem process.
2. Use TLS, certificate-based client authentication, and server authorization.
3. Remove static salts and store secrets in a managed vault.
4. Distribute a versioned encoder vocabulary and feature order.
5. Validate Opacus DP-SGD settings and publish privacy budgets.
6. Deploy Flower SecAgg/SecAgg+ with dropout handling.
7. Add client validation, anomaly detection, and robust aggregation.
8. Keep an untouched local test partition and apply SMOTE only after splitting.
9. Calculate subgroup recall, false-positive rate, demographic parity, and equal opportunity.
10. Record consent, data-sharing agreements, ethics approval, model versions, audit events, and rollback procedures.

## 11. Public Repository Transparency Package

For an openly available research repository, the following items provide transparency and reproducibility while respecting restrictions on confidential human data:

- **Anonymisation code:** hashing, generalisation, identifier removal, and secret-management guidance.
- **Preprocessing code:** schema harmonisation, missing-value handling, age binning, outcome mapping, and quality checks.
- **Federated-learning code:** client training, server orchestration, FedAvg configuration, privacy transformations, and evaluation.
- **Synthetic or example data:** small non-identifying fixtures that demonstrate the pipeline without representing real patients. Never publish confidential human records.
- **Data dictionary:** field definitions, data types, allowed values, transformations, and clinical meaning where appropriate.
- **Aggregate statistics:** per-hospital counts, missingness summaries, class distributions, and aggregate performance only.
- **Experiment configuration:** rounds, local epochs, learning rate, client participation, random seeds, balancing method, DP settings, and model version.
- **Reproducibility scripts:** commands or scripts that recreate preprocessing, training, evaluation, tables, and figures from approved example data.

The repository should explicitly document which artifacts are withheld because of confidentiality. In this project, raw, processed, and cleaned patient CSV files remain excluded through `.gitignore`; generated PNG figures may be published because they are aggregate visual outputs, but every figure should still be reviewed for disclosure risk. Secrets such as hashing salts must never be included in a public repository.

## 12. Reproducibility

From the repository root:

```powershell
python main.py
```

Manual component execution:

```powershell
python client/data_utils.py
python client/clean_app.py
python server/server_app.py
python client/client_app.py data/cleaned/HospitalA.csv
python client/client_app.py data/cleaned/HospitalB.csv
python client/client_app.py data/cleaned/HospitalC.csv
```

The exact dependencies are listed in [requirements.txt](requirements.txt). Before reporting results, record the code revision, Python and dependency versions, privacy flags, clipping norm, noise scale, number of rounds, local epochs, data-splitting policy, and experiment seed policy.

## Conclusion

Mwakatobe establishes the main architectural pattern required for privacy-conscious multi-institutional healthcare learning: data stays at the hospital, local clients train a common model, and a coordinator aggregates model information rather than patient rows. The project also includes preprocessing, anonymisation, imbalance handling, hospital-level evaluation, fairness utilities, and result visualisation.

The current system should be presented as a research prototype. Its core federated workflow and accountant-tracked DP-SGD path are operational. Cryptographic secure aggregation, independent privacy validation, and deployment controls remain engineering tasks. This distinction separates the demonstrated data-locality and DP-SGD configuration from privacy guarantees requiring additional cryptographic implementation and operational validation.