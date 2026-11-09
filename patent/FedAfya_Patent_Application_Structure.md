# FedAfya Patent Application — Structure Matching the Current Draft

> **Working structure, not an official BRELA template or legal opinion.** This outline follows the section order and labels in [the current implementation-evidence application](./FedAfya_Patent_Application_Implementation_Evidence.docx). It is meant to help us complete that document, not replace or reorganize it. Confirm the prescribed BRELA forms, filing format, and legal requirements with BRELA and Tanzanian patent counsel.

## Application front matter

Keep these items before the numbered sections, as in the current draft:

- **Document title:** PATENT APPLICATION — TECHNICAL SPECIFICATION
- **Invention title:** FedAfya: Privacy-Aware Federated Health-Data Processing Method
- **Draft/status note:** Identify it as a draft for review; do not imply novelty, patentability, grant, or deployment has been established.

## A. Applicant and ownership information

### A.1 Title of the Invention

Use the same concise title throughout the application, request form, and related materials.

### A.2 Inventor and Particulars

List only the inventor(s) who contributed to the conception of the claimed invention. For each, confirm the official full name and all particulars requested by the current BRELA form. Do not list a person solely because they supervised, funded, supplied data, or implemented instructions.

**To complete:** confirm inventorship claim by claim, affiliations, addresses, contact particulars, and any required declarations directly with the relevant people.

### A.3 Owner of the Invention

State the proposed applicant/owner and address only after confirming entitlement under institutional policy, employment terms, assignments, funding terms, and relevant third-party agreements.

**To complete:** obtain NM-AIST's ownership determination and verify any assignments or contractual rights. Do not treat this draft statement as an established ownership finding.

## B. Technical specification

### B.1 Technical Field

Briefly identify the technical area of the invention. Keep this consistent with the actual subject matter of the claims.

**FedAfya draft topic:** computer-assisted processing of heterogeneous electronic healthcare records and distributed predictive-model training.

### B.2 Background of the Invention

Describe the relevant technical context and the problem that exists in the field. Explain only facts needed to understand the claimed invention; distinguish general context from features alleged to be new.

### B.3 Known Prior Art

Identify the closest relevant patent and non-patent references. For each reference, provide enough bibliographic information to locate it and explain which technical features it discloses.

**To complete:** conduct and document a patent/database and literature search; prepare a feature-by-feature comparison. The papers and methods mentioned in the current draft are search leads, not a completed prior-art search or novelty opinion.

**Suggested working comparison table (complete only after searching and reading the sources):**

| Reference (full citation / publication no.) | Disclosed technical features | Features relevant to proposed claims | Difference, if supported by the reference | Evidence / search notes |
|---|---|---|---|---|
| [Reference 1] | [Summarize what it actually teaches] | [Map to claim feature(s)] | [Specific difference, or “none identified”] | [URL, database, date searched] |
| [Reference 2] | [Summarize what it actually teaches] | [Map to claim feature(s)] | [Specific difference, or “none identified”] | [URL, database, date searched] |

Do not fill the “difference” column from an abstract or title alone. Avoid asserting that an individual component (for example federated averaging, differential privacy, secure aggregation, or schema harmonization) is new without a claim-focused search.

### B.4 Technical Problem to Be Addressed

State the specific technical problem the claimed subject matter addresses. Avoid defining the problem only as a desired result, such as better healthcare, privacy, or model performance.

**Possible draft wording for discussion (not inserted into the application DOCX):**

> In a distributed learning workflow using healthcare records from heterogeneous source systems, differences in field names, value conventions, and feature availability can cause input representations to be inconsistent across sources. A preprocessing workflow therefore needs to identify and handle those differences before records are used to prepare model inputs. The specific technical problem addressed by the claimed invention remains to be finalized by comparison with the closest prior art and confirmation of the implemented technical features.

Keep the final statement narrower if the prior art already addresses some of these issues. Do not frame unverified privacy protection or clinical benefit as the technical problem solved.

### B.5 Technical Solution

Summarize the specific technical steps or system components relied on in the claims, and how they work together to address the stated problem.

- Use the same terminology and sequence as the claims and detailed description.
- Distinguish required features from optional embodiments.
- Do not describe a proposed feature as implemented, tested, or effective unless the evidence supports that statement.
- Identify the precise technical difference from the closest prior art only after the comparison is complete.

**Suggested compact summary structure (fill after the claims and prior-art review; not inserted into the application DOCX):**

> In one embodiment, a computing system receives [defined source records], identifies [defined schema/provenance information], maps [specified source fields] according to [specified mapping rules], and produces [defined output representation]. The system [retains / does not yet retain] [precisely defined missingness or availability state] and [applies / does not yet apply] [a defined quality decision]. Where included, [training or update-handling step] is performed according to [specified protocol]. The technical distinction relied upon is [claim-specific difference supported by the prior-art comparison].

If a concise equation helps explain the data flow, use it only after the variables and operations are defined in B.6. For example, the following is a **proposed abstraction**, not a description of verified current behavior:

$$
R_h \xrightarrow{\;M_v,\;Q\;} X_h
$$

Here, `R_h` is the input from source `h`, `M_v` is a specified mapping procedure at version `v`, `Q` is any defined quality decision, and `X_h` is the output representation. Define what happens when `Q` fails; do not imply that the current evaluation metrics enforce a gate.

### B.6 Detailed Description / Manner of Carrying Out

Describe the invention in enough detail for a technically skilled reader to understand how to carry it out. Keep the numbered step structure and reference numerals in the current application. For each step, cite the matching drawing component and, where useful, a short equation and an exact code location. Equations describe the technical operation; code citations document the inspected repository. Neither a code citation nor a drawing substitutes for a complete written description.

#### Step-by-step description and evidence references

1. **Step 1 — Hospital-controlled clinical data source (100)**
   - **Describe:** the data source, authorized processing environment, input records, and data-flow boundary.
   - **Figure:** FIG. 1 component 100; FIG. 2 for the illustrative sites and data-flow boundary.
   - **Code/evidence:** [`main.py`](../main.py) orchestrates clients using CSV files in the research workspace. This demonstrates the current experiment arrangement, not physical hospital-local deployment.
   - **Equation:** none is needed unless the invention defines a concrete input-selection or boundary rule.
   - **Code excerpt — experiment orchestration:**

     ```python
     server = subprocess.Popen(
         [sys.executable, "server/server_app.py"]
     )
     for csv in cleaned_csvs:
         t = threading.Thread(
             target=start_flower_client,
             args=(csv.as_posix(), class_to_index),
             daemon=True,
         )
         t.start()
     ```

     This starts a server process and client threads that read cleaned CSVs available to the research host. It does not show clients running inside hospital-controlled environments.

2. **Step 2 — Source-schema and provenance capture (110)**
   - **Describe:** which source fields and metadata are captured, how site/source identity is represented, and how schema versions are generated and retained.
   - **Figure:** FIG. 1 component 110; include capture outputs and their relationship to the mapping step.
   - **Code/evidence:** [`find_column()` and `preprocess_single_file()`](../client/data_utils.py) provide heuristic column detection and normalization. Do not describe this as a versioned provenance registry: one was not identified in the inspected production path.
   - **Possible notation:** `S_h` may denote the schema observed for source `h`, and `v_h` its version, if the final implementation actually creates and stores these values.
   - **Code excerpt — source-column matching:**

     ```python
     def find_column(df, possible_names):
         cols_lower = df.columns.str.lower().str.replace(
             r"[\s_]", "", regex=True
         )
         for name in possible_names:
             mask = cols_lower.str.contains(name, na=False)
             if mask.any():
                 return df.columns[mask].tolist()[0]
         return None
     ```

     This selects the first column matching a normalized name pattern. It is not a versioned semantic mapping or provenance capture mechanism.

3. **Step 3 — Versioned semantic mapping and validation (120)**
   - **Describe:** the source-to-target mapping rules, value/type/unit validation, handling of ambiguous or unmapped values, and how rule versions are applied.
   - **Figure:** FIG. 1 component 120; show rejected, flagged, or mapped values only if that behavior exists.
   - **Code/evidence:** [`find_column()`](../client/data_utils.py) and the `outcome_map` in [`clean_app.py`](../client/clean_app.py) show current heuristic field matching and explicit outcome coding. The comparison-only [`make_b1_frame()`](../evaluation/preprocessing_comparison.py) applies evaluator mappings; do not present it as the production mapping service.
   - **Code excerpt — current outcome mapping:**

     ```python
     outcome_map = {
         "Home": 0,
         "Referral": 1,
         "Death": 2,
     }
     df_clean["Outcome"] = df_clean["Outcome"].map(outcome_map)
     df_clean["Outcome"] = (
         df_clean["Outcome"].fillna(UNKNOWN_OUTCOME).astype(int)
     )
     ```

     This is a fixed label mapping in the cleaner. It does not implement a versioned mapping registry.
   - **Possible equation — proposed mapping notation:**

     $$
     y_{ij}=f_{j,v}(x_{ij})
     $$

     Here, $x_{ij}$ is a source value, $f_{j,v}$ is a defined mapping rule at version $v$, and $y_{ij}$ is the mapped value. Define the rule and version mechanism in words; do not use this notation to imply versioning that is not implemented.

4. **Step 4 — Observed/missing/structurally unavailable state encoding (130)**
   - **Describe:** define each state at record and source-schema level, show how the representation preserves it through imputation and training, and explain how a structurally absent feature differs from a missing value in an existing field.
   - **Figure:** FIG. 1 component 130; a small state-transition example can show observed, missing, structurally unavailable, and (if applicable) imputed-with-original-state-retained.
   - **Code/evidence:** in [`clean_app.py`](../client/clean_app.py), vital-sign imputation occurs before `*_was_missing` is assigned, so the current flag does not retain pre-imputation missingness. [`source_feature_status()` and `make_b1_frame()`](../evaluation/preprocessing_comparison.py) assess structural availability in the evaluation harness only.
   - **Code excerpt — current order in the cleaner:**

     ```python
     df_clean[col] = df_clean[col].fillna(fill_val)
     # ...
     df_clean[f"{col}_was_missing"] = df_clean[col].isna()
     ```

     Since the flag is calculated after filling, this excerpt documents the current limitation rather than the intended preservation behavior.
   - **Possible equations — proposed state definitions:**

     $$
     M_{ij}=\mathbf{1}[x_{ij}\text{ is missing before imputation}]
     \qquad
     A_{hj}=\mathbf{1}[\text{feature }j\text{ exists in source }h]
     $$

     These definitions are proposed for the description, not evidence that the production model currently consumes both indicators.

5. **Step 5 — Data-quality and update eligibility gate (140)**
   - **Describe:** each measured quality indicator, its denominator, any predeclared threshold, gate logic, and the action taken when a check fails (for example flag, withhold, or reject).
   - **Figure:** FIG. 1 component 140; show both pass and fail paths if a gate is part of the actual embodiment.
   - **Code/evidence:** [`feature_metrics()` and `evaluate_hospital()`](../evaluation/preprocessing_comparison.py) calculate descriptive evaluation metrics. The inspected `main.py` flow does not use them to block data, clients, or updates.
   - **Code excerpt — metrics reported by the evaluation harness:**

     ```python
     valid_rate = (
         round(valid_count / valid_denominator, 4)
         if valid_denominator else None
     )
     retention_rate = (
         round(retained_count / retention_denominator, 4)
         if retention_denominator else None
     )
     ```

     These values are reported for comparison; they are not used by the main pipeline as an eligibility gate.
   - **Possible equations — descriptive rates and a proposed gate:**

     $$
     V=\frac{N_{\mathrm{valid}}}{N_{\mathrm{assessed}}}
     \qquad
     R=\frac{N_{\mathrm{retained}}}{N_{\mathrm{assessed}}}
     $$

     A future gate might be expressed as:

     $$
     G=\mathbf{1}[V\geq\tau_V\land R\geq\tau_R]
     $$

     Use the gate equation only after the indicators, thresholds, and enforcement behavior are defined and implemented. It is not current behavior.

6. **Step 6 — Local model-training operation (150)**
   - **Describe:** the input representation, model/task, training procedure, train/test separation, and the actual execution location.
   - **Figure:** FIG. 1 component 150; FIG. 2 may show where records remain and where computation occurs, but only if verified.
   - **Code/evidence:** [`load_data()` and `train_local_model()`](../client/client_app.py) show client preprocessing, splitting, and training preparation. `StandardScaler` is fitted before the train/test split, and separately for each loaded client dataset; describe this accurately when reporting the current experiment.
   - **Code excerpt — feature scaling and split order:**

     ```python
     scaler = StandardScaler()
     X_scaled = scaler.fit_transform(df.values)
     X = torch.tensor(X_scaled, dtype=torch.float32)

     X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
         X_np, y_np, test_size=0.2, random_state=42, stratify=stratify
     )
     ```

     The scaler is fitted before the split, so its statistics include rows later assigned to the test set. This is a limitation to disclose when describing current experiments.
   - **Equation — current standardization:**

     $$
     z_{ij}=\frac{x_{ij}-\mu_{hj}}{\sigma_{hj}}
     $$

     Here, $\mu_{hj}$ and $\sigma_{hj}$ are the mean and standard deviation estimated from client $h$'s loaded feature matrix. If describing SMOTE, specify that it is applied to the training split only in the inspected path; its interpolation is:

     $$
     x_{\mathrm{new}}=x_a+\lambda(x_b-x_a),
     \qquad \lambda\sim U(0,1)
     $$

7. **Step 7 — Conditional privacy control for model updates (160)**
   - **Describe:** which mechanism is applied, where it is applied, its configuration/accounting, and the conditions under which it is enabled. Distinguish local differential privacy mechanisms from secure aggregation.
   - **Figure:** FIG. 1 component 160 should be marked optional only if the embodiment truly makes it optional; show the mechanism's position relative to local training and aggregation.
   - **Code/evidence:** [`FlowerHospitalClient`](../client/client_app.py) configures an Opacus `PrivacyEngine` when `MWAKATOBE_USE_DP` is true (default true unless overridden), and records epsilon/delta values. Enabling `MWAKATOBE_USE_SMPC` raises an error in the inspected client path; this is not secure aggregation.
   - **Code excerpt — conditional Opacus setup:**

     ```python
     if USE_DP:
         self.privacy_engine = PrivacyEngine(accountant="rdp")
         self.model, self.optimizer, self.train_loader = (
             self.privacy_engine.make_private(
                 module=self.model,
                 optimizer=self.optimizer,
                 data_loader=self.train_loader,
                 noise_multiplier=DP_NOISE_MULTIPLIER,
                 max_grad_norm=DP_MAX_GRAD_NORM,
             )
         )
     ```

     Confirm the run-specific flag, privacy accounting, and configuration before describing a particular experiment as differentially private.
   - **Equation:** include a mechanism-specific equation only after confirming the exact clipping, noise, sampling, and accounting definition for the embodiment. Do not infer a formal privacy guarantee from the presence of a library call or a single epsilon/delta log.

8. **Step 8 — Federated aggregation coordinator (170)**
   - **Describe:** the aggregation algorithm, participating updates, weights, validation, and handling of absent or rejected participants.
   - **Figure:** FIG. 1 component 170; FIG. 2 may show model-update communications but must not imply that updates are intrinsically private.
   - **Code/evidence:** [`get_strategy()`](../server/strategy.py) and [`FederatedAveraging`](../server/server_app.py) use standard Flower FedAvg. The client returns `len(self.X_train)` as its aggregation weight; this can count SMOTE-generated examples.
   - **Code excerpt — server strategy and client result:**

     ```python
     return fl.server.strategy.FedAvg(
         fraction_fit=1.0,
         fraction_eval=1.0,
         min_fit_clients=3,
         min_eval_clients=3,
         min_available_clients=3,
     )

     # Client fit result
     return self.get_parameters(), len(self.X_train), privacy_metrics
     ```

     This is standard weighted FedAvg; the inspected server does not configure secure aggregation.
   - **Equation — weighted FedAvg:**

     $$
     \theta_{t+1}
     =
     \sum_{h=1}^{H}
     \frac{n_h}{\sum_{r=1}^{H}n_r}
     \theta_{h,t+1}
     $$

     Here, $n_h$ is the weight sent by client $h$. In the current implementation it is `len(self.X_train)`, which may include SMOTE-generated examples; do not describe it as the count of original patient records.

9. **Step 9 — Model-version audit and feedback (180)**
   - **Describe:** each audit record, model/schema/mapping version, participation decision, evaluation output, and feedback or suspension behavior.
   - **Figure:** FIG. 1 component 180; show the actual record/output path and any return path.
   - **Code/evidence:** [`client_app.py`](../client/client_app.py) writes experiment metrics and, when the privacy engine runs, a privacy-budget CSV. A complete schema/mapping version trail and eligibility-decision log were not established in the inspected path.
   - **Code excerpt — privacy accounting record:**

     ```python
     epsilon = self.privacy_engine.get_epsilon(DP_DELTA)
     privacy_metrics = {"dp_epsilon": epsilon, "dp_delta": DP_DELTA}
     pd.DataFrame([{
         "hospital": self.hospital_name,
         "round": round_num,
         "epsilon": epsilon,
         "delta": DP_DELTA,
         "noise_multiplier": DP_NOISE_MULTIPLIER,
         "max_grad_norm": DP_MAX_GRAD_NORM,
     }]).to_csv(privacy_path, mode="a", index=False)
     ```

     This excerpt is conditional on the privacy engine being enabled. It does not establish a complete schema/mapping-version or eligibility-decision audit trail.
   - **Possible record form — proposed only:** `L = (model_version, source_schema_version, mapping_version, decision, metrics)`. Include only fields the embodiment actually records and define how identifiers and sensitive values are protected.

10. **Example embodiment and technical evidence**
    - Give reproducible input, configuration, processing sequence, output, and test procedure.
    - Cite the experiment records and state whether each item is production-path, evaluation-only, or a proposed embodiment.
    - Include baselines, metrics, and limitations; do not characterize a one-run evaluation as repeatability evidence.
    - The existing [implementation map](./FedAfya_Figure_Implementation_Map.png) and [evaluation snapshot](./FedAfya_Figure_Evaluation_Snapshot.png) are evidence/overview figures. They are currently FIG. 3/FIG. 4 in Annex C; decide with counsel whether they belong in the formal patent drawing set before renumbering.

#### How to present code snippets

Use a short, exact excerpt only when it clarifies a technical operation. Introduce it with the source file/function and explain each relevant input, transformation, and output in prose. Preserve enough surrounding description that the invention is understandable without the repository. Label existing implementation, evaluation-only code, and proposed/pseudocode separately. Review snippets for secrets, personal data, third-party code, and material not intended for public disclosure.

#### How to cross-reference figures and equations

- Use the same FIG. numbers and reference numerals in B.6, D, E, and F.
- Place a figure reference in the step where its component or data flow is first explained.
- Define every equation symbol immediately and explain the operation in words.
- Make equations match the described embodiment and claims; mark illustrative/proposed equations as such until verified.
- Keep evidence figures separate from formal patent drawings unless counsel decides they form part of the application drawing set.

**Current implementation/evidence status:** The present DOCX's Step 3–9 descriptions and Annex C identify observed gaps. In particular, the production missingness flag is computed after imputation, the inspected main path has no threshold-based eligibility gate, hospital-local deployment is not verified, and the server uses ordinary FedAvg without secure aggregation. Preserve these caveats unless and until the implementation and evidence change.

## C. Proposed Claims (for counsel review)

Claims define the protection sought. Keep them in this section and in the same numbering/order as the application draft.

- Draft an independent claim around the specific, supported technical combination.
- Use dependent claims for narrower embodiments that are fully described.
- Ensure that each claimed feature is supported in Section B.6 and in any referenced drawings.
- Use consistent terms, clear claim dependencies, and concrete technical limitations.
- Remove or revise unsupported features; do not claim optional privacy or deployment functions as present unless they are disclosed and evidenced.
- Have Tanzanian patent counsel review scope, clarity, subject-matter eligibility, unity, and formal claim requirements.

**Current status:** Claims in the DOCX are proposals only. The method's novelty and the claims' support have not been established by the current code review or preprocessing comparison.

**Useful internal claim-support table (not claim language):**

| Proposed claim | Supporting description | Drawing reference | Implementation / evidence status | Open issue |
|---|---|---|---|---|
| Claim 1 | B.5; B.6 Steps [ ] | FIG. [ ] | [Implemented / evaluation-only / proposed] | [Support, prior art, or clarity question] |
| Claim 2 | B.6 Step [ ] | FIG. [ ] | [Implemented / evaluation-only / proposed] | [Open issue] |

Use this table to check support and consistency; keep it in working notes unless counsel wants it included in the application package.

## D. Brief Description of Drawings

List each figure once, in numerical order, with a short explanation of what it depicts. Match the reference numerals in Sections B.6, E, and F.

**Current draft:** FIG. 1 describes the method flow; FIG. 2 depicts the conceptual multi-hospital arrangement. Revise the descriptions if the drawings or verified implementation change.

**Suggested description wording (confirm against the final drawings; not inserted into the application DOCX):**

- **FIG. 1** is a flow diagram illustrating an embodiment of a data-processing and model-training method, including source records (100), source-schema information (110), field mapping (120), state handling (130), a quality decision (140), local training (150), optional update handling (160), aggregation (170), and audit/feedback (180).
- **FIG. 2** is a schematic illustration of multiple healthcare data environments and a coordinating server, showing the record and model-update paths for an embodiment.

Retain steps in a figure only when they are described and supported. Label proposed or unverified deployment elements as illustrative in the description, and do not use figures to assert that a feature is implemented.

## E. Detailed Description of Drawings

Explain each drawing in order, identify its components/reference numerals, and describe the connections or sequence shown. This section should help the reader understand the embodiments; it should not add unsupported features that are absent from Section B.6 or the claims.

**Suggested writing pattern (not inserted into the application DOCX):**

> In FIG. 1, source records (100) are provided to [component or operation] (110). The source information is processed by [mapping operation] (120) to produce [defined output]. [State handling (130) and quality decision (140)] are included only as described for the particular embodiment. If the quality decision is not passed, [state the actual action]. The resulting input is used by [training operation (150)]. [Optional operation (160)] is performed only when configured as described. The coordinating operation (170) combines [defined inputs] and the audit operation (180) records [defined information].

Replace each bracketed item with the actual disclosed operation. Do not describe an intended pass/fail branch, local-only data boundary, secure aggregation, or audit record unless B.6 explains how it works.

## F. Drawings

Include the drawings identified in Section D with legible labels and consistent reference numerals. Confirm BRELA's required drawing format and whether sheets/files must be submitted separately.

For the current conceptual figures:

- **FIG. 1:** method sequence using reference numerals 100–180.
- **FIG. 2:** illustrative three-hospital arrangement.

The drawings must not imply that physical hospital-local training, a functioning quality gate, or a privacy protocol has been verified where it has not.

## G. Abstract

Provide a concise summary consistent with the final description and claims. It should identify the technical field, problem, essential solution, and use without arguing merits or making speculative performance or privacy claims.

**Current draft:** the abstract is labeled with a 150-word maximum. Verify the current BRELA form/rules and confirm the final abstract word count after the claims and description are settled.

**Abstract drafting frame (replace bracketed text; not inserted into the application DOCX):**

> The disclosure relates to [technical field]. It addresses [specific technical problem]. A method [state the essential, supported processing steps] to produce [technical output]. In an embodiment, [include a supported optional feature, if essential to understanding]. The method is for [technical use]. 

Keep the abstract aligned with the final independent claim. Do not include an unsupported novelty statement, performance result, privacy guarantee, or details better placed in B.6.

## Annexes in the current application draft

These are supporting material in the current DOCX structure. Confirm with counsel whether each belongs in the filed specification, should be a separate attachment, or should remain in the internal review copy.

### Annex A — Patent Drawings (conceptual)

Include or cross-reference the conceptual drawings, with an explicit note where a diagram illustrates a proposed embodiment rather than a verified deployment.

### Annex B — Code and Evidence Cross-Reference

Map each described feature to the relevant source files, experiments, and evidence. Record limitations and configuration-dependent behavior. Do not include confidential code or sensitive data without authorization and a disclosure review.

### Annex C — Implementation Steps, Code Excerpts, and Measured Evidence

Use this annex to make the description traceable to the inspected implementation and experiments:

- C.1 Source schema discovery and mapping
- C.2 Missing values and structural availability
- C.3 Quality measurements (identify whether they are descriptive or enforced)
- C.4 Reported preprocessing comparison
- C.5 Model input, training, and aggregation
- C.6 Conditional differential privacy and ordinary FedAvg
- C.7 Figures and source cross-reference

Label equations, code excerpts, and measured results with their source, run configuration, and limitations. Do not present an audit/evaluation script as production behavior.

## Separate filing-package checklist — confirm with BRELA

This checklist is separate from the A–G specification above. It is preparatory only and does not assert that every item is required for every filing. Use the current prescribed form and filing instructions.

- [ ] Obtain and complete the current BRELA patent application/request form.
- [ ] Confirm the required applicant, inventor, and entitlement particulars.
- [ ] Confirm whether assignments, declarations, or other ownership documents are required.
- [ ] Complete an agent/representative appointment if applicable.
- [ ] Include priority documents and translations only if a priority claim is made and required.
- [ ] Confirm language, signatures, copies/file formats, and drawing requirements.
- [ ] Confirm filing fees and payment method from the current BRELA schedule.
- [ ] Retain the filing receipt and record the application number and subsequent deadlines.

BRELA links to [patent forms](https://www.brela.go.tz/documents/patent) and a [patent-fees schedule](https://www.brela.go.tz/pages/patent-fees). Verify the current forms, amounts, and filing instructions before filing.

## Project files used for this structure

- [Current implementation-evidence application](./FedAfya_Patent_Application_Implementation_Evidence.docx)
- [Original application draft](./FedAfya_Patent_Application_With_Figures_For_Review.docx)
- [Invention disclosure draft](./FedAfya_Invention_Disclosure_For_Review.docx)
- [Implementation map figure](./FedAfya_Figure_Implementation_Map.png)
- [Evaluation snapshot figure](./FedAfya_Figure_Evaluation_Snapshot.png)
- [Preprocessing comparison code](../evaluation/preprocessing_comparison.py)
- [Preprocessing comparison report](../evaluation/preprocessing_comparison_report.csv)
- [Preprocessing comparison summary](../evaluation/preprocessing_comparison_summary.json)

## Drafting references

These PCT rules are international drafting references only; they do not establish Tanzanian national requirements:

- [WIPO PCT Rule 4 — Request](https://www.wipo.int/en/web/pct-system/texts/rules/r4)
- [WIPO PCT Rule 5 — Description](https://www.wipo.int/en/web/pct-system/texts/rules/r5)
- [WIPO PCT Rule 6 — Claims](https://www.wipo.int/en/web/pct-system/texts/rules/r6)
- [WIPO PCT Rule 7 — Drawings](https://www.wipo.int/en/web/pct-system/texts/rules/r7)
- [WIPO PCT Rule 8 — Abstract](https://www.wipo.int/en/web/pct-system/texts/rules/r8)
- [WIPO PCT Rule 11 — Physical requirements](https://www.wipo.int/en/web/pct-system/texts/rules/r11)
