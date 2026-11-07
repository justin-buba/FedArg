# Preprocessing comparison protocol

**Status:** Draft; not approved for execution

**Scope:** Documentation only. No preprocessing method or dataset has been changed or run for this comparison.

## 1. Objective

Compare three preparation methods on the same three hospital datasets, using documented rules and reporting results by hospital:

| Method | Description | Role |
|---|---|---|
| B0 | Basic type cleaning and handling of missing entries | Reference baseline |
| B1 | Common-schema mapping and categorical normalisation | Established-method comparison |
| B2 | Existing, locally executed FedAfya preparation workflow | Proposed approach |

The comparison is descriptive and empirical. It must not assume that B2 is best on every measure.

## 2. Current project facts to verify at experiment start

- The repository README reports 15,208 raw records and 13,329 records after identifier-based deduplication (Hospital A: 10,569; B: 990; C: 1,770).
- Raw patient CSVs are excluded from Git by `.gitignore`; they are not present in the tracked repository. The README's figures therefore need to be verified against the exact local input files before reporting them as experiment results.
- The current code path is `client/data_utils.py` followed by `client/clean_app.py`. It maps source columns into a common representation, parses age, categorises sponsor, and deduplicates by identifier; the cleaning step imputes or replaces vital signs, creates age bins, and maps outcomes.
- The README describes 11 logical model features and notes that Hospital C does not have the five vital-sign source columns and has no source `Medications` column. A placeholder or default must be reported as engineered/structural, not as recovered clinical information.
- `client/clean_app.py` maps `Home`, `Referral`, and `Death` to numeric labels 0, 1, and 2, respectively. `ARCHITECTURE_REPORT.md` contains an older, conflicting outcome mapping and must not be used as the B2 rule source without reconciliation.
- Existing result summaries and manuscript notes are not substitutes for measurements produced by this comparison.

## 3. Decisions required before implementation or execution

Do not run the comparison until these decisions are recorded here:

1. **Input snapshot:** Identify the three exact source files, their versions, and safe checksums. Confirm that all methods receive the same input rows and record order.
2. **Deduplication boundary:** Decide whether deduplication is part of each method or a shared pre-comparison step. B2 currently deduplicates in its existing pipeline; report records removed separately. Do not silently compare methods on different row populations.
3. **Target schema:** Approve the required logical feature list and, for each hospital-feature pair, classify it as source-observed, structurally unavailable, derived, or placeholder. Record the source-column mapping.
4. **B0 rules:** Specify accepted type conversions and missing-value handling, including invalid values and features absent from a source schema.
5. **B1 rules:** Specify column aliases, categorical normalisation maps, unknown/unmapped handling, and missing-value rules.
6. **B2 snapshot:** Record the exact Git commit and any pre-existing local changes that affect the pipeline. Do not alter B2 to improve its comparison result.
7. **Value-retention policy:** Define which transformations count as preserving an observed meaningful value versus changing, replacing, or discarding it. Record the policy per feature where necessary.
8. **Timing and repeats:** Specify the number of repeated runs, timing boundaries, and whether timings include file I/O. Use the same environment and timing method for all methods.

## 4. Metrics and reporting definitions

Report counts and denominators alongside percentages, separately for each hospital and method. Freeze exact denominators after the input snapshot and target schema are approved.

| Measure | Planned definition and interpretation |
|---|---|
| Source feature availability | Required features represented by an original source field divided by all required logical features. Report derived and placeholder features separately; do not count them as source availability. |
| Output schema conformance | Whether the output has the approved required feature names and structure, plus the number and names of missing or unexpected features. This measures format, not clinical completeness. |
| Category consistency | For each mapped categorical feature, count distinct non-missing source forms that represent the same approved category before mapping and count unmapped/inconsistent values after mapping. Publish the mapping and denominator. |
| Valid-value rate | Valid observed values divided by values assessed under a documented validation rule. Keep missing, structurally unavailable, and invalid values distinct. |
| Information retention | Originally observed, meaningful values unchanged under the approved value-retention policy divided by originally observed meaningful values. Report transformed, imputed, defaulted, and discarded values separately; placeholders are never retained source information. |
| Missingness decomposition | Report ordinary missing cells for collected features separately from structurally unavailable features/columns. Also report invalid values that remain unresolved and values imputed or defaulted. |
| Preprocessing time | Elapsed time under the approved timing boundary, reported per method and hospital. |
| Reproducibility | Compare repeated outputs and feature mappings for deterministic equality; document any nondeterministic fields or differences. |

An invalid-value flag is evidence of detection only. Describe a value as corrected only if the resulting value was actually changed according to an explicit rule; otherwise report it as flagged, unchanged, or unresolved as applicable.

## 5. Execution and audit record

After the decisions above are approved, record each run before interpreting results:

| Run ID | Date/time | Git commit | Input snapshot/checksums | Method and rules version | Command/environment | Output checksums | Notes |
|---|---|---|---|---|---|---|---|
| Not run | — | — | — | — | — | — | Protocol draft only |

Store only approved aggregate results and documentation in Git. Do not add raw, patient-level, or otherwise sensitive healthcare data. Keep detailed local logs outside the repository if they could expose patient information.

## 6. Results table template

Populate only after protocol decisions are approved and all methods have been run on the same input snapshot.

| Hospital | Method | Source features available | Derived features | Placeholder features | Output schema conforms? | Category inconsistencies before → after | Valid-value rate | Observed values retained | Ordinary missingness | Structural unavailability | Unresolved invalid/missing values | Time | Repeat outputs identical? |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| A | B0/B1/B2 | — | — | — | — | — | — | — | — | — | — | — | — |
| B | B0/B1/B2 | — | — | — | — | — | — | — | — | — | — | — | — |
| C | B0/B1/B2 | — | — | — | — | — | — | — | — | — | — | — | — |

The report should interpret trade-offs, explain limitations, and avoid claiming that a common output schema recovers unavailable source information.

## 7. Change and rollback practice

- Make one reviewable change at a time; keep this protocol separate from pipeline implementation.
- Before a later implementation step, inspect `git status` and confirm only intended files are staged.
- Save approved milestones as small commits containing only the files for that milestone. Do not include unrelated pre-existing worktree changes.
- Record the commit hash in the run audit table. To undo a published milestone, use a new revert commit rather than rewriting shared history.
- At the start of this protocol draft, the worktree already contained unrelated modified and deleted files. They were not changed, staged, or committed as part of this step.
