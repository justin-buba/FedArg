# Word Manuscript Update Checklist - URGENT FOR SUBMISSION DAY

## Repository Status Updates (COMPLETED)
The following issues in the repository have been FIXED and pushed to GitHub:

### ✅ FIXED: Static Hashing Salts (Security)
- **What was fixed**: Hardcoded salts removed from source code
- **New behavior**: Salts loaded from environment variables (MWAKATOBE_SALT_*)
- **User action**: Set environment variables before running in production
- **Related code**: client/data_utils.py

### ✅ FIXED: SMOTE Methodology (Evaluation Bias)
- **What was fixed**: SMOTE applied after train/test split (now only to training)
- **Impact**: Test set is now unbiased; synthetic samples don't leak into evaluation
- **Related code**: client/client_app.py::train_local_model()

### ✅ FIXED: Implausible Vital Signs (Data Quality)
- **What was fixed**: Values outside clinical ranges replaced with local median
- **Results**: 
  - Hospital A: 74 implausible values → 0 (7 Pulse, 28 Resp, 25 Temp, 14 Dia)
  - Hospital B: 6 implausible values → 0 (2 Pulse, 2 Resp, 2 Dia)
  - Hospital C: 0 implausible values
- **Ranges used**: Pulse 30-200, Resp 5-50, Temp 33-43, Sys 50-250, Dia 30-150
- **Related code**: client/clean_app.py

### ✅ UPDATED: README.md
- Corrected outcome mapping (Referral/Death, not Admitted/Referred)
- Updated outcomes table with verified distributions
- Documented range filtering for vital signs
- Fixed SMOTE methodology documentation
- Added environment variable setup instructions

---

## Manuscript Changes Required (URGENT)

### SECTION 1: EXECUTIVE SUMMARY / INTRODUCTION
**OLD TEXT** (example)
> The system contains 14,818 records after removing duplicates...

**NEW TEXT - UPDATE TO:**
> The system contains 13,329 records after removing duplicates from 15,208 raw records...

**MAPPING FOR NEW NUMBERS:**
- Total raw records: 15,208 (12,000 + 1,379 + 1,829)
- Duplicates removed: 1,879 (1,431 + 389 + 59)
- Final cleaned records: 13,329 (10,569 + 990 + 1,770)

---

### SECTION 2: OUTCOME DEFINITIONS
**OLD TEXT** (if present)
> Outcomes are mapped to Admitted, Referred, Discharged, and Unknown...

**NEW TEXT - UPDATE TO:**
> Outcomes are mapped to:
> - Home (class 0): Patient discharged home
> - Referral (class 1): Patient referred to another facility  
> - Death (class 2): Patient died during stay
> - Unknown (class 3): Missing or unrecorded outcome

**VERIFIED DISTRIBUTIONS (NEW):**
- Hospital A (10,569): Home=10,553 | Referral=8 | Death=8
- Hospital B (990): Home=978 | Referral=1 | Death=3 | Unknown=8
- Hospital C (1,770): Home=1,750 | Referral=2 | Death=6 | Unknown=12

---

### SECTION 3: DATA QUALITY / PREPROCESSING
**ADD NEW PARAGRAPH:**
> Implausible vital-sign values (outside clinically justified ranges) are replaced with the local median during cleaning. The clinical ranges used are:
> - Pulse: 30–200 beats per minute
> - Respiratory rate: 5–50 breaths per minute
> - Temperature: 33–43°C
> - Systolic blood pressure: 50–250 mmHg
> - Diastolic blood pressure: 30–150 mmHg
>
> This filtering removed 74 implausible values from Hospital A and 6 from Hospital B, resulting in zero invalid vital-sign values in all cleaned datasets.

**UPDATE TABLE** (if present):
| Hospital | Invalid values (before) | Invalid values (after) |
|----------|------------------------|----------------------|
| A        | 82                     | 0                    |
| B        | 10                     | 0                    |
| C        | 0                      | 0                    |

---

### SECTION 4: PRIVACY / SECURITY
**VERIFY / ADD:**
The following security and privacy measures are in place or planned:

**HASHING (Pseudonymisation - NOT Anonymisation):**
- ✅ Hashing salts are now loaded from environment variables (MWAKATOBE_SALT_*)
- ✅ Static salts have been removed from the public repository
- ⚠️  Hashing is pseudonymisation, not formal anonymisation

**DIFFERENTIAL PRIVACY:**
- ✅ Parameter clipping and noise functions exist (prototype)
- ❌ Formal DP accounting is NOT implemented
- ❌ Opacus PrivacyEngine is NOT currently used
- ❌ Do NOT claim formal (ε, δ) privacy guarantees

**SECURE AGGREGATION:**
- ✅ Additive masking functions exist (prototype)
- ❌ Cryptographic secure aggregation is NOT implemented
- ❌ Pairwise masks are NOT used
- ❌ Do NOT claim cryptographic protection

**COMPLIANCE:**
- ❌ GDPR compliance is NOT established
- ❌ HIPAA compliance is NOT established
- ❌ PDPA compliance is NOT established
- ✅ System is a research prototype with privacy-oriented controls

**ENSURE YOUR MANUSCRIPT:**
- Does NOT claim formal differential privacy
- Does NOT claim cryptographic secure aggregation
- Does NOT claim regulatory compliance (GDPR/HIPAA/PDPA)
- Uses terminology "prototype," "experimental," or "pseudonymisation"

---

### SECTION 5: METHODOLOGY - SMOTE
**IF PRESENT, UPDATE:**
> SMOTE is now applied ONLY to the training partition after the initial train/test split. This ensures the test set contains only original data for unbiased evaluation.

---

### SECTION 6: LIMITATIONS / FUTURE WORK
**ADD IF NOT PRESENT:**
> Formal security properties remain as future work. Specifically:
> 1. Deterministic encoder vocabulary for reproducible feature dimensions
> 2. Formal differential privacy accounting with (ε, δ) reporting
> 3. Cryptographic secure aggregation with pairwise masks
> 4. Regulatory compliance assessment (GDPR, HIPAA, PDPA)

---

### SECTION 7: REPRODUCIBILITY
**UPDATE SETUP INSTRUCTIONS:**
Before running the system in production, set environment variables for hashing salts:
```bash
export MWAKATOBE_SALT_ID=<strong-random-uuid>
export MWAKATOBE_SALT_NAME=<strong-random-uuid>
export MWAKATOBE_SALT_REGION=<strong-random-uuid>
export MWAKATOBE_SALT_DISTRICT=<strong-random-uuid>
export MWAKATOBE_SALT_WARD=<strong-random-uuid>
export MWAKATOBE_SALT_DIAGNOSES=<strong-random-uuid>
```

---

## FIND & REPLACE SUMMARY
| Search Term | Replace With | Sections |
|---|---|---|
| 14,818 | 13,329 | Throughout |
| 15,208 | 15,208 (no change, verify present) | Totals |
| Admitted | Referral | Outcome definitions |
| Referred | Death | Outcome definitions |
| formal differential privacy | prototype differential privacy | Privacy section |
| secure aggregation | experimental masking | Privacy section |
| GDPR/HIPAA compliance | (REMOVE if claiming compliance) | Security section |

---

## CRITICAL VALIDATION BEFORE SUBMISSION
- [ ] All old record counts (14,818) replaced with 13,329
- [ ] All outcome mappings show Referral/Death (not Admitted/Referred)
- [ ] Privacy section uses "prototype" or "experimental" terminology
- [ ] No claims of formal DP or cryptographic security
- [ ] No claims of GDPR/HIPAA/PDPA compliance
- [ ] Vital signs range filtering documented
- [ ] SMOTE methodology clarified (split first)
- [ ] No duplicate conclusion sections
- [ ] No working placeholders remain
- [ ] Cross-referenced with updated README.md

---

## README.md CROSS-REFERENCE
Manuscript should align with current README.md sections:
- Section 4.2: Data quality and missingness → README section 4.2
- Section 4.3: Effects of preprocessing → README section 4.3
- Section 4.4: Outcome harmonisation → README section 4.4
- Section 7.3-7.6: Privacy methods → README sections 7.3-7.6

If manuscript references different section numbers or old content, this indicates sync issues.

---

## FILE LOCATION
**Where is the Word manuscript?**
Please provide the full path so it can be updated directly:
- [ ] c:\Users\Frank\Documents\...\manuscript.docx
- [ ] c:\Users\Frank\Documents\...\paper.docx
- [ ] Other location: _______________

Once you provide the path, specific line numbers and updates can be made.
