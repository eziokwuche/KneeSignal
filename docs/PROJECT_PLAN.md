# MenisciNet project charter

## Mission

Build a reproducible, MRI-only ensemble that predicts the 12 RSNA Knee
Abnormality Detection targets from each study's DICOM series and accompanying
series metadata. The delivered result is a Kaggle notebook that completes in
under nine GPU hours with Internet disabled and writes `submission.csv`.

## Success criteria

- A valid notebook submission with the exact required 12 probability columns.
- Reproducible five-fold, study-level validation on the 58 expert-labelled
  studies; macro ROC AUC is the decision metric.
- A documented pseudo-label source and a measurable comparison against gold
  labels before pseudo-labels are used for image training.
- An inference run comfortably below the nine-hour notebook limit, including
  DICOM decoding and submission creation.

## In scope

- Data integrity checks, DICOM-series selection, and robust image loading.
- Conservative report-to-label teacher development for the unlabeled training
  studies. Reports are a training signal only.
- 2.5D and compact 3D visual models, cross-validation, calibration, and
  probability ensembling.
- Kaggle notebook packaging, offline weight attachment, runtime profiling, and
  final reproducibility materials.

## Explicitly out of scope

- Clinical deployment, diagnostic claims, patient-facing use, or prospective
  validation.
- Any use of test reports or network access during notebook execution.
- Optimizing for a single split, using leaked identifiers, or treating pseudo
  labels as expert ground truth.

## Delivery milestones

### September 4–7 — Foundation

Confirm the Kaggle data layout, count studies/series/slices, identify corrupt
DICOMs, and establish the fixed gold-only fold assignment. Exit criterion: a
deterministic audit report and a valid constant-prediction submission.

### September 8–14 — Label teacher

Benchmark conservative regex labels and a vetted public/offline report teacher
on the 58 gold studies. Preserve uncertainty as soft targets instead of forcing
unmentioned findings to negative. Exit criterion: selected pseudo-label source
with per-target AUC and error review.

### September 15–28 — Visual baseline

Train the 2.5D baseline over all pseudo-labelled studies while keeping gold
labels at full weight. Compare series selection policies and image resolutions.
Exit criterion: reproducible OOF predictions, learning curves, and a first
public-LB submission.

### September 29–October 5 — Model improvement

Evaluate a stronger offline-pretrained encoder, compact 3D alternative, and
per-target calibration. Retain only changes that improve gold-only OOF AUC and
fit the runtime budget. Exit criterion: two complementary validated models.

### October 6–12 — Ensemble and efficiency

Train final folds, blend model families from out-of-fold predictions, profile
DICOM decoding and inference, and produce a fast track candidate. Exit
criterion: versioned OOF scores, model weights, and timing evidence.

### October 13–15 — Entry lock

Accept competition rules, ensure team membership is final, and freeze a fully
reproducible submission notebook. Exit criterion: a committed dry run that
creates `submission.csv` without Internet.

### October 16–21 — Finalization buffer

Rerun the locked notebooks, validate schema and probability ranges, submit the
best candidate, and prepare solution documentation. Use this buffer only for
evidence-backed fixes—not new unvalidated architecture ideas.

### October 22 — Final submission

Submit the validated main and optional efficiency-track notebooks before the
11:59 PM UTC deadline. Preserve code, configs, OOF files, and model provenance
for the November 5 winners' obligations.

## Risks and controls

- **Only 58 gold studies:** hold out gold labels exclusively for model choice;
  report uncertainty and per-target support alongside macro AUC.
- **Pseudo-label noise:** gate each teacher on gold-label performance, lower its
  loss weight, and audit its most confident disagreements.
- **DICOM/runtime failures:** cache decoded series features when permitted,
  handle invalid files, and profile before final training.
- **Hidden domain shift:** favor three-plane, sequence-aware sampling and simple
  ensembles over overfitting a small validation fold.
