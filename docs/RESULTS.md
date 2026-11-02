# Recorded results

Results were recorded in Kaggle logs and submission screenshots; notebook outputs are cleared.

- Baseline V2 public macro AUROC: 0.594; local validation: 0.6186.
- Revised V3 public macro AUROC: 0.659; local validation: 0.5052.
- Final training: 4,395 studies, 8:02:24, weighted loss 0.2619.
- Validation: 12 expert-labelled studies, 10 evaluable targets. Lateral OA and Fracture lacked both classes.
- Checkpoint: kneesignal_resnet18_v03_fold0.pt, approximately 45.1 MB.
- Visible inference check: three studies, 12 predictions each; zero sampled read failures. This does not establish hidden-test data quality.
- Short loading benchmark: 20 studies in 3:47 with zero workers versus 1:47 with two workers. Caching and other conditions were not controlled.

The public +0.065 gain cannot be attributed to loss weighting alone. Old and revised losses have different normalization and are not directly comparable. Final private results remain unverified here.

Future priorities: grouped/multilabel validation where identifiers permit, pseudo-label auditing, controlled experiments, tested resume support, and external evaluation.
