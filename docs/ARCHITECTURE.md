# Architecture

`data` owns DICOM discovery, pixel normalization, and study/series manifests.
`labels` owns gold-label masking and report-derived soft labels. `models` owns
the study-level network. `training` owns folds, loss weighting, validation, and
out-of-fold artifacts. `inference` owns model loading, test prediction, blend
weights, and strict submission validation.

The boundary is intentional: only `labels` may access reports, and no inference
module may accept a report field. This keeps hidden-test behavior honest.

```text
train.csv reports ──> labels / pseudo-label teacher ──┐
train_series DICOM ─> data / selected study tensors ──┼─> training ─> weights + OOF
test_series DICOM ──> data / selected study tensors ──┴─> inference ─> submission.csv
```
