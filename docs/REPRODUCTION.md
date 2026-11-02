# Notebook execution

These exports have outputs, execution counts, attachments, and cell metadata cleared. Source code is preserved. They have not been rerun locally or independently validated. Retain original Kaggle versions separately.

## Training

1. Import notebooks/training.ipynb into Kaggle and enable a compatible GPU.
2. Attach competition data after accepting its terms. Expected root: /kaggle/input/competitions/rsna-knee-abnormality-detection.
3. Attach yunusgmsoy/rsna-knee-llm-report-labels and the ResNet-18 resource at the paths specified in source. Adjust mounts if necessary.
4. Review exploratory cells: running them displays an MRI and individual record previews. Clear regenerated outputs before publishing an export.
5. Run sequentially: a later loader definition overrides an earlier one. The unused MAX_TRAIN_STEPS variable does not limit training.
6. Legacy encoder loading uses weights_only=False; use a trusted source. Saved v03 inference weights load with weights_only=True.
7. Use a saved background run and confirm output preservation. Full training saves kneesignal_resnet18_v03_fold0.pt and an optimizer recovery snapshot. Automatic or exact resume is not implemented.

## Inference

1. Import notebooks/inference.ipynb into a separate Kaggle notebook.
2. Attach competition inputs and the completed v03 training output. The loader searches for exactly one kneesignal_resnet18_v03_fold0.pt. An old error message mentions Version 4; the filename is the relevant requirement.
3. Run three cells in order. Check prediction shape, finite probabilities, and aggregate read failures.
4. Output is /kaggle/working/submission.csv. Do not use the training notebook's constant example CSV.
5. For competition execution, disable internet and follow the competition submission requirements.

Exact package versions and a frozen training image are not recorded here. Local dependency ranges are not a claim of exact Kaggle reproduction. Data and weights are obtained separately. Resolve licensing before public release.
