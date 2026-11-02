# KneeSignal — Knee MRI Abnormality Detection

A research pipeline predicting 12 abnormalities from multi-series knee MRI examinations, developed with AI-assisted coding for the RSNA competition.

**Recorded public macro AUROC: 0.659**, compared with baseline 0.594 (+0.065 absolute). Final private results and rank have not been verified. AUROC is not diagnostic accuracy.

## Structure

- `notebooks/training.ipynb`: cleaned export of the v03 Kaggle training notebook.
- `notebooks/inference.ipynb`: cleaned export of the successful inference notebook.
- `docs/REPRODUCTION.md`: resources and execution instructions.
- `docs/RESULTS.md`: recorded results and limitations.
- `src/meniscinet/`, `tests/`, `configs/`: earlier local utilities, tests, and scaffolding.
- `rsna_knee_baseline.py`: experimental local baseline, different from the submitted model.
- Other documents in `docs/`: historical proposals and revision notes; planned features are not necessarily implemented.

The notebooks record the submitted workflow. The local script uses neighboring-slice channels and regex pseudo-labels and cannot reproduce the submitted checkpoint.

## Approach

Available data comprised 4,407 studies and 24,371 series, with 58 expert-labelled studies. Public report-derived pseudo-labels supplied supervision for the remaining 4,349. Fold 0 reserved 12 expert-labelled studies and trained on 4,395.

Up to five series and 24 slices per series are resized to 224 × 224. Grayscale slices are repeated across three channels for a pretrained ResNet18 encoder. Mean slice features are combined with plane, fluid sensitivity, and fat suppression metadata; attention pools series into 12 predictions.

Training used one epoch, batch size 1, AdamW (learning rate 0.0002, weight decay 0.0001), two loader workers, and weighted binary cross-entropy (expert weight 1.0, pseudo-label weight 0.55). Reports contribute indirectly through external labels, rather than being model inputs.

## Running

Follow [notebook execution instructions](docs/REPRODUCTION.md). Training and inference run separately on Kaggle with authorized competition inputs and externally obtained weights. Recorded training took approximately eight hours.

For the earlier local utilities:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest
```

Dependency ranges in pyproject.toml describe local installation, not an exact frozen Kaggle environment. Bit-for-bit reproducibility has not been established.

## Results and limitations

See [recorded results](docs/RESULTS.md). Local validation contained only 12 studies and supported AUROC for 10 targets. It was too small for dependable model selection. Pseudo-label provenance and quality require further auditing. Public score improvement cannot be attributed to one code change in isolation.

The exported training notebook retains exploratory cells and duplicate loader definitions; execute sequentially. Failed slices may be repeated or replaced by zeros, and entirely unreadable series are not explicitly masked out. Checkpoints are saved, but automatic resume is not implemented.

This is a research prototype, not a clinically validated diagnostic tool. Calibration, external validation, and subgroup performance have not been established.

## Credits and distribution

- [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection): competition images, labels, metadata, and evaluation.
- [RSNA Knee LLM Report Labels — yunusgmsoy](https://www.kaggle.com/datasets/yunusgmsoy/rsna-knee-llm-report-labels): external pseudo-label table.
- Pretrained encoder resource used: `/kaggle/input/datasets/pytorch/resnet18/resnet18.pth`.
- PyTorch, torchvision, pydicom, NumPy, pandas, scikit-learn, and tqdm.

Data, reports, weights, prediction CSVs, and notebook outputs are excluded. Obtain resources from their authorized sources and follow their terms. No repository-wide license has been selected: competition code-sharing requirements and third-party licenses need resolution before public release. This repository does not grant rights to external data or weights.
