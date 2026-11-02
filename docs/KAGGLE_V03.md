# KneeSignal training revision after public baseline 0.594 (V2)

Preserve the successful baseline and inference versions. Copy the training
notebook to `KneeSignal — Training v03`. The local baseline script has different
module names and three-channel neighboring slices; it is NOT a replacement for
the Kaggle model definition or compatible with its checkpoint.

## First edit: loss and checkpoints (Cell 15)

Replace the weighted-loss expression with:

```python
loss = (loss_per_target * weights).mean()
```

Dividing by the weight sum previously cancelled uniform weights at batch size
one. This fixes loss/gradient scaling; Adam's adaptive updates mean this alone
does not guarantee proportional parameter updates or higher AUC. Consider mixed
gold/pseudo gradient accumulation in a later measured experiment.

Before the epoch loop add:

```python
def save_training_checkpoint(epoch, step):
    destination = Path('/kaggle/working/kneesignal_latest.pt')
    temporary = destination.with_suffix('.tmp')
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'step': step,
        'targets': TARGETS,
    }, temporary)
    temporary.replace(destination)
```

Change the training loop header to:

```python
for step, batch in enumerate(
    tqdm(train_loader, desc=f'Training epoch {epoch + 1}'), start=1
):
```

Immediately after `optimizer.step()`, inside that loop, add:

```python
if step % 200 == 0:
    save_training_checkpoint(epoch, step)
```

After the training loop and before validation, add:

```python
save_training_checkpoint(epoch, len(train_loader))
```

These are recovery snapshots, not exact deterministic resume support. Files in
an interactive session still need to be downloaded or persisted; merely writing
a checkpoint does not guarantee it survives session termination. Continue using
saved batch runs for durable successful outputs.

## Required checks before the next long run

- Remove the old `load_mri_series` definition from Cell 6. Move the corrected
  Cell 11 loader above its first use. Keep exactly one definition so Run All
  cannot change behavior through notebook execution order.
- Keep identical loading and normalization in training and inference; do not
  change normalization only when applying existing trained weights.
- Remove the Cell 14 smoke-training step from production training. Initialize
  a fresh model and optimizer after setting the random seed.
- Validate pseudo-label range and UID join, and audit teacher performance on
  gold labels before any overrides. Check provenance for gold-label copying;
  agreement with copied gold labels is not independent teacher validation.
- Run a short timed test before full training. Report loading failures as
  aggregate counts rather than silently treating all failures as valid images.
- Save a fixed fold assignment. Produce held-out predictions for each of the
  five folds, combine them, then compute each target's AUC on all gold studies.
  Do not present an average over only evaluable targets as the competition's
  full 12-target metric. With 58 studies, repeated tuning remains very noisy.

## Local script improvements

The local script now supports `--fold 0` through `--fold 4`, writes per-fold
OOF predictions, best model weights, and optimizer snapshots every 200 batches.
It uses one shared preprocessing function for training and inference. Adjacent
slice channels repeat the boundary slice instead of wrapping to the opposite
end of the scan. It still uses a primitive regex teacher and is experimental;
do not replace the public-label Kaggle workflow with it for score comparisons.
