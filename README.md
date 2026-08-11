# AdaRCG

This repository provides the code needed to run the AdaRCG experiments on four EEG emotion-recognition datasets:

- SEED
- MPED
- SEED-IV
- SEED-V

The implementation follows the accompanying paper. This README focuses on installation, data preparation, execution and output files; the motivation and interpretation of the model components are described in the manuscript.

## 1. Installation

Python 3.10 or later is recommended.

```bash
git clone https://github.com/QilinLi147/AdaRCG.git
cd AdaRCG
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

For a CUDA run, install the PyTorch build appropriate for the local CUDA driver before running `pip install -e .`.

## 2. Obtain the datasets

The datasets are governed by their respective licences and are not redistributed here. Download the released feature files from the official dataset providers, then keep each dataset in a separate directory.

The preparation command expects the following released layouts:

| Dataset | Expected source layout |
|---|---|
| SEED | `DE/raw/DE_1.mat` to `DE/raw/DE_45.mat` below the source directory |
| MPED | `DE_1.mat` to `DE_23.mat`, `mped_label.mat` and `mped_label_3type.mat` |
| SEED-IV | session directories `1/`, `2/` and `3/`, each containing the released subject MAT files |
| SEED-V | `1_123.npz` to `16_123.npz` |

## 3. Prepare the four caches

Run one command per dataset. The output is a split-free ten-frame, five-band cache used by the training entry point.

```bash
python -m adarcg.prepare \
  --dataset seed \
  --source /path/to/SEED \
  --output data/seed.npz

python -m adarcg.prepare \
  --dataset mped \
  --source /path/to/MPED \
  --output data/mped.npz

python -m adarcg.prepare \
  --dataset seediv \
  --source /path/to/SEED_IV/eeg_feature_smooth \
  --output data/seediv.npz

python -m adarcg.prepare \
  --dataset seedv \
  --source /path/to/SEED_V/EEG_DE_features \
  --output data/seedv.npz
```

Each command also writes a neighbouring `*.manifest.json` containing the cache fingerprint, shape and data contract. No normalisation is applied during cache construction; the training command fits normalisation using the permitted training partition of each fold.

## 4. Run AdaRCG

### Quick single-subject run

Subject identifiers in the command line are one based.

```bash
python -m adarcg.run \
  --dataset seed \
  --cache data/seed.npz \
  --subject 1 \
  --output runs/seed \
  --device cuda:0
```

For MPED, a single trial-disjoint fold may also be selected:

```bash
python -m adarcg.run \
  --dataset mped \
  --cache data/mped.npz \
  --subject 1 \
  --fold 1 \
  --output runs/mped \
  --device cuda:0
```

### Complete dataset run

Omit `--subject` and `--fold` to execute every formal fold for a dataset:

```bash
python -m adarcg.run \
  --dataset seediv \
  --cache data/seediv.npz \
  --output runs/seediv \
  --device cuda:0
```

Use the same command with `seed`, `mped`, `seediv` or `seedv`. `--device auto` selects the first CUDA device when available and otherwise uses the CPU. Completed folds are detected from their `metrics.json` file and skipped, so the same command can resume an interrupted dataset run.

All runs load the fixed dataset configuration used for the paper. The command line intentionally exposes only dataset, cache, subject/fold, output and device controls.

## 5. Replay a checkpoint

```bash
python -m adarcg.evaluate \
  --checkpoint runs/seed/subject_01/fold_01/checkpoint.pt \
  --cache data/seed.npz \
  --device cuda:0 \
  --output replay/seed_subject01_predictions.npz
```

The evaluator verifies the cache fingerprint stored in the checkpoint before producing predictions.

## 6. Outputs

Each completed fold contains:

| File | Content |
|---|---|
| `checkpoint.pt` | Model state, train-fitted normalisation and fold record |
| `metrics.json` | Accuracy, balanced accuracy and macro-F1 |
| `predictions.npz` | Test indices, labels, predictions and class probabilities |
| `history.csv` | Development selection and fresh refit trace |
| `split.json` | Exact sample indices used by the fixed protocol |

`aggregate.json` is written at the dataset output root. MPED folds are averaged within each subject before the across-subject mean and standard deviation are calculated.

## 7. Cache contract

The public runner accepts caches produced by `adarcg.prepare`. The principal arrays are:

- `x`: `[sample, 62, 50]`, ordered as ten frames by five frequency bands within each channel;
- `y`: zero-based class IDs;
- `subject`, `session`, `trial`: zero-based sample identifiers;
- `emotion`: original seven-emotion trial ID for MPED only;
- `metadata`: JSON data contract stored as a scalar string.

The five-band order is delta, theta, alpha, beta and gamma. Altering channel order, feature order or fold metadata changes the input contract and is not supported by the released checkpoints.

## 8. Minimal code check

```bash
python -m adarcg.prepare --help
python -m adarcg.run --help
python -m adarcg.evaluate --help
```

## Citation

Please cite the accompanying AdaRCG paper when using this code. Publication metadata can be added here after the article receives its final bibliographic record.

## Licence

This project is released under the [MIT License](LICENSE).
