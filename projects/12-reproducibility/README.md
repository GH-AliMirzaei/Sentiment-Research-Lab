# Project 12 ? Reproducibility and CNN Ablation Study

## Objective

Evaluate random-seed sensitivity and the contribution of convolutional branches in the frozen-BERT + CNN sentiment classifier.

## Experimental setup

- Dataset: IMDb.
- Frozen BERT backbone with a trainable CNN classification head.
- CNN kernel sizes: 2, 3 and 4; 96 filters per branch.
- Training samples: 512.
- Epoch-selection samples: 200.
- Evaluation samples: 1,000.
- Random seeds: 11, 22, 33.
- Training epochs: 2.
- Epoch selection: highest selection macro-F1.
- Evaluation and selection subsets have zero observed overlap.
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU.

## 1. Random-seed sensitivity

| Seed | Best epoch | CNN accuracy | BERT accuracy | McNemar p |
|---|---:|---:|---:|---:|
| 11 | 1 | 90.80% | 90.20% | 0.361595 |
| 22 | 2 | 91.10% | 90.20% | 0.078354 |
| 33 | 1 | 90.90% | 90.20% | 0.229523 |

- Mean CNN accuracy: 90.93%.
- Sample standard deviation: 0.153 percentage points.
- Baseline BERT accuracy: 90.20%.
- None of the three per-seed paired McNemar tests reached p < 0.05.

## 2. CNN branch ablation

Each variant was retrained with one convolutional branch functionally disabled by zeroing its output using a forward hook.
The classifier shape remained unchanged.

| Architecture | Mean accuracy | Delta vs full (pp) |
|---|---:|---:|
| Full CNN (2, 3, 4) | 90.93% | 0.000 |
| without_kernel_2 | 90.87% | -0.067 |
| without_kernel_3 | 90.77% | -0.167 |
| without_kernel_4 | 90.73% | -0.200 |

## 3. Interpretation

- The full CNN achieved the highest mean accuracy among the four evaluated configurations.
- Removing kernel 4 produced the largest average reduction, but the difference was small.
- Some individual seeds improved after branch removal.
- These descriptive results do not establish a statistically significant contribution for any single kernel.

## 4. Limitations

- Only three seeds were evaluated.
- The training set contains only 512 selected examples.
- Evaluation uses IMDb validation examples previously used in the broader research workflow; these are not a new independent final test set.
- Paired per-example predictions were not retained for full-versus-ablated comparisons, so no paired significance test is reported for the ablation.
- GPU deterministic-algorithm enforcement was not enabled; exact bitwise reproducibility was not established.
- Ablation comparisons are descriptive rather than causal proof of individual kernel importance.

## 5. Reproduction

Run from the repository root with the local Python environment:

```powershell
.\.venv\Scripts\python.exe .\projects\12-reproducibility\cnn_seed_study.py
.\.venv\Scripts\python.exe .\projects\12-reproducibility\cnn_ablation.py
```

Existing completed JSON results are reused or protected by the scripts.

## 6. Result files

- `results/project12_imdb/cnn_seed_summary.json`
- `results/project12_imdb/cnn_ablation_summary.json`
- Individual per-seed and per-ablation JSON reports.
