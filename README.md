# Measured Robustness of ML-NIDS Depends on Evaluation

Code for the paper *Measured Robustness of ML-NIDS Depends on Evaluation:
Perturbation Budget, Adversary Capability, and Attack Configuration*
(R. M. Gichila and P. S. Wang, National Taiwan University of Science and
Technology).

One program produces every model-dependent number in the paper: data
preparation (with deduplication), training of all models, every attack, and
every analysis stage. This repository contains code only: results, trained
models and data are not included (see `.gitignore`); running the program
downloads the data and creates them.

## Repository layout

| Path | What it is |
|---|---|
| `mlnids_robustness_evaluation.py` | The complete evaluation program (9,128 lines). MD5 `531032fd1a70cd592c49248c32718463`. |
| `run_all.py` | Runs the whole evaluation in one notebook cell or from the command line: data step, all 60 parts, the tables, and zips of the results and of the trained models. |
| `run_stage.py` | Runs one stage on the saved models of a finished run (set `STAGE`, or pass its name as an argument). |
| `strong_lattice_cic.py` | Adds the C2 lattice of CSE-CIC-IDS2018 to the stronger-attack set, flow by flow, from saved per-flow results (no attack, no training). |
| `make_paper_figures.py` | Draws Figures 1-3 of the paper as separate PDF files (dpi 1600) from the reported ten-seed values, which it contains; it reads no result file. |
| `requirements.txt` | Library versions of the reported run. |

## Data

CSE-CIC-IDS2018 (corrected release) and UNSW-NB15 are downloaded through
`kagglehub` from the Kaggle datasets `dhoogla/distrinetcsecicids2018` and
`dhoogla/unswnb15`. To use a local copy, set `NIDS_DATA_DIR` to a folder whose
path contains the dataset slug. The data step removes exact duplicate flows and
feature vectors that occur with both labels before the 80/20 split, and freezes
the preprocessed data in `data_final/`; every later step loads that file.

## Running

From the repository root, in a folder with no earlier results:

```
pip install -r requirements.txt
python run_all.py
```

or, equivalently, with the program alone:

```
RERUN_PART=data    python mlnids_robustness_evaluation.py       # data step
python mlnids_robustness_evaluation.py                          # all 60 parts, in parallel
RERUN_PART=summary python mlnids_robustness_evaluation.py > rerun_summary.txt   # the tables
```

A part is one (dataset, activation, seed): 2 datasets x 3 activations x
10 seeds (1-9 and 42) = 60 parts. Each part trains and saves its models in
`models_final/`, runs the base attacks and then every stage; per-part rows go to
`results_final/`, and the summary step combines them into
`results_final/<table>_final.csv`, prints the tables to `rerun_summary.txt` and
draws draft figures into `figures_final/`. Then:

```
python strong_lattice_cic.py                     # lattice in the stronger-attack set (CSE-CIC-IDS2018)
python make_paper_figures.py                     # Figures 1-3 of the paper, in figures_paper/
```

Every model and every attack is seeded from its own keys, so running parts in
parallel changes no number. The run is resumable: start the same command again
and finished work is skipped. To run a single stage on saved models:

```
python run_stage.py cxenum
```

Environment variables: `RERUN_PART` (`data`, `summary`, a part such as
`unsw-relu-1`, or empty for all parts), `RERUN_STAGES` (comma-separated stage
names; empty runs all), `RERUN_PARALLEL` (parts at a time; 0 = number of CPU
cores), `HSJ_THREADS` (threads per process, default 1), `NIDS_DATA_DIR`.

## Stages

After the base attacks (PGD, BIM, SPSA, transfer and HopSkipJump at capability
level C0 on every model; Model 4 also at C1 and C2), each part runs:

| Stage | Content |
|---|---|
| `wbbb` | White-box against black-box attacks at matched query budget; HopSkipJump under two search norms |
| `wbunbm` | PGD at unbounded budgets with cross-entropy or margin loss, last or best iterate |
| `restarts` | PGD and APGD-CE with 1-100 random starts, feature mask inside the search |
| `capa` | Source-side (C2) adversary: whole units, lattice of the UNSW-NB15 box (9^4 points), large budgets |
| `capb` | Arithmetic on the base results for the capability section (no attack) |
| `cnn` | The convolutional detector |
| `advtrain` | PGD adversarial training (PGD-AT) and its evaluation |
| `xai` | Feature attribution and the first-order margin-to-gradient diagnostic |
| `caat` | Source-side adversarial training (SS-AT) |
| `capu` | C2 with integer fields moved by whole units and TCP-only fields fixed on non-TCP flows |
| `fgsm` | FGSM on every standard model and budget |
| `capd` | C2 perturbation confined to one direction |
| `rvstd1` | Standard Model 4 retrained with one optimizer state for the whole run |
| `rvspsam` | SPSA with the logit-margin loss (SPSA-M) |
| `rvthr` | Attacks at a decision threshold re-tuned to a common false-positive rate |
| `rvstrong` | Stronger worst case: 100-start PGD, SPSA-M, HopSkipJump at C0, lattice at C2 |
| `rvhsjc2` | HopSkipJump with the C2 mask inside its search (standard, PGD-AT, SS-AT) |
| `cxcomb` | C2 restrictions combined (direction, TCP-only fields, whole units, recomputed flag totals); which features the successful perturbations move |
| `fam` | Attack-family and protocol composition of the pools; worst case by family at C0 and C2 |
| `cxenum` | Lattice of the CSE-CIC-IDS2018 C2 box (5^6 points) beside the attacks |
| `atfgsm` | FGSM on the saved PGD-AT networks |

Each stage has its own settings block (marked `stage <name>: settings`) and
code block (`stage <name>: code`). The results were produced by this program
before its comments were condensed for publication. Only comments,
docstrings and the wording of some headings printed in `rerun_summary.txt`
were changed; the code is identical (checked with Python's `ast`), so every
number it computes is the same.

## Environment

The reported results were computed on CPU (the program sets
`CUDA_VISIBLE_DEVICES=""`), one thread per process, with PyTorch's
deterministic algorithms in warn-only mode, under Python 3.12.6, PyTorch 2.14.0
and ART 1.20.1 (all versions are printed at the top of `rerun_summary.txt`).
Repeatability was checked on one machine and configuration; identical numbers
on other machines or library versions were not tested.
