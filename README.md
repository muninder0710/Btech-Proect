# What drives arrhythmia classification on UCI Arrhythmia?

This repository holds the code, pre-registrations and results of a B.Tech research project on the [UCI Arrhythmia dataset](https://doi.org/10.24432/C5BS32) (452 patients, 279 ECG attributes).

The main study is a factorial comparison. It asks which of three design choices moves the test score the most:

- **Missing values:** median fill, or median fill plus "value was missing" indicator columns.
- **Feature selection:** none, three filters (ReliefF, mutual information, mRMR) or four wrappers (IBGWO4, and guided GWO-CPO, BPSO and GA).
- **Classifier:** 5-nearest neighbours, a class-balanced RBF SVM, random forest or XGBoost.

All 64 combinations are scored on the same 25 outer folds (5 × 5 stratified cross-validation), for the two-class task (normal against abnormal) and the twelve-class task. Every setting is chosen inside the training part of each fold.

**Main result.** The classifier explains about 40% of the variation in macro-F1, feature selection about 2 to 3.5%, and missing-value handling almost none. Feature selection mainly helps 5-NN. The best pipelines reach macro-F1 0.819 for two classes (mRMR with random forest) and 0.479 for twelve (SVM on all features). The numbers are in [results/what_matters/summary/](results/what_matters/summary/).

The repository also holds the project's earlier experiments: a new wrapper (GWO-CPO), measurements of testing shortcuts, and re-tests of two published results. They are listed in [Every experiment](#every-experiment) below.

## Quick start

Python 3.12 on macOS or Linux. Run every command from this folder.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python fetch_data.py                                     # once: checks the UCI files in data/, downloads any that are missing
python check_what_matters.py                             # must print "all checks passed"
python what_matters_demo.py --smoke                      # synthetic data, short run      -> results/what_matters_smoke/
python what_matters_demo.py                              # the real run, about 9 minutes  -> results/what_matters/
python analyze_what_matters.py                           # claims, tables, figures 1-10   -> results/what_matters/summary/
```

The smoke run uses synthetic data, so a bug can be found without looking at real results. If the real run stops, run the same command again: it resumes from the folds already finished.

The library versions are pinned in `requirements.txt`. A different scikit-learn or BLAS version can round a distance differently and change a few predictions, so report numbers from one machine.

On macOS, XGBoost needs an OpenMP library. If Homebrew's `libomp` is missing, the scripts that use XGBoost restart themselves once with the copy that scikit-learn ships. On Linux nothing changes.

### Run on Google Colab

```text
!git clone https://github.com/<your-account>/<this-repository>.git
%cd <this-repository>
!pip install -r requirements.txt
!python check_what_matters.py && python what_matters_demo.py && python analyze_what_matters.py
```

Colab's Python and library versions differ from the pinned ones, so its numbers can differ slightly in the third decimal.

## How the main study is protected against fooling itself

- **Pre-registration.** The five questions, their decision rules and the ten figures were written down before the real data was run. See [design/amendment-13-what-matters.md](design/amendment-13-what-matters.md).
- **Nested testing.** Preprocessing, indicator columns, filter sizes and wrapper searches are fitted on the training patients of each outer fold only. The test patients are scored once.
- **Checks before the run.** `check_what_matters.py` tests four things:
  1. the indicator columns are built from training patients only;
  2. each filter ranks the informative column of a toy problem first;
  3. mRMR skips a duplicated column;
  4. a fold gives identical rows when run twice.
- **Decision log.** [design/decisions.tsv](design/decisions.tsv) is an append-only log with one row per design decision, result, check or correction, each with a timestamp.

## Every experiment

Each experiment has a pre-registration in [design/](design/) and writes to its own folder in `results/`.

| Experiment | Design | Scripts | Results |
| --- | --- | --- | --- |
| **Main study:** missing values × selection × classifier | Amendment 13 | `what_matters_demo.py`, `check_what_matters.py`, `analyze_what_matters.py` | `what_matters/` |
| First study: five optimisers on six UCI datasets ([guide](docs/first-study.md)) | `grounding.md`, `rubric.md` | `run_experiments.py`, `analyze.py`, `check_optimizers.py`, `check_protocol.py` | `full/` |
| Leakage before cross-validation | Amendments 3–4 | `leakage_demo.py` | `leakage/` |
| Choosing the subset on the reported split | Amendments 3–4 | `test_reuse_demo.py` | `test_reuse/` |
| How much one 42-patient test split can show | Amendments 3–4 | `small_split_demo.py` | `small_split/` |
| GWO-CPO under the base paper's own protocol | Amendment 5 | `base_protocol_demo.py` | `base_protocol/` |
| Mutual-information (guided) start | Amendment 6 | `informed_init_demo.py` | `informed_init/` |
| Tuning on the test patients (Dhanka & Maini 2025) | Amendment 7 | `tuning_on_test_demo.py` | `tuning_on_test/` |
| Proposed selection against the base filter, same classifier | Amendment 8 | `same_classifier_demo.py` | `same_classifier/` |
| Seventeen optimisers from two starts | Amendments 9–10 | `extra_optimizers.py`, `check_extra_optimizers.py`, `more_optimizers_demo.py`, `plot_optimizers.py` | `more_optimizers/` |
| Benchmark functions, as published and shifted (side check) | Amendment 11 | `benchmark_functions_demo.py`, `check_benchmark_functions.py`, `plot_benchmark_functions.py` | `benchmark_functions/` |
| The base paper's nine gene-expression datasets | Amendment 12 | `fetch_gems.py`, `base_datasets_demo.py`, `check_base_datasets.py`, `plot_base_datasets.py` | `base_datasets/` |

Run each script from this folder, for example `python leakage_demo.py`. Each script's first lines say what it does and how to run it. Scripts named `check_*.py` stop with an error at the first failed check.

`deck/build_review_pptx.py` builds the review slides. It reads a slide template that is not part of this repository, so it runs only on the project machine.

## Code layout

- `arrfs/`: the shared package.
  - `data.py` loads the tasks and holds the per-fold preprocessing, fitted on training rows only.
  - `fitness.py` holds the wrapper fitness (5-NN over 3 inner folds) and `class_scores`, the one metric function.
  - `optimizers.py` holds BPSO, BGWO, BCPO, IBGWO4 and GWO-CPO behind `search(...)`.
  - `experiment.py` runs one outer fold of the first study.
- `extra_optimizers.py`: the twelve further optimisers of Amendments 9–10, kept outside `arrfs/` so that `results/full/` stays valid.
- `design/`: pre-registrations, the decision log, and checks of published papers (`design/checks/`). `candidate-1/` and `candidate-2/` are early design drafts, kept for the record.
- `docs/first-study.md`: the detailed guide to the first study.

## Data and licences

- **UCI datasets in `data/`.** The six datasets (Arrhythmia, dermatology, lymphography, SPECT, Statlog heart, WDBC) are licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). [data/SOURCES.txt](data/SOURCES.txt) gives each file's URL, DOI and checksum. Cite each one by its DOI.
- **GEMS gene-expression datasets.** These are the nine datasets used by Khaseeb et al. 2025 (Statnikov et al. 2005, Bioinformatics 21:631). They are **not included**, because their source states no licence. `python -I fetch_gems.py` downloads them into `data/gems/` and checks their checksums.

## Notes

- **Expected warnings.** The twelve-class task has a class of 4 patients, so scikit-learn warns that the least populated class has fewer members than there are folds. This is expected. See [docs/first-study.md](docs/first-study.md#warnings-you-will-see).
- **Reading p-values.** The 25 folds of repeated cross-validation share training patients, so p-values are a guide, not exact. Read the mean difference, its bootstrap interval and the win/tie/loss counts first.

## Authors

<!-- Add the authors, the supervisor and the institution here. -->

## Licence

<!-- Choose a licence for the code (for example MIT) and add a LICENSE file. The data keeps its own licence (see above). -->
