# Feature selection with GWO-CPO on UCI Arrhythmia and five medical datasets

> This is the guide to the project's first study (`run_experiments.py`, `results/full/`), written on 2026-11-20. The main study of the repository is described in the top-level [README](../README.md).

This folder runs the study that compares five binary feature-selection optimisers. The method base paper is Khaseeb, Keshk and Youssef 2025 (Applied Sciences 15:489). Its best method, IBGWO4, runs a grey wolf (GWO) step, then a particle swarm (PSO) step, then keeps the best of the three populations. The proposed GWO-CPO is the same framework with the PSO step replaced by one step of the Crested Porcupine Optimizer (CPO, Abdel-Basset, Mohamed and Abouhawwash 2024, Knowledge-Based Systems 284:111257). The proposed model is GWO-CPO with the imbalance-aware fitness.

UCI Arrhythmia is the main dataset. Five more medical UCI datasets show whether the result generalises. Every number on the slides comes from `results/<run>/summary/`, and the commands below produce every file there.

## Run the experiment

Run every command from this `code/` folder with the project's Python.

```sh
cd code
source .venv/bin/activate
python fetch_data.py                   # once: download the six datasets into data/ and check them
python check_optimizers.py             # 1 s: each optimiser on a toy problem with a known answer
python check_protocol.py               # 16 s: leakage, resume, results table, KNN and metrics vs scikit-learn
python run_experiments.py --smoke      # 1 repeat: 7 tasks x 5 folds          -> results/smoke/
python analyze.py results/smoke        # tables and figures                   -> results/smoke/summary/
python run_experiments.py              # 5 repeats: 7 tasks x 25 folds        -> results/full/
python analyze.py results/full         # the numbers for the review           -> results/full/summary/
```

The check scripts stop with an error at the first failed check and print one line per passed check.

Each time below is one run on the project Mac, which has 10 cores (4 performance and 6 efficiency) and had the editor and a browser open. The runner used 10 worker processes. The smoke run (35 folds) took 87 s and the full run (175 folds) took 6 min 21 s. `analyze.py` took 3 s on the smoke run and 4 s on the full run.

The searches use 96% of the full run's CPU time. Each fitness call runs a 5-NN over 3 inner folds and costs about 1 ms, and that cost grows with the number of training records more than with the number of attributes. So the ten searches of one fold take about 32 s on Arrhythmia, 28 s on WDBC, 17 s on dermatology, and 3 to 6 s on the other three datasets.

To use fewer cores, add `--workers 4`. To write somewhere else, add `--out PATH`.

### Resume after a crash

If the run crashes, or you press Ctrl-C, run the same command again. It skips every fold that already has a file in `results/<run>/folds/`.

If you edited a file in `arrfs/` or changed a setting since the folder was started, the runner renames the old folder to `results/<run>.stale-<date-time>` and starts again. One table never mixes two versions of an algorithm. Nothing is deleted.

### Warnings you will see

Some classes have fewer records than there are folds, so scikit-learn prints warnings like these:

```text
UserWarning: The least populated class in y has only 4 members, which is less than n_splits=5.
UserWarning: The least populated class in y has only 2 members, which is less than n_splits=5.
UserWarning: The least populated class in y has only 1 members, which is less than n_splits=3.
```

They are expected. The 12-class Arrhythmia task has a class of 4 records, and lymphography has a class of 2 ("normal find"). Some outer test folds therefore hold none of them, and the 3 inner folds of a training fold can hold only 1 or 2. The pooled scores below count those records in full.

The full run also prints `RuntimeWarning: divide by zero encountered in divide` once, from the ANOVA F-test of fold `lymphography_r4_f0`. In that fold's training records, the attribute `lym.nodes dimin` is 1 in every class except fibrosis, where it is always 3. Its spread inside each class is zero, so its F score is infinite and `ANOVA-k` ranks it first. This warning is expected too.

### Run on Kaggle

1. Upload this `code/` folder as a Kaggle dataset, with `data/`. Leave out `.venv/` and `results/`.
2. In a CPU notebook, copy the folder to the writable working directory and run the same commands:

```text
!cp -r /kaggle/input/<your-dataset>/code /kaggle/working/code
%cd /kaggle/working/code
!pip install -r requirements.txt
!python fetch_data.py && python check_optimizers.py && python check_protocol.py
!python run_experiments.py
!python analyze.py results/full
```

With `data/` uploaded, `fetch_data.py` downloads nothing and only checks the files. Without it, the notebook needs internet access.

The `pip install` line pins the library versions used on the project Mac. Without it the code still runs, but a different BLAS or scikit-learn version can round a KNN distance differently and change a few predictions. Report the review numbers from one machine.

## The datasets

`fetch_data.py` downloads each dataset once from the UCI Machine Learning Repository into `data/<dataset>/`. `data/SOURCES.txt` lists each file's URL, DOI and sha256. All six datasets are licensed CC BY 4.0, so cite each one by its DOI.

| Task | Records | Attributes | Records per class |
| --- | --- | --- | --- |
| `arrhythmia-binary` | 452 | 279 | Normal 245, Arrhythmia 207 |
| `arrhythmia-multiclass` | 452 | 279 | 245, 44, 15, 15, 13, 25, 5, 9, 50, 4, 5, 22 (12 classes; labels 7 and 8 merged) |
| `lymphography` | 148 | 18 | normal find 2, metastases 81, malign lymph 61, fibrosis 4 |
| `dermatology` | 366 | 34 | psoriasis 112, seborrheic dermatitis 61, lichen planus 72, pityriasis rosea 49, chronic dermatitis 52, pityriasis rubra pilaris 20 |
| `spect` | 267 | 22 | diagnosis 0: 55, diagnosis 1: 212 |
| `statlog-heart` | 270 | 13 | absence 150, presence 120 |
| `wdbc` | 569 | 30 | benign 357, malignant 212 |

- Categorical attributes keep their numeric codes, as the base papers do. For example, statlog-heart's chest pain type stays 1 to 4. Every attribute is min-max scaled on the training fold, so the KNN treats a code as a number.
- Arrhythmia has missing values in 5 attributes (J angle in 376 records), and dermatology in age (8 records). The preprocessing drops an attribute missing in more than half of the training fold and fills the rest with the training-fold median.
- SPECT joins the files `SPECT.train` and `SPECT.test` (80 and 187 records), and the study makes its own folds. `SPECT.names` calls the two classes normal and abnormal but does not say which code is which, so the tables show the codes.

## The optimisers

All five search over masks, which say for each attribute whether to keep it. In one fold, all five start from the same 20 masks, and each gets exactly 2,000 fitness calls. Schedules that a paper writes with iteration t over T use the share of the 2,000 calls spent instead.

Each optimiser proposes a new continuous position Y from the current mask X, and bit d flips with probability |tanh(Y_d - X_d)|. This is the V-shaped transfer function that Khaseeb et al. found best. One rule for all five means the comparison is between update rules, not between binarisations.

- `BPSO` is binary particle swarm (Kennedy and Eberhart 1997). For PSO, Y - X is the velocity.
- `BGWO` is binary grey wolf (Mirjalili et al. 2014). The three best different masks found so far lead the wolves.
- `BCPO` is binary CPO (Abdel-Basset et al. 2024) with its four defences and greedy acceptance. Its population shrinks from 20 porcupines to 13 and then returns to 20, twice over the 2,000 calls.
- `IBGWO4` is the base paper's best method. Each generation runs one GWO step on the population, then one PSO step on the GWO result, and keeps the best 20 of the three populations.
- `GWO-CPO` is IBGWO4 with the PSO step replaced by one CPO step. Nothing else differs, so "base paper against ours" differs in exactly one step.

The CPO settings come from Abdel-Basset et al. 2024, Table 2: alpha 0.1, Tf 0.5, 2 reduction cycles, and a smallest population of 2/3 of the full one. The docstring of `_cpo_move` in `arrfs/optimizers.py` gives each defence's formula with the paper's equation number.

## Read the results

`analyze.py` computes every metric from the saved predictions with `arrfs.fitness.class_scores`, so per-fold and pooled numbers share one definition.

- Use the tables in `summary/KNN/` as the headline tables. KNN (k = 5) is the classifier inside the search. The `LogReg/`, `SVM/` and `RF/` folders score the same subsets with other classifiers.
- Read every accuracy against the `Majority` row, which always predicts the most common class. On SPECT, 212 of 267 records are in one class, so it scores about 0.79 accuracy with no features.
- Each slide table gives mean ± sd over the outer folds. The `Pooled` columns join all out-of-fold predictions of one repeat and score them once, so a class with 2 or 4 records counts in full. With one repeat (the smoke run) they show a single value.
- Macro-F1 is the mean of the per-class F1 scores, where F1 = 2 precision recall / (precision + recall). On one fold, the mean runs over the classes present in that fold's test records. scikit-learn's default also counts a class that appears only in the predictions, with F1 0. The two differ only on a fold that is missing a rare class.
- `Time (s)` is the time to choose the subset in one fold: the whole search for the optimisers, the ANOVA filter for `ANOVA-k`, and nothing for the other references.
- `<classifier>/cross_dataset.csv` puts the macro-F1 of every method on every task in one table and adds each method's rank averaged over the tasks. The average rank is descriptive. No test stands behind it.

The p-values in `compare_<task>.csv` and `ablation_<task>.csv` are exploratory. The 25 folds of 5 x 5 repeated cross-validation share training records, so they are not independent samples, and the Wilcoxon test treats them as if they were. Read the mean difference, its bootstrap interval and the win, tie and loss counts first. With the smoke run's 5 folds, the smallest possible two-sided Wilcoxon p-value is 0.0625.

## Files

- `fetch_data.py` downloads the six datasets into `data/`, writes `data/SOURCES.txt`, and checks that every task loads with the expected size.
- `arrfs/data.py` holds the seven tasks behind `load_task(task)` and the per-fold preprocessing, which it fits on training rows only.
- `arrfs/fitness.py` is the wrapper fitness (numpy 5-NN over 3 inner folds) and `class_scores`, the one metric function.
- `arrfs/optimizers.py` holds the five optimisers behind `search(name, fitness, X0, budget, rng)`. Every optimiser gets exactly `budget` fitness calls (2,000) and returns a 100-point convergence curve. `_ibgwo4` and `_gwo_cpo` sit side by side and differ only in their second step.
- `arrfs/experiment.py` holds `Settings`, the classifiers, the reference rows, `run_fold` (one outer fold), and the only code that reads or writes a run directory.
- `run_experiments.py` runs the pending folds on a process pool. Only the parent process writes files.
- `analyze.py` turns a run directory into `summary/`.
- `check_optimizers.py` and `check_protocol.py` are the two check scripts above.
- `requirements.txt` pins the library versions installed in `.venv`.

A run directory holds `manifest.json` (the settings and a hash of the `arrfs/` source) and one JSON file per outer fold in `folds/`. A fold file stores predictions and selected attribute indices, never metrics.

`summary/` holds these files:

- `rows.csv` has one row per task, repeat, fold, method, fitness and classifier.
- `<classifier>/slide_<task>.csv` is the review table.
- `<classifier>/recall_<task>.csv` is the pooled recall of each class.
- `<classifier>/compare_<task>.csv` compares GWO-CPO / imbalance with every other row, paired by fold.
- `<classifier>/ablation_<task>.csv` gives the imbalance fitness minus the accuracy fitness for each optimiser.
- `<classifier>/cross_dataset.csv` gives macro-F1 per task and method, with the average rank over tasks.
- `stability_<task>.csv` gives the Nogueira stability index and the mean pairwise Jaccard index of the fold subsets.
- `convergence_<task>.png` plots the mean best fitness so far for each optimiser.
- `leads_<task>.csv` and `leads_<task>.png` give the share of each ECG lead's attributes that were selected. Arrhythmia tasks only.
- `features_<task>.csv` gives how often each of the 279 Arrhythmia attributes was selected. Arrhythmia tasks only. The SHAP step reads it.
