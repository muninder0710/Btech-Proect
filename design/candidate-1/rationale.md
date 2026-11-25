# Candidate 1: whole-fold work units, one recorder, predictions as the record

## Problem

The students need five feature-selection algorithms, each run with two fitness functions, on two tasks, over 5 x 5 repeated outer cross-validation, plus four reference models. They must be able to explain every number at the review. `arrfs/data.py` and `arrfs/fitness.py` exist and are measured. The optimisers, the runner and the analysis are missing.

Four facts make the shape non-obvious.

- The five algorithms must report the best mask, the best fitness and the convergence curve in exactly the same way. If they don't, the comparison is unfair in a way no table shows.
- CPO updates porcupines one at a time, and its global best changes in the middle of an iteration. A population-level `step()` interface does not fit it.
- `FoldPreprocessor` keeps a different set of columns in each fold. A mask over kept columns means nothing across folds until it is mapped back to the 279 raw attributes, and the per-lead analysis needs that mapping.
- The 12-class task has classes with 4 and 5 records. One test fold holds 0 or 1 of them, so per-fold recall for those classes is noise. Pooled out-of-fold metrics need the stored predictions, not only per-fold scores.

The existing code fixes some things. `WrapperFitness` is a callable from mask to float with its own cache and call counter. `class_scores` is the metric code. The runner must use spawned processes (macOS), one BLAS thread per worker, resume after a crash, and give the same numbers for the same command.

## Usage (caller's view)

README for the teammate:

```
cd code
source .venv/bin/activate
python check_optimizers.py            # each algorithm on a toy problem, a few seconds
python run_experiments.py --smoke     # 1 repeat: 2 tasks x 5 folds    -> results/smoke/
python analyze.py results/smoke       # tables and plots               -> results/smoke/summary/
python run_experiments.py             # 5 repeats: 2 tasks x 25 folds  -> results/full/
python analyze.py results/full
```

If the run crashes, or you press Ctrl-C, run the same command again. It skips every fold that already has a file. If you edited a file in `arrfs/` or changed a setting since the folder was started, the runner renames the old folder to `results/full.stale-<date-time>` and starts again. One table never mixes two versions of an algorithm. On Kaggle, `cd /kaggle/working` and run the scripts by their full path. Results land in `/kaggle/working/results/`.

The smoke run's folds are exactly the full run's repeat 0. I checked this on scikit-learn 1.5.2. `RepeatedStratifiedKFold(5, n_repeats, random_state=42)` gives identical repeat-0 test folds for 1 and 5 repeats.

Call site 1. Trying one algorithm in a notebook. This uses all 452 rows, so it checks the optimiser and is never reported.

```python
import numpy as np
from sklearn.model_selection import StratifiedKFold
from arrfs.data import FoldPreprocessor, load_raw, make_task
from arrfs.fitness import WrapperFitness
from arrfs.optimizers import search

X, y_raw = load_raw()
y, classes = make_task(y_raw, "multiclass")
Xs = FoldPreprocessor().fit_transform(X)
inner = list(StratifiedKFold(3, shuffle=True, random_state=0).split(Xs, y))
fitness = WrapperFitness(Xs, y, len(classes), inner, kind="imbalance")
X0 = np.random.default_rng(0).random((20, Xs.shape[1]))
res = search("GWO-CPO", fitness, X0, T=100, rng=np.random.default_rng(1))
print(res.mask.sum(), res.fitness, len(res.curve), res.calls)   # curve has 101 points, calls is 1776
```

Call site 2. Debugging one fold of the real grid without the process pool:

```python
from arrfs.experiment import FoldKey, Settings, run_fold
result = run_fold(FoldKey("multiclass", repeat=0, fold=3), Settings(repeats=1))
for m in result.methods:
    print(m.method, m.fitness, len(m.selected), round(m.seconds, 1))
```

Call site 3. Reading results in a notebook:

```python
from pathlib import Path
from analyze import load_rows, pooled_scores, slide_table, wilcoxon_vs_proposed
rows = load_rows(Path("results/full"))
print(slide_table(rows, pooled_scores(rows), "multiclass"))
print(wilcoxon_vs_proposed(rows, "multiclass", metrics=("bacc",)))
```

## Shape

### Modules

There are five new files. The longest path a reader traces is three files. `run_experiments.py` calls `arrfs/experiment.py`, which calls `arrfs/optimizers.py`.

- `arrfs/optimizers.py` holds all five algorithms, `search()`, `SearchResult` and the `ALGORITHMS` names.
- `arrfs/experiment.py` holds `Settings`, `FoldKey`, `run_fold`, the result types, and the only code that reads or writes a run directory.
- `run_experiments.py` is the command line, the process pool and the progress lines.
- `analyze.py` is pure functions from one DataFrame to tables and figures, plus `main`.
- `check_optimizers.py` runs three behaviour checks per algorithm on a toy problem with a planted target mask.

### The optimiser interface

There is one public function, `search(name, fitness, X0, T, rng) -> SearchResult(mask, fitness, curve, calls)`. Inside, `search` wraps the fitness in a `_Recorder`. The recorder is the only owner of the best mask, the best fitness, the curve and the call count. Each algorithm is a private function `(rec, X0, T, rng) -> None`. It calls `rec(mask)` to evaluate, and `rec.end_iteration()` once after the initial population and once per iteration. `search` asserts that the curve has T + 1 points, so an algorithm that miscounts fails at once (per encode-lessons-in-structure). Every field of the result means the same thing for all five algorithms, and each algorithm's body holds only its paper's equations.

Where each algorithm lives:

- BPSO is `_bpso`.
- BPSO-GWO, the method base paper, is `_bpso_gwo`, using `sigmoid10` and `_best_three`.
- GWO is `_gwo`, using `_gwo_move` and `_best_three`.
- CPO is `_cpo_family` with `exploit=_odor_or_attack`.
- GWO-CPO, the proposed model, is `_cpo_family` with `exploit=_gwo_hunt`. `_gwo_hunt` calls the same `_gwo_move` as plain GWO.

The registry line `"GWO-CPO": partial(_cpo_family, exploit=_gwo_hunt)` is the whole difference between CPO and the proposed model. That is the sentence on the slide, and an examiner can check it in one line of code (per model-the-domain, a registry instead of branches).

CPO's memory is one pair of arrays, `X` and `f`. Under greedy acceptance a porcupine's position equals its personal best once its turn ends, and the global best is always `X[argmin f]`. So there is no separate personal-best array or stored global best to drift out of sync.

The CPO family spends 1,776 fitness calls per search and the other three spend 2,020. I computed this from the population-reduction formula with N = 20 and T = 100. The `calls` column reports it.

### Adding an algorithm

Write `_woa(rec, X0, T, rng) -> None` in `optimizers.py`. It evaluates through `rec` and calls `rec.end_iteration()` T + 1 times. Then add `"WOA": _woa` to `_ALGORITHMS`. `Settings.algorithms`, the runner's grid, the analysis tables, the Wilcoxon comparisons and `check_optimizers.py` all read `ALGORITHMS` or the saved rows, so no other file changes. The edit changes the source fingerprint, so the next run starts a fresh folder and reruns the whole grid. The grounding's numbers put that near two minutes on 10 cores. That is an estimate from 50 folds x 10 searches x about 2,000 calls x about 1 ms per call.

### Results schema

There is one JSON file per outer fold. Facts about the fold are stored once, facts about each method once per method.

```
FoldResult
  key           FoldKey(task, repeat, fold)
  class_names   list[str]
  test_idx      int[n_test]    rows of the 452-record file
  y_true        int[n_test]
  kept          int[D]         raw attributes that survived preprocessing on this fold's training rows
  methods       list[MethodResult], 14 per fold: 5 algorithms x 2 fitness kinds, then 4 references

MethodResult
  method        "BPSO" ... "GWO-CPO", or "KNN-all", "LogReg-all", "SVM-all", "ANOVA-KNN"
  fitness       "accuracy" | "imbalance" | "none" (references)
  y_pred        int[n_test], aligned with test_idx
  selected      int[k], indices into the 279 raw attributes
  seconds       wall time to produce y_pred, the only field that varies between runs
  best_fitness, calls, curve (float[T + 1])    search rows only
```

Metrics are not stored. `analyze.load_rows` derives accuracy, balanced accuracy, macro-F1, worst recall and per-class recall from `y_pred` with the existing `fitness.class_scores`. `pooled_scores` runs the same function on each repeat's concatenated out-of-fold predictions. One metric definition serves both, so they cannot disagree (per foundational-thinking, data first). `selected` uses raw attribute indices, so the lead and feature frequency tables compare folds directly. JSON round-trips floats exactly and does not depend on the pandas version, which matters if the run happens on Kaggle and the analysis on a laptop.

### Runner, resume and determinism

The unit of parallel work is one outer fold, 50 in the full run. A fold preprocesses once, builds its inner folds and `X0` once, then runs the 10 searches and the 4 references. ANOVA-KNN needs the number of attributes GWO-CPO/imbalance chose in the same fold. Keeping the fold whole keeps that dependency inside one function.

Workers only compute. The parent process is the only writer of the run directory (per separate-before-serializing-shared-state). It saves each `FoldResult` as it arrives, through a temp file and `os.replace`, so a fold file is either complete or absent. A fold that raises is printed and skipped, and the exit code becomes 1.

Resume follows from that. `prepare_run_dir` looks at what exists and returns the folds that have no file (per make-operations-idempotent). Running it twice, or after a crash at any point, ends in the same folder. Leftover `.tmp` files never count. The manifest stores the settings and a sha256 of `arrfs/*.py`. On a mismatch the runner renames the old folder aside instead of resuming into it. Nothing is deleted.

Seeds come from sha256 of `(task, repeat, fold)` for the fold and `(task, repeat, fold, algorithm, kind)` for a search. Python's `hash()` of a string changes per process, so it is not used. Results do not depend on which worker ran a fold or in what order. Each search gets a fresh `WrapperFitness`, so no search inherits another's cache and wall times do not depend on run order. The pool uses the spawn start method on every platform, so Kaggle's Linux behaves like macOS. `run_experiments.py` sets four BLAS thread variables to 1 before numpy is imported, and spawned workers inherit them.

### Boundaries and what the design leaves out

The command line parses into a frozen `Settings`, and `load_run` parses JSON into the result types (per boundary-discipline). Code inside trusts the types. There are no flags to choose algorithms or tasks, no logging framework and no config file. `--smoke` is the only reduced run. SHAP is a later step and reads `features_<task>.csv`.

### Interface depth

The optimiser module exposes `search` and `ALGORITHMS`. Behind them sit five algorithms, the bookkeeping and the iteration contract. The experiment module exposes `Settings`, `FoldKey`, `run_fold` and three run-directory functions. Behind `run_fold` sit the outer split, preprocessing, inner folds, seeding, 14 methods and the final evaluation. A caller never needs the leader rules, the population reduction or the JSON layout. The analysis exposes one function per output, which is wide, but each function is one table the review needs and none calls another's internals.

## Synthesis decision

Filled in by arena.

## Tradeoffs accepted

- We accept a mutable `_Recorder` handed to each algorithm in exchange for a single owner of the best mask, curve and call count. Five algorithms each returning their own curve would mean five definitions of "best so far".
- We accept that any edit under `arrfs/` reruns the whole grid in exchange for never mixing two code versions in one table. The run is short enough (estimate above) that this costs minutes.
- We accept recomputing metrics in the analysis in exchange for one metric definition shared by per-fold and pooled numbers.
- We accept three search-only fields that are `None` on reference rows in exchange for one flat row type that matches the results table.
- We accept coarse parallel units (50 folds on 10 workers, so the last wave can leave cores idle) in exchange for keeping each fold's preprocessing and the ANOVA dependency in one place.
- We accept the papers' budgets (the same N and T for all) rather than equal fitness calls. The CPO family uses 1,776 calls against 2,020, and the `calls` column shows it.
- We accept losing cache reuse across searches in exchange for wall times that do not depend on run order.
- We accept that `Settings.fitness_kinds` and the `kind` branch in `WrapperFitness` both list the two kinds. A third kind means editing both, and `WrapperFitness` raises on an unknown kind, so a mismatch fails at the first call.

## Alternatives considered

- An `Optimizer` base class with `init()` and `step()` and five subclasses. It looks uniform but hides less. CPO updates one porcupine at a time with a live global best, so its `step()` would either break the paper's sequential update or carry private state the base class cannot describe. The teammate would also trace an inheritance chain for what are five functions.
- Ask-and-tell generators, where each algorithm yields masks and the driver sends fitness values back. This removes the recorder, but CPO yields one porcupine at a time, so iteration boundaries need a sentinel value, and `generator.send()` is unfamiliar to the teammate.
- One work unit per (fold, algorithm, fitness kind), with workers appending rows to one shared CSV. It balances load better. But every worker writes one file, which races unless locked, preprocessing repeats 14 times per fold, and the ANOVA reference must wait for another unit's output.
- Storing metrics instead of predictions, or pickling a pandas DataFrame. Stored metrics cannot give pooled scores, and pickles break between pandas versions.

## Open questions and risks

- The grounding's GWO clips to [0, 1], while CPO resets out-of-range coordinates to random values. This design has GWO-CPO use CPO's random reset on every branch, because the grounding says "CPO exactly as in 4" apart from the exploit step. Is that the intended reading?
- This design keeps the three best distinct positions found so far as leaders. Mirjalili's original code does not demote alpha to beta when a better wolf appears. Should the method notes say which rule the code uses?
- BPSO's initial velocity is 0 here, because Kennedy and Eberhart do not fix it. Is 0 acceptable, given that the method notes already state BPSO-GWO's 0.3 N(0, 1)?
- GWO-CPO reads its leaders from the memory at each porcupine's move, not once per iteration. This matches CPO's live global best. Should the method description say so?
- The Wilcoxon tests pair 25 folds that share training records, so the p-values are optimistic (Nadeau and Bengio 2003 treat this problem). On the 5 pooled repeats, the smallest possible two-sided Wilcoxon p is 0.0625, so that test can never reach 0.05. Should the slide show fold-paired p-values with this caveat, or only effect sizes and win counts?
- "Same command, same numbers" holds on one machine. Kaggle's BLAS can round distances differently, and an exact KNN distance tie could then break the other way. Should the review numbers come from one named machine?
- One of the 12 classes has 4 records, fewer than the 5 outer folds, so scikit-learn warns on every multiclass split. The warning is expected, and the README should say so.

## Next implementation step

Implement `_Recorder`, `_best_three`, `_gwo_move` and `_gwo`, then make `check_optimizers.py` pass for GWO before writing the other four algorithms.
