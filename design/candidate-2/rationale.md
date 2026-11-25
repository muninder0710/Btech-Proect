# Candidate 2: one file per fold, algorithms as registered functions

## Problem

The students need real, defensible numbers for the Arrhythmia feature-selection claim (GWO-CPO with an imbalance-aware fitness). The code to produce them does not exist: five search algorithms, a runner over tasks x 25 outer folds x algorithms x fitness kinds plus reference models, and an analysis step. The base paper's archive is unusable, so every algorithm is written from its equations. Constraints from `grounding.md` that shape the design: deterministic seeds derived from (task, repeat, fold, algorithm, kind), parallel over folds with spawn processes and one BLAS thread each, restartable, under 30 minutes on 10 cores, one command to run and one to analyse, the same scripts on Kaggle, and numpy, scipy, pandas, scikit-learn and matplotlib only. `data.py` and `fitness.py` already exist and stay as they are. `WrapperFitness` is a callable `mask -> float` with a per-instance cache and a `.calls` counter.

## Usage (caller's view)

Commands, run from `code/`:

```
.venv/bin/python run_experiments.py --repeats 1 --out results/smoke   # a few minutes
.venv/bin/python run_experiments.py --repeats 5 --out results/full    # the review run
.venv/bin/python analyze.py results/full                              # tables and figures
```

If the machine sleeps or the run is killed, run the same command again. Finished folds are skipped. A smoke run can be grown into the full run by re-running with `--repeats 5` on the same `--out` (this relies on `RepeatedStratifiedKFold` giving the same first repeat for 1 and 5 repeats, which implementation must check with a test).

Call sites a teammate will actually write:

```python
# 1. Adding an algorithm: one function in arrfs/optimizers.py, nothing else.
@register("de", "Binary DE")
def de(problem: Problem, rng: np.random.Generator) -> SearchResult:
    ...  # start from problem.X0, evaluate through _Tracker, return tracker.result()

# 2. Re-running one search by hand (debugging, or a notebook cell).
from arrfs.optimizers import REGISTRY, Problem
res = REGISTRY["gwo_cpo"].run(Problem(fitness, X0, iters=100), np.random.default_rng(0))
res.best_mask, res.best_fitness, res.curve

# 3. Answering an examiner from the saved results.
from arrfs.results import load_rows
rows = load_rows(Path("results/full"))
rows.query("task == 'multiclass' and method == 'gwo_cpo'")[["fold", "fitness", "macro_f1", "n_features"]]
```

### Results schema

One pickle per outer fold at `<out>/<task>/r<repeat>_f<fold>.pkl`, holding a `FoldResult`: `test_idx`, `rows` (dict keyed by `MethodKey(method, fitness)`) and `oof` (out-of-fold predictions aligned with `test_idx`). `load_rows` flattens this into a DataFrame with one row per (task, repeat, fold, method, fitness). Columns: `acc, bacc, macro_f1, worst_recall, recall (array over classes), n_features, selected (indices in the original 279 columns), seconds, calls, best_fitness, curve`. Reference models use `fitness="none"`, `calls=0`, `curve=None`, `best_fitness=nan`. `selected` is in original-column coordinates (mapped through `FoldPreprocessor.keep_` inside `run_fold`) so the lead-frequency table and the later SHAP step never need to know which columns a fold dropped. `<out>/manifest.json` holds the `Settings` the directory was created with.

### Where each algorithm lives

All in `arrfs/optimizers.py`: `bpso`, `bpso_gwo`, `gwo`, `cpo`, `gwo_cpo`. `cpo` and `gwo_cpo` are two-line wrappers over `_cpo_search(problem, rng, exploit="cpo" | "gwo")`, so they differ in exactly the one branch the research claim is about. `gwo` and `gwo_cpo` share `_gwo_move`, so the GWO exploitation step has one definition.

### How a new algorithm is added

Write the function with `@register(name, label)`. The grid (`expected_keys`), the tables and the plot labels derive from `REGISTRY`, so no list elsewhere needs editing. Seeds use the algorithm name, so existing results do not change. Re-run `run_experiments.py` on the existing `--out`: every fold file lacks the new keys, so `run_fold` computes only those and merges them, leaving finished algorithms untouched.

### How the runner resumes after a crash

Each worker writes only its own `r<repeat>_f<fold>.pkl`, via a temp file and `os.replace`. A file is therefore either absent, complete from an older run, or complete from this one, never half-written. On start, `pending_jobs` keeps the folds whose file is missing, unreadable, or missing any key in `expected_keys()`. Only those are submitted. A fold that was killed mid-way is redone from the start (about 1 to 2 minutes of work at most). A fold that raised is printed, skipped, and retried by the next run. Settings changes cannot silently mix: `manifest.json` is compared on every start and a mismatch stops the run with a message to choose a new `--out`.

## Shape

Core data: `Settings` (frozen, everything that changes a number), `FoldJob` (task, repeat, fold, train and test indices), `Problem` (fitness, `X0`, iters), `SearchResult` (best mask, best fitness, curve), `Row` and `FoldResult`.

Flow, three files deep: `run_experiments.main` -> `experiment.run_fold` -> `REGISTRY[name].run`. Results go back out through `results.save_fold`. Analysis reads through `results.load_rows` and never touches the optimisers except for labels.

Decisions:

- **A fold is the unit of work, parallelism, restart and storage.** One definition of "a unit" means the pool, the resume check and the file layout cannot disagree. The reference models need the k that GWO-CPO + imbalance chose in the same fold, which is free when the fold is the unit and awkward in any finer grain. Per separate-before-serializing-shared-state: no two workers share a file, so there is no lock and no merge step at write time.
- **Algorithms are functions with one signature, held in a registry.** `Problem` carries `X0` and `iters`, so an algorithm cannot invent its own start or budget (the fairness rule in grounding). `_Tracker` is the only code that calls the fitness during a search, so "best mask by fitness" and "curve is non-increasing with length T+1" hold for every algorithm without trust. Per encode-lessons-in-structure and make-illegal-states-unrepresentable (type-system-discipline).
- **Fitness calls and wall time are measured by the runner, not reported by the algorithm.** One owner per measurement. The one `WrapperFitness` per kind per fold is shared across algorithms so its cache helps; `calls` counts requests including cache hits, which keeps it comparable between algorithms.
- **Seeds come from `SeedSequence` over crc32 of names and integer indices**, never from `hash()` (randomised per process) and never from list position. Per make-operations-idempotent: running a fold twice, in any order, on any worker gives the same bytes.
- **Boundaries.** Dataset and `Settings` are validated once in `main` and `check_manifest`. Inside `run_fold` and the optimisers, types are trusted. Pickle is the file format; it never crosses a machine or version boundary except Kaggle, where the same Python and numpy majors apply, and `load_fold` treats an unreadable file as missing.
- **BLAS threads** are pinned in `arrfs/__init__.py`, which every spawn worker imports before numpy. One place, no per-worker initialiser.

Interface depth: the whole public surface the student touches is three commands, `REGISTRY`, `register`, and `load_rows`. Hidden behind it are seeding, preprocessing per fold, inner folds, the shared X0, cache sharing, back-mapping masks to original columns, atomic writes, resume and merge, and the one-k-per-fold coupling to ANOVA. Exposed on purpose: the `Problem`/`SearchResult` contract, because that is what someone adding an algorithm must learn, and the DataFrame columns, because the examiner questions will be asked against them.

Deliberately not done: no config files or CLI per-hyperparameter flags (all settings are fixed before any run, per grounding, so they live in one frozen dataclass), no SHAP, no logging framework, no abstract base classes.

## Synthesis decision

Left blank for arena. This is candidate 2 of several.

## Tradeoffs accepted

- We accept losing the work of a fold killed mid-way (1 to 2 minutes) in exchange for no checkpoint format inside a fold.
- We accept pickle files in exchange for storing numpy arrays, masks and curves with no schema code. They are not human-readable, so `load_rows` is the reading path.
- We accept parallelism of only 25 folds per task (50 jobs on 10 cores, five waves) in exchange for the simple unit. Load can be uneven because multiclass searches may differ in cost from binary ones; if the tail looks bad, sort jobs longest-first.
- We accept that `seconds` includes contention between 10 processes, so it ranks algorithms but is not a clean benchmark. The table footnote must say so.
- We accept a shared fitness cache across algorithms in a fold. It saves time and does not change any fitness value.
- We accept that `anova_knn` makes the reference depend on a proposed-method result (as grounding requires), so a fold is complete only when both exist.

## Alternatives considered

- **Class hierarchy `Optimizer` ABC with `step()` and a shared loop in a base class.** Rejected. The five algorithms disagree on state (velocities, leaders, personal bests, population reduction), so the base class either exposes that state or forces overrides, and the interface gets larger than a single function. The shared parts that are truly shared (`_Tracker`, `_gwo_move`, `_cpo_search`) are helpers.
- **One flat results file (a single CSV or SQLite appended by all workers).** Rejected. It needs a lock or a queue back to the parent, and it cannot hold curves, masks and predictions without extra tables. Per-fold files remove the shared writer.
- **Job grain of one (fold, algorithm, kind) search.** Rejected. It gives 500 jobs and better load balance, but the X0 sharing, the ANOVA k coupling, the shared fitness cache and the reference models would all need a coordinator step, and results would be thousands of files.
- **Resume by a "done" ledger file.** Rejected. A ledger and the result files can disagree. The result file is the only truth, so completeness is derived from it.
- **YAML experiment config.** Rejected as machinery for a protocol whose settings must not change.

## Open questions and risks

- Does `RepeatedStratifiedKFold(random_state=42)` give an identical first repeat at `n_repeats=1` and `n_repeats=5`? Growing smoke into full depends on it. A test should assert it.
- Should the paired Wilcoxon test treat the 25 folds as independent? They overlap in training data, so p-values are optimistic. Is a per-repeat pooled comparison (5 pairs, too few for Wilcoxon) or a corrected resampled test preferred for the paper?
- Is `acc, bacc, macro_f1` per fold enough for the 12-class task, where some classes have 4 to 5 records and many test folds contain none? `pooled_table` addresses it; which should be the headline number on the slides?
- Is crc32 collision risk acceptable for tag hashing? With a handful of names it is negligible, but the implementation could assert uniqueness in `register`.
- Do 50 spawn processes' startup costs matter on Kaggle's 4 CPUs? The full run will take longer there, and the user should set expectations.

## Next implementation step

Implement `_Tracker`, `register`, `seed_for` and `final_scores` with tests (curve non-increasing, same seed gives same result, `fitness` identical across processes), then `bpso` end to end so one fold can run through `run_fold` before the other four algorithms are written.
