# Synthesis: the design to implement

Base: `design/candidate-1/` (all five files). Implement it at the same paths under `code/` (for example `code/arrfs/optimizers.py`, `code/run_experiments.py`). The changes below override the sketch where they disagree. `code/arrfs/data.py` and `code/arrfs/fitness.py` stay as they are; report any bug found in them instead of working around it.

## Why candidate 1

It stores predictions and derives every metric from them with `fitness.class_scores`, so per-fold and pooled numbers share one definition. Its manifest carries a hash of the `arrfs/` source, so one table never mixes two code versions. Results are JSON, which moves between Kaggle and a laptop. It ships `check_optimizers.py`. Candidate 2 converged on the same overall shape (fold as the unit, one tracker, registry, shared X0, name-based seeds, atomic writes, raw-attribute indices, pooled out-of-fold scores), which is strong agreement.

Kept from candidate 1's open questions, as it chose: GWO-CPO uses CPO's random reset on every branch; leaders are the three best distinct positions found so far; BPSO starts at zero velocity; GWO-CPO reads its leaders live when each porcupine moves.

## Change 1. Equal fitness-evaluation budget, not equal iterations

CPO shrinks its population, so equal iterations give it fewer evaluations (1,776 against 2,020). Every algorithm now gets exactly `budget` fitness calls.

- `Settings`: `population = 20`, `budget = 2000`. Remove `iterations`.
- `search(name, fitness, X0, budget, rng) -> SearchResult(mask, fitness, curve, calls)`.
- `_Recorder` owns the budget. It counts every requested call (cache hits included). The call after the budget is spent raises a private exception, and `search` catches it and returns the best so far. Algorithms contain no budget logic. They loop until stopped.
- `_Recorder.progress` is `calls / budget` in [0, 1]. Every schedule uses it instead of t/T: GWO and BPSO-GWO `a = 2 - 2 * progress`; BPSO `w = 0.9 - 0.5 * progress`; CPO `Yt = 2 rand (1 - p)^p` with p = progress; CPO's cyclic population size `floor(N_min + (N - N_min) * (1 - frac(progress * cycles)))`. This matches my recollection of CPO's MATLAB code, where `t` counts function evaluations and `Tmax` is the evaluation budget. That recollection is unverified until the CPO paper or the authors' code is read; the porcupy port uses iterations instead.
- The curve is the best fitness so far after every `budget // 100` calls, so it has exactly 100 points for every algorithm. `search` asserts `calls == budget` and `len(curve) == 100`.
- BPSO-GWO's docstring: the original binary PSO-GWO hybrid of Al-Tashi et al. 2019, which recent GWO-PSO work builds on. Do not call it "the method base paper".

## Change 2. Score every subset with four classifiers

Each row's selected subset is scored on the outer-test rows with four classifiers fitted on the outer-train rows. KNN is primary because it is the classifier inside the search.

- `CLASSIFIERS = ("KNN", "LogReg", "SVM", "RF")`, defined once. KNN: k = 5. LogReg: `class_weight="balanced"`, `max_iter=5000`. SVM: RBF, `class_weight="balanced"`. RF: 300 trees, `class_weight="balanced"`, `random_state` from the fold seed.
- `MethodResult.y_pred` becomes `dict[str, int[n_test]]` keyed by classifier name.

## Change 3. Reference rows

- `"All features"`: every kept column, scored with the four classifiers.
- `"ANOVA-k"`: `SelectKBest(f_classif, k)` fitted on outer-train rows only, k = the number of features GWO-CPO / imbalance chose in the same fold (assert this), scored with the four classifiers.
- `"Majority"`: `DummyClassifier(strategy="most_frequent")`, stored under the key `"Dummy"` only. Every accuracy on a slide is read against this row.

## Change 4. Analysis additions

- Every table is produced per classifier. The KNN tables are the headline tables.
- Subset stability: the Nogueira, Sechidis and Brown (2018, JMLR 18:174) stability index over the five fold masks of each repeat, in the 279 raw-attribute space, mean ± sd over repeats. Also the mean pairwise Jaccard index.
- Comparisons against GWO-CPO / imbalance: mean difference, win/tie/loss counts over the 25 paired folds, a 95% percentile bootstrap interval of the mean difference (2,000 resamples of the paired folds, fixed seed), and the Wilcoxon p-value in a column named `p_exploratory` with Holm adjustment in `p_holm_exploratory`. Folds share training rows, so the p-values are exploratory. Say so in the README.
- The Majority row appears in the slide tables.

## Change 5. Checks (failure modes from `design/rubric.md`)

`check_optimizers.py`, on a planted-target toy problem:
- exactly `budget` calls, a 100-point non-increasing curve, and returned fitness equal to fitness(returned mask), to `curve[-1]` and to the minimum over everything evaluated;
- the same seed gives the same mask, curve and calls;
- the final fitness beats the best of the initial population, and beats random search with the same budget, on each of 5 seeds.

`check_protocol.py`, with tiny settings (budget 200, one repeat, two algorithms) in a temp directory:
- replacing the outer-test feature values with noise leaves every selected mask unchanged (give `run_fold` an optional `X_raw` argument; `None` loads the dataset);
- deleting one fold file and rerunning recomputes only that fold, and the predictions match the first run exactly;
- every (task, repeat, fold, method, fitness) key appears once, and each record is predicted once per repeat per method;
- `class_scores` matches scikit-learn's accuracy, balanced accuracy and macro-F1 on random label vectors, including classes missing from the predictions;
- the numpy 5-NN in `WrapperFitness` matches `KNeighborsClassifier` on random subsets.

## Change 6. README

`code/README.md` for the teammate: what each file is, the exact commands (check, smoke, full, analyse), the expected time (measured on the smoke run, not guessed), the expected scikit-learn warning that the least populated class has 4 members, Kaggle instructions, and the note that p-values are exploratory.
