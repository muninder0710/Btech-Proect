# Grounding: feature-selection experiment for UCI Arrhythmia

## Who this is for

Two final-year B.Tech students (DTU) present this at a project review and write a paper from it. They will explain every number on a slide, and an examiner will ask "how did you get this?". The next person to own the code is the teammate, who knows Python and scikit-learn but not this codebase. The code must therefore be short, readable, deterministic, and rerunnable with one command.

## The research claim being tested

Base paper (method): Al-Tashi et al. 2019, "Binary Optimization Using Hybrid Grey Wolf Optimization for Feature Selection", IEEE Access 7:39496. Binary PSO-GWO hybrid ("BGWOPSO"). In their words, it uses "exploration of PSO and exploitation of GWO". KNN (k=5) wrapper, fitness 0.99 * error + 0.01 * |S|/|T|, 10 wolves, 100 iterations, 20 runs, 18 UCI datasets (not Arrhythmia).

Base paper (dataset): Yao et al. 2024, HSPSO, Sci. Rep. 14:24928. PSO with four add-ons, SVM, 2-class Arrhythmia task, 80.21% accuracy with 6 features. It compared its 2-class result against 13-class numbers copied from Li, Xue & Zhang 2021.

Proposed model: swap PSO (the explorer) for the Crested Porcupine Optimizer (Abdel-Basset et al. 2024, Knowledge-Based Systems 284:111257). CPO's first two defences (sight, sound) explore. GWO's three-leader hunting exploits. Plus an imbalance-aware fitness, plus SHAP explanations grouped by ECG lead (later step, out of scope here).

## Existing code (read these files)

- `code/arrfs/data.py`. `load_raw()` returns X (452 x 279 float, NaN for missing) and raw labels. `make_task(y_raw, "binary" | "multiclass")` returns integer labels 0..C-1 and class names. Binary is normal (245) vs arrhythmia (207). Multiclass has 12 classes (labels 7 and 8 merged), counts 245 44 15 15 13 25 5 9 50 4 5 22. `FoldPreprocessor` is fitted on a training fold only. It drops columns with more than 50% missing and constant columns, imputes medians, and min-max scales. About 257 of 279 columns survive. `feature_names_and_leads()` gives the 279 names and the ECG lead of each ("General" or one of 12 leads).
- `code/arrfs/fitness.py`. `WrapperFitness(X, y, n_classes, inner_folds, kind)` is a callable `mask (bool[D]) -> float`, lower is better. It runs numpy 5-NN over the given inner folds, caches by mask bytes, and counts calls in `.calls`. `kind="accuracy"` is the base papers' fitness. `kind="imbalance"` is 0.7 * (1 - macro-F1) + 0.3 * (1 - balanced accuracy) + 0.01 * |S|/D. An empty mask returns 2.0. `class_scores(y, yhat, C)` returns acc, bacc, macro_f1, worst_recall, per-class recall.

Measured on this machine (10 cores, 16 GB, Python 3.12, numpy 2.1.3, scikit-learn 1.5.2): one fitness call takes 0.8 to 1.0 ms with half the features selected. The numpy KNN predictions matched scikit-learn's KNeighborsClassifier on 40 random subsets with 0 mismatches. With all features, inner-CV 5-NN gives 2-class accuracy 0.64 and 12-class accuracy 0.59 with balanced accuracy 0.18.

## Algorithms to implement (all minimise fitness over D binary decisions)

Shared settings: population N = 20, iterations T = 100. Every algorithm in one fold starts from the same initial population `X0 = rng.random((N, D))`. Binary-native algorithms use `X0 > 0.5`. Continuous ones use X0 directly and decode a position with `x > 0.5`.

sigmoid10(z) = 1 / (1 + exp(-10 (z - 0.5))).

1. Binary PSO (Kennedy & Eberhart 1997). v = w v + c1 r1 (pbest - x) + c2 r2 (gbest - x), w linear 0.9 to 0.4, c1 = c2 = 2, |v| <= 6, x = rand < 1 / (1 + exp(-v)).

2. Binary PSO-GWO (Al-Tashi 2019, equations 16 to 21), the method base paper. Leaders alpha, beta, delta are the best three binary positions found so far. Each iteration a = 2 - 2t/T and w = 0.5 + rand/2. For each leader L with A = 2 a r - a and C = 0.5: D_L = |C * X_L - w * X|, cstep = 1 / (1 + exp(-10 (A * D_L - 0.5))), bstep = cstep >= rand, x_L = (X_L + bstep) >= 1. Then v = w (v + c1 r1 (x_1 - X) + c2 r2 (x_2 - X) + c3 r3 (x_3 - X)) with c1 = c2 = c3 = 0.5, and X = sigmoid10((x_1 + x_2 + x_3) / 3 + v) >= rand. Initial velocity is not specified in the paper. Use 0.3 * N(0, 1) and say so in the method notes.

3. GWO with threshold decoding (Mirjalili 2014). Continuous positions in [0, 1]. a = 2 - 2t/T. For each leader, A = 2 a r1 - a, C = 2 r2 (vectors), D = |C X_L - X|, X_L' = X_L - A D. X = mean of the three, clipped to [0, 1]. Leaders are the best three found so far.

4. CPO with threshold decoding (Abdel-Basset 2024; checked against the public port `porcupy` 0.2.0). Continuous in [0, 1]. Keeps a personal-best copy per porcupine and accepts a new position only if its fitness is not worse (greedy). Parameters: N_min = round(0.8 N), cycles = 2, alpha = 0.2, Tf = 0.8. Per iteration: r2 = rand once. Per porcupine i, U1 = rand(D) > rand.
   - With probability rand < rand, explore:
     - With probability rand < rand, sight: y = (X_i + X_r) / 2, X_new = X_i + randn * |2 rand gbest - y|.
     - Else, sound: y = (X_i + X_r) / 2, X_new = U1 X_i + (1 - U1)(y + rand (X_r1 - X_r2)).
   - Else, exploit, with Yt = 2 rand (1 - t/T)^(t/T), U2 = ±1 per dimension, S = rand U2:
     - With probability rand < Tf, odor: St = exp(f_i / (sum f + eps)), S = S Yt St, X_new = (1 - U1) X_i + U1 (X_r1 + St (X_r2 - X_r3) - S).
     - Else, physical attack: Mt = exp(f_i / (sum f + eps)), Ft = rand(D) * Mt (X_r - X_i), S = S Yt Ft, X_new = gbest + (alpha (1 - r2) + r2)(U2 gbest - X_i) - S.
   - Any coordinate outside [0, 1] is replaced by a uniform random value.
   - After each iteration, the active population size is floor(N_min + (N - N_min)(1 - (t mod (T/cycles)) / (T/cycles))). Only the first that many porcupines update.
   Note (unverified, from memory of the authors' MATLAB code; check against the paper or the official code): the original MATLAB may write `U2=rand(1,dim)<0.5*2-1`, which by operator precedence is all zeros. Use the intended ±1, as the porcupy port does.

5. GWO-CPO (proposed). CPO exactly as in 4, except that the exploit branch (odor and physical attack) is replaced by the GWO three-leader step from 3, with leaders taken from the personal-best memory and a = 2 - 2t/T. Exploration (sight, sound), greedy acceptance and cyclic population reduction stay CPO's. This mirrors the base paper's split: an explorer (PSO there, CPO here) plus GWO's exploitation.

## Evaluation protocol

- Tasks: "binary" and "multiclass".
- Outer loop: `RepeatedStratifiedKFold(n_splits=5, n_repeats=R, random_state=42)`. R = 5 for the full run, R = 1 for a smoke run.
- Per outer fold: fit `FoldPreprocessor` on the outer-train rows. Inner folds are `StratifiedKFold(3, shuffle=True, random_state=<fold seed>)` on outer-train labels. One shared X0 per fold.
- Grid per fold: 5 algorithms x 2 fitness kinds ("accuracy", "imbalance"). The algorithm returns its best mask by fitness. Then 5-NN (scikit-learn, k = 5) is fitted on the outer-train rows with that mask and predicts the outer-test rows.
- Reference rows per fold, with no search: 5-NN on all features; logistic regression (class_weight="balanced", max_iter 5000) on all features; RBF SVM (class_weight="balanced") on all features; ANOVA `SelectKBest(f_classif, k)` + 5-NN, where k is the number of features GWO-CPO picked with the imbalance fitness in the same fold.
- No choice of variant, weight or setting may look at outer-test results. Settings are fixed above before any run.

## What the results must support

One row per (task, repeat, fold, method, fitness kind). Columns needed: accuracy, balanced accuracy, macro-F1, per-class recall, number of features, the selected mask (or indices), wall time, fitness calls, best fitness, and the convergence curve. Out-of-fold predictions must also be kept so pooled metrics per repeat can be computed (rare classes appear in only a few test folds).

The analysis then produces:
- A slide table, mean ± sd over folds, per task.
- Wilcoxon signed-rank tests, GWO-CPO + imbalance against every other row, paired by fold.
- Accuracy fitness vs imbalance fitness for each algorithm (the ablation for innovation 2).
- How often each ECG lead's features are selected (input to the later SHAP step).
- Convergence curves.

## Constraints

- Deterministic. The same command gives the same numbers. Seeds are derived from (task, repeat, fold, algorithm, fitness kind).
- Parallel over folds with processes (macOS uses spawn). Set BLAS threads to 1 per worker.
- Restartable. A crashed or interrupted run resumes without redoing finished folds.
- Target under 30 minutes for the full run on 10 cores. That is about 2 tasks x 25 folds x 10 configs x ~2,000 fitness calls at ~1 ms, plus the reference models.
- One command to run, one command to analyse. A short README for the teammate. The same scripts must run on Kaggle (CPU) unchanged.
- Python 3.12, numpy, scipy, pandas, scikit-learn, matplotlib are installed in `code/.venv`. No other packages.
