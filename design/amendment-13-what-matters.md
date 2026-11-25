# Amendment 13: which choices matter on UCI Arrhythmia

Decided on 2026-11-24, before this experiment produced any result on the arrhythmia data. The code was first run on synthetic data only, to find bugs.

## Why

Papers on UCI Arrhythmia change missing-value handling, feature selection and the classifier at once, and report one number. Phatai et al. 2026 (IEEE ICCI, doi 10.1109/ICCI68752.2026.11506453) separated these choices for two classes, one filter with a fixed 50 features, and three classifiers, and found that missing-value indicators help SVM but not trees. They name multi-class as future work. This experiment crosses the three choices for two and twelve classes under nested cross-validation. It also tests one new idea: the missing-value indicators are candidate features that any selector may pick.

This is a new question, so it gets fresh outer folds. The stop rule of Amendment 8 still holds: nothing here is a new "beats the base paper" claim.

## Design

- Data. UCI Arrhythmia, `data/arrhythmia/arrhythmia.data`, all 452 patients. Two classes: `load_task("arrhythmia-binary")` (normal 245, abnormal 207). Twelve classes: `load_task("arrhythmia-multiclass")` (labels 7 and 8 merged).
- Outer folds. `RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=2027)`, 25 folds per task. Inner split: `StratifiedKFold(3, shuffle=True, random_state=key.seed())` on the outer-train rows. Everything below is fitted on outer-train rows only.
- Knob M, missing values (2 levels).
  - `median`: `FoldPreprocessor` as in every earlier run (drop columns more than half missing in the training fold, median fill, min-max scaling, drop constant columns).
  - `median + indicators`: the same, plus one 0/1 column per raw attribute that has at least one missing value in the training fold (1 = missing). Indicators of attributes that the preprocessor drops are kept. An indicator that is constant in the training fold is dropped.
- Knob S, feature selection (8 levels).
  - `none`: all columns.
  - `ReliefF`, `mutual information`, `mRMR`: filters. ReliefF uses 10 nearest hits and misses, Manhattan distance, misses weighted by class prior (Kononenko 1994), all training rows as samples. Mutual information is `mutual_info_classif` with a fixed seed. mRMR is the greedy difference form of Peng, Long and Ding 2005 on features discretised into 10 equal-frequency bins fitted on the training rows. Each filter's number of kept features is chosen per classifier from {5, 10, 20, 30, 50, 100} by mean macro-F1 over the inner 3 folds, with the ranking recomputed on each inner-train part. Ties go to the smaller number.
  - Four wrappers, each with 5-NN inside on the inner 3 folds, population 20, 2,000 fitness calls (`arrfs/optimizers.py` and `extra_optimizers.py`, unchanged):
    - `IBGWO4 as published`: accuracy fitness, random start (the method base paper).
    - `GWO-CPO guided`: imbalance fitness, mutual-information start of Amendment 6 (the earlier proposed model).
    - `BPSO guided`, `GA guided`: imbalance fitness, the same start.
  - The subset a wrapper finds is given to every classifier.
- Knob C, classifier (4 levels), default settings, no tuning: 5-NN (`KNeighborsClassifier(5)`), SVM (`SVC`, RBF, `class_weight="balanced"`), random forest (`RandomForestClassifier(300 trees)`), XGBoost (`XGBClassifier` 2.1.4 defaults, as in Amendment 8). Fixed seeds, one thread.
- Every fold therefore scores 2 × 8 × 4 = 64 pipelines per task. Rows record accuracy, balanced accuracy, macro-F1, per-class recall, the number of features, and how many kept features are indicators.

## Questions and decision rules, fixed now

Primary metric: macro-F1, both tasks. Accuracy is reported for comparison with the literature. Pairing: the 25 folds. Statistics: `analyze._paired` (mean difference, 95% bootstrap interval, two-sided Wilcoxon), Holm adjustment where stated. Each question is answered separately for each task.

- Q1, missing indicators. For each classifier with no selection: `median + indicators` minus `median`. "Indicators help classifier X" holds if the mean difference is above 0 and the Holm-adjusted p (over the 4 classifiers) is below 0.05; "indicators hurt" likewise below 0. Otherwise "no clear difference". The same comparison for every selector is reported without a claim.
- Q2, selection. With `median`, for each classifier, each of the 7 selectors minus `none`. "Selector Y helps classifier X" holds if the mean difference is above 0 and the Holm-adjusted p (over the 7 selectors of that classifier) is below 0.05; "hurts" likewise.
- Q3, optimiser choice. With `median`, for each classifier, the three guided wrappers pairwise. "The guided optimisers are interchangeable for classifier X" holds if all three 95% intervals lie inside [-0.03, 0.03]. Also `GWO-CPO guided` minus `IBGWO4 as published` with the 5-NN classifier, as a replication of the earlier headline on fresh folds: holds if the mean is above 0 with p below 0.05.
- Q4, which knob matters most. On the 25 × 64 fold scores, after removing each fold's mean, the sums of squares of the balanced design give the share of variance of M, S, C, and the interactions M×C and S×C (the rest is residual). The shares get 95% bootstrap intervals by resampling folds. "Knob A matters more than knob B" holds if the interval of share(A) minus share(B) lies above 0.
- Q5, missingness-aware selection. With `median + indicators`, for each selector except `none`: observed number of indicator columns kept minus the number expected if the same number of columns were drawn at random (kept × indicators / columns). "Selector Y picks indicators more often than chance" holds if the mean is above 0 with Wilcoxon p below 0.05.

## Figures, fixed now

1. Heatmap of mean macro-F1, selector × classifier, per task and missing-value level.
2. Indicator effect (Q1, all selectors): paired differences with 95% intervals.
3. Selection effect (Q2): each selector minus none, per classifier.
4. Variance shares (Q4) with intervals.
5. The 25 fold scores of every pipeline with no selection and with `GWO-CPO guided`, to show spread.
6. Number of features against macro-F1.
7. Per-class recall, twelve classes, for every classifier with no selection and `median`.
8. How often each indicator column is kept, per selector.
9. Twelve-class confusion matrix of XGBoost, all features, `median`, summed over the 25 folds.
10. Mean convergence curves of the four wrappers.

Plus a table of two-class accuracy next to Phatai et al.'s published numbers, marked as not paired (different folds).

## Stop rule

After the run, no setting, level or rule above changes. Anything computed beyond this list is labelled exploratory in the write-up.

The script is `what_matters_demo.py`, its checks are `check_what_matters.py`, and the results go to `results/what_matters/`.
