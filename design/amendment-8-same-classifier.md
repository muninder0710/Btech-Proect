# Amendment 8: the proposed selection against the base paper's filter, with the same classifier

Decided on 2026-11-21, before this experiment produced any result. It is the last planned experiment. The broad combination search discussed with the students is dropped. Eighty combinations are already measured (five optimisers, two starts, two fitness kinds, four classifiers) and they show that the optimiser changes little.

## Why

The base paper (Dhanka & Maini 2025) selects features with two filters and classifies with XGBoost. Tuned fairly, that pipeline scores 80.3% on the two-class task (Amendment 7). The proposed model so far wraps 5-NN and scores 75.0%. The two differ in the classifier, so the comparison says nothing about the feature selection. This experiment gives both the same classifier. It also covers the twelve-class task, which the base paper names as future work.

## Design

- Tasks. Two classes on the base paper's 420 patients (`small_split_demo.load_420`). Twelve classes on the project's 452 patients (`load_task("arrhythmia-multiclass")`).
- Outer folds. Fresh folds, as the stop rule of Amendment 6 requires: `RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=2026)`, 25 folds per task. Everything below is fitted on the outer-train rows only.
- Classifier for every arm. `xgboost.XGBClassifier` 2.1.4 with default settings, one thread, `random_state=0`.
- Arms.
  - All features. `FoldPreprocessor`, then the classifier.
  - Base filter. Two classes: `small_split_demo.Steps`, the rebuild of the paper's two filters used in experiments 3(c) and 3(d) (189 attributes with the fewest zeros, then the 100 with the largest absolute Pearson correlation, then min-max). Twelve classes: the same two filters after `FoldPreprocessor`, with the ANOVA F statistic in place of Pearson correlation, because Pearson correlation with a twelve-valued label has no meaning. This twelve-class version is our extension, not the paper's.
  - Proposed. `FoldPreprocessor`, then GWO-CPO from `arrfs/optimizers.py` (unchanged, population 20, 2,000 fitness calls) from the informed start of Amendment 6 (`informed_init_demo.informed_start`). The fitness wraps the classifier it serves. It is the usual wrapper fitness with a light XGBoost (50 trees, depth 3) in place of 5-NN, on an inner stratified 3-fold split. Fitness kind: accuracy for two classes, the imbalance fitness for twelve classes.
  - Proposed with 5-NN inside the search. The same, with the original 5-NN fitness. Secondary, to show what wrapping the final classifier changes.
- Secondary, two classes only. The base-filter arm and the proposed arm again with XGBoost tuned fairly by the PSO of Amendment 7 (inner 3-fold on the outer-train rows). Twelve-class tuning is left out for its cost.

## Claims, fixed now

Primary metric: accuracy for two classes (the base paper's metric) and macro-F1 for twelve classes (the project's primary metric). Pairing: the 25 fresh folds. Statistics: `analyze._paired`.

For each task, on the default-classifier arms:
- "The proposed selection beats the base paper's filter" holds if the mean paired difference (proposed minus base filter) is above 0 with two-sided Wilcoxon p below 0.05.
- "The proposed selection matches the base paper's filter with fewer features" holds if the 95% bootstrap interval of that mean difference lies entirely above -0.02 and the proposed arm uses fewer than 50 features on average (the base filter keeps 100).

What the slides say, by outcome.
- The first claim holds. "With the same classifier, the proposed selection scores higher than the base paper's filter."
- Only the second holds. "With the same classifier, the proposed selection matches the base paper's filter with N features in place of 100."
- Neither holds and the interval lies entirely below 0. "The base paper's filter does better on this task."
- Neither holds otherwise. "No clear difference", with the numbers.

Reported with no claim attached: every arm against all features, the 5-NN-inside arm, the tuned two-class arms, and the number of features.

## Stop rule

Whatever this shows, no further variant is tested for a "beats the base paper" claim. The paper is written from the results in hand.

The script is `same_classifier_demo.py`, outside `arrfs/`. Results go to `results/same_classifier/`.
