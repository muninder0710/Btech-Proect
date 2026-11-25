# Amendment 5: GWO-CPO against IBGWO4 under the base paper's own protocol

Decided on 2026-11-20, after the main results were read (claim 1 came out "comparable") and before this check produced any result. It is an exploratory robustness check. It adds no headline claim, and the verdict of claim 1 does not change whatever this check shows.

## Why

The students asked whether GWO-CPO failed to beat IBGWO4 because our testing differs from the base paper's. In the main experiment both methods were tested the same way, so the protocol cannot favour one of them by itself. It could still interact with them. The base paper uses a smaller population, and it does not say which rows its fitness is scored on. So both methods run again under the base paper's protocol.

## The base paper's protocol (Khaseeb et al. 2025)

- Split, p. 13. "the associated instances were randomly split into two different subsets, namely, 80% training subsets and 20% different testing subsets."
- Settings, Table 2 on p. 14. T = 100 iterations, R = 30 runs, v = 0.99, P = 10, c1 = c2 = 2, K = 5 for KNN.
- Fitness, Eq. 15 on p. 12. Fit = v ClassErr + (1 - v) |q| / |Q|, with ClassErr from KNN.
- Metric, Eq. 16 on p. 12. ACA, the test accuracy averaged over the R runs.
- Initial population, p. 14. It "needed to be the same to enable a fair comparison across all compared approaches."
- Not stated. Which rows ClassErr is computed on during the search, and whether the 30 runs share one split.

The main experiment already matches several of these. Each of its 25 outer folds is an 80/20 split scored with 5-NN, and IBGWO4 there uses this exact fitness. What differs is the population (20 against 10), the fitness GWO-CPO used in claim 1 (imbalance against accuracy), and the split scheme (stratified folds against random splits).

## Design

- Tasks. arrhythmia-multiclass and arrhythmia-binary.
- Methods. IBGWO4 and GWO-CPO from `arrfs/optimizers.py`, unchanged, both with the accuracy fitness (v = 0.99).
- Runs. 30. Run r uses split r of `ShuffleSplit(n_splits=30, test_size=0.2, random_state=42)`, random and unstratified as the paper describes, and one initial population of 10 masks shared by both methods and both readings.
- Budget. 2,000 fitness calls. P + 2PT = 2,010 for T = 100, and `search` needs a multiple of 100, so the last half-iteration is cut for both methods.
- Preprocessing. `FoldPreprocessor` fitted on the 80% rows. Arrhythmia has missing values, and the paper's gene-expression data has none to impute.
- Two readings of where the fitness is scored, both run.
  - Paper-style. 5-NN trained on the 80% rows and scored on the 20% rows, which are also the reported rows (`HoldoutFitness` from `test_reuse_demo.py`).
  - Clean. Inner stratified 3-fold CV on the 80% rows only, as in the main experiment. Both methods share the inner folds.
- Reported score. scikit-learn `KNeighborsClassifier(5)` trained on the 80% rows with the chosen subset and scored on the 20% rows. Accuracy is primary, because it is the paper's ACA. Macro-F1 and balanced accuracy are secondary.
- Statistics. `analyze._paired` and `analyze.holm`, the same functions as the main tables, paired by run.

## Decision rule, fixed now

For each task and reading, "GWO-CPO beats IBGWO4 under the base paper's protocol" holds if the mean paired difference in test accuracy over the 30 runs is above 0 and the two-sided Wilcoxon signed-rank p is below 0.05. Holm-adjusted p over the four (task, reading) cells is reported next to it.

What the slides say, by outcome.
- Holds under the clean reading. "Under the base paper's settings, GWO-CPO scored higher than IBGWO4." Claim 1 stays "comparable" under the main protocol, and the two results are shown side by side.
- Holds only under the paper-style reading. "The gain appears only when the subset is chosen on the reported test rows, so it comes from that shortcut, not from better feature selection."
- Holds under neither. "Even under the base paper's own protocol, GWO-CPO is comparable to IBGWO4, so the main result is not caused by our stricter testing."
- IBGWO4 higher with p below 0.05. Reported as it is.

Also reported, with no claim attached.
- The share of the 30 runs in which GWO-CPO's test accuracy is strictly higher than IBGWO4's. This is what a single split could have shown.
- GWO-CPO with the accuracy fitness against IBGWO4 with the accuracy fitness, paired over the main experiment's 25 folds, read from its saved rows with no new search. This separates the fitness difference from the protocol difference.

The script is `base_protocol_demo.py`, outside `arrfs/`, so `results/full` stays valid. Results go to `results/base_protocol/`.
