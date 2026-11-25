# Amendment 6: an informed start from a top-journal paper, tested once

Decided on 2026-11-21, after claim 1 ("comparable") and Amendment 5 were read, and before any search with the new start had run. The only thing looked at beforehand was the size of the starting subsets on outer-train rows (`scratchpad/fwdi_probe.py`, 8 sampled folds: 14 to 23 features expected per starting mask, no empty masks). No score was seen.

## Why

The students asked for an algorithm or an idea from a top journal that could make the proposed model beat the base paper on the same dataset. Li, Xue & Zhang 2021 ("Improved binary particle swarm optimization for feature selection with new initialization and search space reduction strategies", Applied Soft Computing 106:107302) report on UCI Arrhythmia with a clean protocol. Their Section 4.4 says the data is split 70/30 and the test set "is the unseen data for the FS method". They use 5-NN, and their fitness is 5-fold cross-validation on the training set.

Their Arrhythmia results calibrate against ours. Their sticky binary PSO reaches 62.70% test accuracy with 107.3 features. Our binary PSO reaches 62.9% with 116 features. Their own method, ISBPSO, reaches 71.13% with 14.8 features. Our best searches reach 68% to 69% with 13 to 21 features.

ISBPSO adds three things to binary PSO. One of them does not depend on the optimiser. It is the feature-weighting directed initialisation (FWDI, their Section 3.2).

## The idea, as the paper states it

- The weight w_d of feature d is its mutual information with the class (their Eq. 6).
- p_d = λ / (1 + exp(σ − 2σ (w_d − min w) / (max w − min w))) (their Eq. 14), with λ = 0.5 and σ = 10 (their Section 4.3).
- Each starting mask keeps feature d with probability p_d.

## Design

- The main experiment's protocol in every respect. The 25 outer folds per task of `RepeatedStratifiedKFold(5, 5, random_state=42)`, `FoldPreprocessor` on outer-train rows, the fold's inner 3-fold split, population 20, 2,000 fitness calls, the same search seeds, and the four classifiers at final scoring. Only the starting population X0 changes.
- Tasks. arrhythmia-multiclass and arrhythmia-binary.
- All five algorithms and both fitness kinds get the informed start. The start is one new factor applied to everything, not a privilege of the proposed model.
- Mutual information comes from scikit-learn's `mutual_info_classif` with default settings and a seed from the fold key, on the outer-train rows after preprocessing. This deviates from the paper, which discretises with MDL first. λ and σ are the paper's values and are not tuned here.
- The start is drawn once per fold and shared by all algorithms and both fitness kinds, as the random start is.
- Check before the run. With the random start, the script must reproduce the main run's saved subsets exactly on sampled folds. Then X0 is the only difference.
- The script is `informed_init_demo.py`, outside `arrfs/`. Results go to `results/informed_init/`.

## Claim, fixed now, in the same form as claim 1

"GWO-CPO with the informed start beats the base paper's method as published" holds, per task, only if GWO-CPO with the imbalance fitness and the informed start has higher mean KNN macro-F1 than IBGWO4 with the accuracy fitness and the random start (the saved main-run rows) over the 25 folds, with two-sided Wilcoxon p below 0.05.

If the claim fails, the slides say the informed start did not make the proposed model better than the base paper.

Attribution, reported whatever the claim shows.
- IBGWO4 with the accuracy fitness and the informed start, against the same with the random start. If the base method gains as well, and GWO-CPO does not beat IBGWO4 when both have the informed start, the slides say the gain comes from the start and that it helps the base paper's method too.
- Informed minus random for all ten algorithm-and-fitness rows.
- Number of features, accuracy and balanced accuracy are secondary. The other three classifiers are exploratory.

## Stop rule

This is the only new variant tested on these folds for a "beats the base paper" claim. Whatever it shows, no further variant is tried on these 25 folds for that claim. A later idea needs fresh outer folds from a new outer seed, fixed before it runs.
