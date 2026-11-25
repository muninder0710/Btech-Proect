# Amendment 4: what the CPO review changes, and the small-test-split experiment

Decided on 2026-11-20, before experiment 3(c) existed and before any result of the amended main design. Sources: Yu et al. 2026, "A systematic review of the crested porcupine optimizer: variants and applications", Applied Intelligence 56:422 (full text, CC BY-NC-ND, supplied by the user); Dhanka & Maini 2025, International Journal of Cardiology 432:133266 (pages viewed as screenshots supplied by the user; quotes and page numbers must be re-checked against the publisher's copy before the paper is submitted).

## Contribution 1: the novelty claim, narrowed

CPO-based feature selection already exists, all outside medicine and none hybridised with GWO.
- PLGCPO, Li & Dong 2024 (ICMCCE, IEEE), intrusion detection (review ref. 75).
- LF-CPO, Valavan & Joseph 2025 (ICDCECE, IEEE), intrusion detection (ref. 78).
- CPO-FS, Yildizdan & Baş 2025 (Alexandria Eng. J. 129:1061), speech-based gender detection, sixteen transfer functions, KNN and decision trees (ref. 168).

CPO + GWO hybrids exist outside feature selection.
- AEGWCPO, Gao et al. 2025 (Meas. Sci. Technol. 36:116301), UAV swarm path planning; an elite-GWO step runs only when the best fitness stagnates (ref. 60).
- CPO-IGWO, He et al. 2026 (Measurement 257:118711), BiLSTM tuning (ref. 186).
- A PSO + CPO hybrid also exists: Chen et al. 2025 (Sustainability 17:8057), water-injection parameters (ref. 51).

Claim, as it will appear: the first binary GWO + CPO hybrid for feature selection. Geng & Li 2024 already ran binary CPO on UCI Arrhythmia as a baseline, so no claim is made about CPO on this dataset.

## What the review supports, with page numbers

- Binary CPO is called for (section 3.5, p. 26): "Discrete and binary versions barely exist" and "Researchers should develop a binary CPO for 0–1 problems."
- The hybrid's rationale (section 2.6, p. 11): CPO "converges quickly but attains slightly lower final precision than some competitors (e.g., GWO, PSO), indicating room for improvement in local exploitation", and "adaptive or hybrid enhancements could further improve its effectiveness". Also p. 3: "The later stages do not exploit solutions well enough."
- Dropping cyclic population reduction has precedent: three variants remove it (Table 5; refs. 46, 54, 82).
- The equal evaluation budget answers section 4.5 (p. 38): "the budget is seldom equalised, since the population size and iteration count are fixed but the total number of function evaluations ... is rarely controlled", and "Compute budget equivalence across optimizers is not verified."
- Contribution 3 answers section 4.5 (p. 38): "Nested cross-validation is rarely reported. Leakage control is often missing." and future direction 6 (p. 42): "nested cross-validation, leakage-proof pipelines".
- Comparing against the original CPO answers section 3.7 (p. 27): comparisons are "rarely with the original CPO or existing variants".
- The review's settings (p. 12: N_min = 80 of 120, T = 2, alpha = 0.1, Tf = 0.5) match Amendment 2.

Limitation to state: the review's future direction 1 also asks for comparisons with leading variants (CAPCPO, ECPO, BTCPO). This study compares with the original CPO only.

Discrepancy noted, no change: the review describes the branch-choice numbers tau6 to tau10 as normally distributed (p. 8). The code follows the original paper's uniform random numbers.

## Experiment 3(c): one small test split

Dhanka & Maini 2025 report 95.24% accuracy (PSO-tuned XGBoost) for normal vs arrhythmia. Their confusion matrices all sum to 42 test patients (TP 26, TN 14, FP 1, FN 1 for the best model; Fig. 9). Section 5.1 describes, in this order: dropping attribute 14 and the records missing other attributes (452 to 420 records); dropping 89 attributes with many zero values (to 189 plus the label); keeping the 100 attributes most correlated with the label by Pearson correlation; min-max scaling; then "the dataset was split into 90:10 Train-Test Ratio" with "zero random states". The record counts (420; 237 normal, 183 arrhythmia) were reproduced exactly from the raw file before this experiment was written.

Pipeline (replicating the description):
1. Drop attribute 14 and every record missing attribute 11, 12, 13 or 15 (420 records).
2. Drop the 89 attributes with the most zero values (keep 189 of 278).
3. Keep the 100 attributes with the largest absolute Pearson correlation with the label.
4. Min-max scale.
5. Classifiers with scikit-learn defaults: logistic regression, LDA, Gaussian naive Bayes, decision tree, and HistGradientBoostingClassifier as the gradient-boosting stand-in (XGBoost is not installed). No PSO tuning: the search ranges are not in the pages we have.

Arms:
- Paper-style: steps 2 to 4 fitted on all 420 records, then split.
- Clean: steps 2 to 4 fitted on the 378 training records of each split only.

Splits: 200 stratified 90:10 splits, `StratifiedShuffleSplit(n_splits=200, test_size=0.1, random_state=42)`, 42 test patients each, the same splits for both arms. The paper's own split read as `train_test_split(test_size=0.1, random_state=0)` is reported for interest and used for no claim. Reference: 5 x 5 repeated stratified CV accuracy of the clean arm on all 420 records.

Reported, per arm and classifier: accuracy on each 42-patient test set (min, 5th percentile, median, 95th percentile, max) and the share of splits at or above 95.24%. Also the Wilson 95% interval for 40 of 42 correct, 84.2% to 98.7%.

Claims fixed now:
- "A 42-patient test set is too small to support a 95% claim" holds if, for the classifier with the best median, the clean arm's 5th-to-95th-percentile range is wider than 10 accuracy points.
- "Selecting features on all records before the split inflated accuracy" holds if the paired paper-style minus clean difference over the 200 splits has a 95% bootstrap interval that excludes 0. If not, the slides say selection before the split did not measurably raise accuracy in our replication.
- No claim is made that the paper's number is wrong. The slide reports what the described pipeline gives on this data.

Order: implemented after the running delegate finishes, in a script outside `arrfs/`.
