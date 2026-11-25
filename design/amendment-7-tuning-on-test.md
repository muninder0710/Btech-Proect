# Amendment 7: what tuning on the test patients does to the base paper's pipeline

Decided on 2026-11-21, before this experiment produced any result. Source: Dhanka & Maini 2025, International Journal of Cardiology 432:133266, full text supplied by the students and checked in `checks/dhanka_fulltext_checks.py`.

## Why

The students asked how the paper reached 95.24% and why our runs cannot. The paper tunes each classifier with PSO. Its Table 6 says the fitness is computed "on a validation set". Its Section 4.2 says the models were "validated ... by the testing dataset" inside the loop that runs "until a stopping criterion" is met. The paper gives no PSO settings and no search ranges. So the text leaves open whether the tuning saw the 42 test patients. This experiment measures what happens in each case. It is experiment 3(d) of contribution 3.

No claim is made about what the authors did. The slides say what the described pipeline gives when the tuning can see the test patients and when it cannot.

## Design

- Data. The paper's 420 patients, two classes (`small_split_demo.load_420`).
- Pipeline. The paper's filters as rebuilt in `small_split_demo.Steps` (189 attributes with the fewest zeros, then the 100 with the largest absolute Pearson correlation, then min-max scaling), followed by `xgboost.XGBClassifier` (version 2.1.4, one thread, `random_state=0`). Our rebuilt filters share 86 of the 100 attributes the paper lists.
- Splits. 40 random splits. Each split puts 42 patients aside as new patients that no step ever sees, 42 patients in the test set, and 336 in the training set. The paper trains on 378, because it has no third group. Splits are unstratified random permutations, as the paper's split is.
- Tuning. One continuous PSO for every arm: 20 particles, 20 iterations (400 evaluations), inertia from 0.9 to 0.4, c1 = c2 = 2, velocity clamp 0.25 of each range, positions clipped to the range, the same random stream in both tuned arms. The eight tuned settings and their ranges are ours, since the paper gives none: n_estimators 50 to 400, max_depth 2 to 8, learning_rate 0.01 to 0.3 (log scale), subsample 0.5 to 1, colsample_bytree 0.3 to 1, min_child_weight 1 to 10, gamma 0 to 5, reg_lambda 0 to 10.
- Three arms, on the same splits.
  - Shortcut. The filters are fitted on the 378 training and test patients together, as the paper does. The PSO fitness is the accuracy on the 42 test patients. The printed number is the best fitness.
  - Fair. The filters are fitted on the 336 training patients. The PSO fitness is the mean accuracy of an inner stratified 3-fold CV on the training patients. The chosen model is scored once.
  - Default. Filters fitted on the training patients, XGBoost with default settings.
- Every arm reports two accuracies per split: on the 42 test patients and on the 42 new patients.

## Claims, fixed now

- "Tuning on the test patients inflates the printed accuracy" holds if, in the shortcut arm, the mean of (accuracy on the test patients minus accuracy on the new patients) over the 40 splits has a 95% bootstrap interval above 0.
- The same difference is reported for the fair arm. If its interval includes 0, the slides say fair tuning shows no gap.
- Reported with no claim attached: the share of splits in which the shortcut arm prints at least 40 of 42 correct (95.24%), the best printed number over the 40 splits, and the fair accuracy of PSO-tuned XGBoost (the mean over both unseen groups), which becomes the base paper's fair score for later comparisons.

## Notes

- xgboost on macOS needs the OpenMP runtime. Homebrew's libomp is not installed, so the script reuses the copy that scikit-learn ships. This changes nothing in the numbers.
- The script is `tuning_on_test_demo.py`, outside `arrfs/`. Results go to `results/tuning_on_test/`.
