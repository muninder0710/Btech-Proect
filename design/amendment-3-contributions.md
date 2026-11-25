# Amendment 3: three contributions, and the test-reuse experiment

Decided on 2026-11-20, before any result of the amended design existed. The students asked for fewer, solid contributions grounded in recent work. The novelty checks behind this are logged in `decisions.tsv`.

## What recent work already covers

- CPO + GWO hybrids exist outside feature selection. AEGWCPO (Gao et al.) uses elite GWO guidance inside CPO for multi-UAV path planning. CPO-IGWO-BRLSTM (He et al.) tunes an LSTM. Both are listed in the 2026 CPO review (Yu et al., Applied Intelligence 56:422). Neither is a binary feature selector.
- Binary CPO is called for, not established. The 2026 CPO review says binary versions barely exist and asks for one. Geng & Li 2024 ran a binary CPO as a competitor on Arrhythmia without describing it.
- Imbalance-aware fitness exists. Gürkan Kuntalp et al. 2024 (Diagnostics 14:2244) used MCC; Cao et al. 2026 (POISE-X, Cancers 18:3353) used macro-F1. Neither compared it head to head with an accuracy fitness under one protocol.
- Feature-selection stability and SHAP on UCI Arrhythmia exist. Banu & Shilpa 2026 (Sci. Rep., doi 10.1038/s41598-026-59984-9) studied Boruta, RFECV and LASSO with SHAP. They used no metaheuristics.

## The three contributions

1. Binary GWO-CPO feature selection. Khaseeb et al. 2025's IBGWO4 framework with its PSO step replaced by a CPO step. It answers the CPO authors' future work (binary variants, feature selection) and the 2026 review's call for a binary CPO. It is tested against IBGWO4, BCPO, BGWO and BPSO on seven datasets and 25 folds each.
2. A controlled test of what the fitness should optimise. The same five optimisers, folds and budget are run with an accuracy fitness and with an imbalance-aware fitness. Recent papers used one or the other, never both under one protocol. Either outcome is a finding.
3. A measured re-evaluation of evaluation shortcuts in recent metaheuristic feature-selection papers. (a) Oversampling and selection before cross-validation: done, `leakage_demo.py`. (b) Choosing the subset on the same split that is reported, as Gürkan Kuntalp et al. 2024 state and Geng & Li 2024's numbers imply: the experiment below.

Demoted to supporting analysis, not claimed as contributions: which ECG leads the selected features come from, subset stability, and SHAP.

## Experiment for contribution 3(b): choosing the subset on the reported split

- Data: arrhythmia-multiclass and arrhythmia-binary.
- Splits: 25 stratified 70/30 splits, `StratifiedShuffleSplit(n_splits=25, test_size=0.3, random_state=42)`. Preprocessing is fitted on the 70% part.
- Search: BCPO and GWO-CPO with the accuracy fitness (the base papers' fitness), population 20, 2,000 evaluations, seeds derived from the split index and algorithm.
- Paper-style arm: the fitness trains 5-NN on the 70% part and scores the 30% part. The reported number is the chosen subset's score on that same 30% part.
- Clean arm: the fitness uses only the 70% part (inner stratified 3-fold CV). The chosen subset is trained on the 70% part and scored once on the 30% part.
- Random-label control: both arms again with labels permuted once per split.
- Reported: accuracy and macro-F1 per arm, mean ± sd, and the paired paper-style minus clean difference with a 95% bootstrap interval over the 25 splits.
- Claim, fixed now: choosing on the reported split inflates the score if the paired difference's interval excludes 0 on true labels. With permuted labels the clean arm must sit within 0.05 of chance; the paper-style arm is expected above it.
- Order: written now, implemented after the current run of the main experiment finishes, so it imports the final `arrfs/optimizers.py`.
