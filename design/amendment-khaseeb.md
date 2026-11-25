# Amendment 1: follow Khaseeb et al. 2025 as the method base paper

Decided on 2026-11-20, before any search algorithm had produced a result (no `results/smoke` existed). It replaces the algorithm list in `grounding.md` and `synthesis.md`. Everything else in those files stands.

## Why

The students chose Khaseeb, Keshk & Youssef 2025 ("Improved Binary Grey Wolf Optimization Approaches for Feature Selection Optimization", Applied Sciences 15:489) as the method base paper. Its best method, IBGWO4-V, runs a GWO step, then a PSO step on the GWO population, then keeps the best P of the previous, GWO and PSO populations. The paper says PSO supplies exploration and GWO exploitation (section 4). The proposed model must therefore be this framework with the PSO step replaced by a CPO step, and nothing else changed. Then "base paper vs ours" differs in exactly one step.

## Shared binarisation (all five algorithms)

Positions are binary masks. Every operator proposes a continuous position Y from the current mask X. Bit d flips with probability V(Y_d - X_d) = |tanh(Y_d - X_d)|, the V-shaped transfer function V2 the paper found best (Mirjalili & Lewis 2013 rule: flip on the step). For PSO, Y - X is the velocity. The paper does not say how the transfer function is applied to GWO's averaged position; this rule is our stated interpretation. One rule for all five means the ablation compares update rules, not binarisations. No bound handling is needed.

## The five algorithms (replace the current registry)

All use population P = 20 and the 2,000-evaluation budget with `progress = calls / budget` schedules, as in `synthesis.md`.

1. **BPSO** (Kennedy & Eberhart 1997, V-shaped). v = w v + c1 r1 (pbest - x) + c2 r2 (gbest - x), c1 = c2 = 2, w from 0.9 to 0.4 by progress, |v| <= 6, flip with |tanh(v)|.
2. **BGWO** (Mirjalili et al. 2014 step, V-shaped). a = 2 - 2 progress; for each leader L: A = 2 a r1 - a, C = 2 r2, X_L' = X_L - A |C X_L - X|; Y = mean of the three; flip rule. Leaders are the three best distinct masks found so far.
3. **BCPO** (Abdel-Basset et al. 2024, V-shaped). The four defences as in `grounding.md` produce Y; flip rule; greedy acceptance per porcupine; cyclic population reduction by progress.
4. **IBGWO4** (Khaseeb et al. 2025, the base paper's best variant). Per iteration: G = one GWO step applied to the current population (evaluate all); Q = one PSO step applied to G, with velocity and pbest kept per population slot and gbest the best found so far (evaluate all); the next population is the best P of current ∪ G ∪ Q. Each iteration spends 2P evaluations.
5. **GWO-CPO** (proposed). IBGWO4 with the PSO step replaced by one CPO step applied to G: each individual takes CPO's exploration or exploitation branch with CPO's own probabilities, using G's fitness values in the odor and physical-attack weights; evaluate all; the next population is the best P of current ∪ G ∪ C. No population reduction and no separate greedy step, because the ranking step already keeps the best.

Removed from the grid: Al-Tashi 2019 binary PSO-GWO (cited as background only) and the earlier "CPO with GWO exploitation" variant. Only one proposed variant exists, so there is nothing to pick from after the fact.

## What the base paper leaves open, which this project fixes

- One random 80/20 split (the paper lists 10-fold CV as future work). We use 5 x 5 repeated stratified CV.
- Accuracy-only fitness and metric on multiclass medical data. We add the imbalance-aware fitness and report macro-F1 and balanced accuracy.
- KNN only (the paper lists SVM and neural networks as future work). We score every subset with four classifiers.
- Population size 10 tuned per dataset on the same datasets the results are reported on, and the paper does not say the test split was held out of that tuning. We fix P = 20 for every algorithm before any run.
- Results shown only as bar charts and ranks, with no significance tests. We report tables, bootstrap intervals and exploratory Wilcoxon tests.
- Tested on gene-expression data only. We apply it to ECG-derived features (UCI Arrhythmia).
- The paper's future work names "using PSO's search mechanism to improve other methods" and comparing against other GWO hybrids. Swapping PSO for a newer explorer is the mirror image of that direction.
