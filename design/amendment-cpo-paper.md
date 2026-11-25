# Amendment 2: CPO details from the original paper

Decided on 2026-11-20 after reading Abdel-Basset, Mohamed & Abouhawwash 2024, "Crested Porcupine Optimizer: A new nature-inspired metaheuristic", Knowledge-Based Systems 284:111257 (PDF supplied by the user). It replaces every CPO detail taken from the porcupy port in `grounding.md`, `synthesis.md` and `amendment-khaseeb.md`. It applies to BCPO and to the CPO step inside GWO-CPO.

## Confirmed by the paper (no change)

- The clock counts function evaluations. Eq. (3) defines t as "the current function evaluation" and Tmax as "the maximum number of function evaluations", and Algorithm 1 increments t once per individual. `progress = calls / budget` is the paper's own clock.
- δ is +1 or -1 per dimension (Eq. 8).
- Sight (Eqs. 4-5), odor (Eqs. 7, 9, 10), physical attack (Eq. 11 with the simplified force of Eq. 13), branch choice (Eq. 14) and cyclic population reduction (Eq. 3) match the current code.
- Greedy acceptance: Algorithm 1 keeps the old position only when the new one is worse.

## Changed

- Parameters. α = 0.1, Tf = 0.5, T = 2 cycles, N_min = round(2/3 N). These are the paper's recommended settings: Table 2 lists N' = 120, α = 0.1, N_min = 80, Tf = 0.5, T = 2, with the tuning in Section 3.1, Table 3 and Fig. 13. Geng & Li 2024 used the same α and Tf for binary CPO on Arrhythmia. The porcupy values 0.2 and 0.8 are not the paper's. The paper's engineering settings (Section 4.1: T = 1, N_min = 25 of 40, Tf = 0.9, α = 0.03) were tuned for constrained design problems and are not used.
- Sound defence follows Eq. (6): new = (1 - U1) x_i + U1 (y + τ3 (x_r1 - x_r2)). The port swaps U1 and 1 - U1. With U1 = rand(D) > rand both forms have the same distribution, but the code should match the equation a reader checks.
- The sums in Eqs. (10) and (12) run over the current population, after reduction.

## Left open by the paper (stated choices)

- U1's distribution is not given ("a binary vector ... generated randomly"). We use rand(D) > rand, as porcupy does.
- When τ4 (Eq. 11) is drawn is not stated. We draw it once per generation.
- The paper shows no code, so nothing is claimed about the authors' MATLAB implementation.

## Why these are fixed now

The values come from the original paper, not from any result on our data. Tuning α or Tf on our folds would be tuning on test data.
