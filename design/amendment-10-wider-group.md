# Amendment 10: a wider group of optimisers as comparison rows

Decided on 2026-11-22, before any of the eight new optimisers had run on the data. The students asked why the comparison table stops at nine optimisers when their research notes list a much larger group. This adds rows to the fair comparison. It tests no claim.

## What this is and is not

- It is a wider comparison: seventeen optimisers in place of nine, on the same folds, with both starts.
- It is not a search for a winner. The stop rule of Amendment 8 stands. None of the eight can become the proposed model, and no "beats the base paper" claim is made from them, whatever they score.
- With seventeen rows it is likely that some row lands above the proposed model by chance. That changes nothing. The table is descriptive, and the write-up must not rank optimisers on differences that are inside the fold-to-fold noise.
- Every row is reported, including any that do badly.

## The rule that chose the eight

Fixed before any of them ran. An optimiser is included when all three hold.

1. An arrhythmia paper in the students' research notes used it.
2. Its published update rule acts on positions, so it maps onto the shared flip rule without redesign.
3. Its authors' reference code or its original paper could be read, and the implementation was checked against it. GA is the exception: it has no single reference, so its settings come from the arrhythmia paper that used it and its operators are the textbook ones.

The eight, each with the arrhythmia paper that used it and the source it was checked against:

- **GA**, genetic algorithm. Used by Atik & Dalyan 2023 on UCI Arrhythmia. Checked against their Table 2.
- **BDE**, differential evolution. Used by Carrillo-Alarcón et al. 2020 on MIT-BIH arrhythmia beats. Checked against Storn & Price 1997 (Eqs. 2 and 4, Section 4).
- **BBA**, bat algorithm. Used by Geng & Li 2024 on UCI Arrhythmia. Checked against the authors' `BBA.m` (MathWorks File Exchange 44707) and Yang 2010 (arXiv:1004.4170).
- **BDA**, dragonfly algorithm. Used by Sharma & Singh 2020 on UCI Arrhythmia. Checked against the author's `BDA.m`.
- **BSCA**, sine cosine algorithm. Used by Kamal et al. 2026 on UCI Arrhythmia. Checked against the author's `SCA.m`.
- **BSSA**, salp swarm algorithm. Used by Geng & Li 2024. Checked against the authors' `SSA.m`.
- **BHHO**, Harris hawks optimisation. Used by Geng & Li 2024. Checked against the authors' `HHO.m` and the paper (Heidari et al. 2019, the first author's copy).
- **BMPA**, marine predators algorithm. Used by Kamal et al. 2026. Checked against the authors' `MPA.m` and the accepted manuscript (Faramarzi et al. 2020).

## Left out, and why

- **Artificial bee colony** (Geng & Li 2024). Its move changes one coordinate at a time. There is no agreed binary form, so a binary version would be mostly our invention.
- **Slime mould** (Kamal et al. 2026) and **butterfly optimisation** (Sharma & Singh 2020). Their step depends on the size of the fitness value. With a fitness between 0 and 1 the step collapses, so the row would measure the scale of our fitness and not the algorithm.
- **Ant lion** (Sharma & Singh 2020). It builds a new position from random walks around two other solutions. It has no current position for the flip rule to start from, and choosing one would be our design.
- **Kepler, nutcracker, crayfish and walrus optimisers** (Geng & Li 2024), **secretary bird and the other 2022 to 2025 optimisers** (Kamal et al. 2026), **satin bowerbird**, **imperialist competitive** and **ant colony with simulated annealing**. Each needs its paper read in full and was not attempted in this round. They can be added later under the same rule.

## Choices that are ours

All eight read progress = calls / budget where their paper writes t / T, and all start from the fold's shared 20 masks. All but GA use the shared flip in place of their own transfer function. Where an authors' code and paper disagree, the implementation follows the code, because the code produced the paper's results.

- **GA.** It works on the masks themselves and has no continuous position, so it does not use the shared flip. From Atik & Dalyan's Table 2 it takes the population of 20, the crossover probability of 1, the two elites and the rank-based roulette-wheel selection. Their crossover uses a fixed even and odd pattern, and their mutation is not described well enough to copy. This uses uniform crossover and bit-flip mutation with probability 1/D per bit, the textbook forms. The two parents of a pair are different members.
- **BDE.** DE/rand/1/bin with F = 0.5 and CR = 0.9, the values of Carrillo-Alarcón et al. Storn & Price call F = 0.5 a good initial choice. For CR they call 0.1 a good first choice and add that trying 0.9 or 1.0 first shows whether a quick solution is possible. The trial replaces the target when it is not worse, as in the code Storn & Price print in their Figure 3. Their text says "smaller", so their text and their code differ on ties, and this follows the code (see the addendum).
- **BBA.** It follows `BBA.m`: frequency between 0 and 2, pulse rate 0.1, and a bit takes the best mask's value with probability 0.9. `BBA.m` never rejects a move, because its loudness test changes a stored fitness value and nothing else. Loudness therefore has no effect on the search and is left out. The BBA paper itself is closed access and was not read. If it describes a rejection step, paper and code differ and this follows the code.
- **BDA.** It follows `BDA.m`: every other dragonfly is a neighbour, separation has the sign of the code, and step vectors start as random 0s and 1s. The paper is closed access and was not read.
- **BSCA.** It follows `SCA.m`, with a = 2.
- **BSSA.** It follows `SSA.m`: the first half of the chain are leaders, and a leader's move takes either sign with probability 1/2. The paper prints one leader and a condition on c3 that can never be false.
- **BHHO.** It follows `HHO.m`: the Lévy numbers carry no factor 0.01, and the second dive candidate is tried only when the first fails, each compared with the hawk's own mask. A hawk's new mask is evaluated right after its move. `HHO.m` evaluates it at the top of the next generation and evaluates the hawk's own position again inside the dives. Here a hawk's own mask is never evaluated again, so the budget goes to proposals.
- **BMPA.** It follows `MPA.m`: the vector U has one fifth of ones (the paper's text says four fifths), the FADs effect is drawn once per generation for the whole population (the paper writes it per prey), and the Lévy numbers are scaled by 0.05 (the paper's Eq. 5). A generation costs two evaluations per prey.

## Checks before the run

- Each of the eight passes the toy checks of `check_optimizers.py` (budget, curve, best mask, determinism, better than random search and than the initial population).
- With the shared flip broken on purpose, each must fail those checks or end worse than with the real rule on every seed. The second case exists for BBA, whose copying of bits from the best mask still makes progress without any flip. GA never calls the flip and is left out of this part.
- A reference, not a pass or fail test: Geng & Li 2024 print the mean number of attributes kept on Arrhythmia as 134 for BBA, 127 for BPSO, 25 for BSSA, 22 for BHHO, 19 for BWOA and 31 for BCPO. Their protocol differs, so these are not targets. A large departure would be looked into before it is reported.

## Design

- As in Amendment 9: both Arrhythmia tasks, the 25 outer folds of `RepeatedStratifiedKFold(5, 5, random_state=42)`, population 20, 2,000 fitness calls, both fitness kinds, both starts, 5-NN inside the search, four classifiers at the end.
- The same script reruns the four optimisers of Amendment 9. Their rows must come out identical.

## Reported

One table and a set of graphs with all seventeen optimisers: macro-F1, accuracy and number of attributes by optimiser and start, and convergence curves. One descriptive line: the spread between the best and the worst optimiser under each start. No test, no claim.

The graphs change form to fit seventeen rows. Single optimisers are listed by the year of their original paper, and convergence is drawn as one small panel per optimiser.

The optimisers are in `extra_optimizers.py`, the run is `more_optimizers_demo.py`, and results and graphs go to `results/more_optimizers/`.


## Addendum, 2026-11-22: corrections from an independent review

An independent reviewer fetched the eight sources again and compared them with `extra_optimizers.py` line by line. It found no undeclared difference in BBA, BDA, BSCA, BSSA, BHHO and BMPA, and five points, corrected here.

When the review arrived, the first full run was in progress. It was stopped before it wrote any result file, and none of its results were read. One smoke test on a single fold had been run before, and its numbers for every optimiser had been seen.

- **BDE, ties.** The first version kept the target on a tie, following the text of Storn & Price. Their Figure 3 prints `if (score<=cost[i])`, which accepts ties. By the rule above, the implementation now follows the code and accepts ties. This is the only change that affects a search, and it was made to match the source, not in response to a result.
- **BDE, parameters.** The sentence on what Storn & Price recommend was incomplete and is now stated in full above.
- **GA.** "Checked against its authors' reference code or paper" was not true of GA and is now stated as the exception it is.
- **BHHO.** "No mask is evaluated twice" was too strong. A flip that changes no bit, or a mask met before, is a repeated call, as in the five base optimisers. What holds is that a hawk's own mask is never evaluated again.
- A docstring in `extra_optimizers.py` still said the module held four algorithms.

The reviewer did not read the BBA, BDA, SSA, HHO and MPA papers. What this amendment says about those papers comes from the first source check, and what it says about the authors' code was confirmed by both.
