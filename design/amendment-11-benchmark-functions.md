# Amendment 11: the five optimisers on standard benchmark functions, as a side check

Decided on 2026-11-23, before any optimiser had run on any function. The students asked for the hybrid to be run on the mathematical benchmark functions that the original optimiser papers use. This is an appendix. It tests nothing about arrhythmia.

## What this is and is not

- It answers one question: on standard continuous test functions, how does GWO + CPO compare with its two parts and with GWO + PSO?
- It is the same five optimisers on a different test bed. It is not a new variant, and its outcome cannot support a "beats the base paper" claim in the arrhythmia paper. The stop rule of Amendment 8 stands.
- The claims of the arrhythmia paper do not change, whichever way this falls. The result is reported in full, including every function where the hybrid loses.
- The smart start of Amendment 6 has no meaning here, because a function has no patients and no labels. Every run starts from a uniform random population.

## The optimisers

The five of the main experiment: PSO, GWO, CPO, GWO + PSO and GWO + CPO.

- They run the update code of `arrfs/optimizers.py` unchanged. That code proposes a continuous position and then flips bits. Here the flip is replaced by the ordinary position update, new position = old position + step, kept inside the bounds by clipping. Nothing else in the update rules differs from the main experiment.
- PSO's velocity limit is half the width of the function's range (Vmax = Xmax, Shi & Eberhart 1999), in place of the limit of 6 that belongs to binary PSO. Every other setting is the main experiment's: inertia from 0.9 to 0.4, c1 = c2 = 2, and CPO's alpha 0.1, Tf 0.5, two cycles and N_min = 2/3 of N.
- GWO + PSO is the two-step frame of Khaseeb et al. 2025 applied to real vectors. Their paper defines it for binary feature selection only, so this continuous form is our analogue. GWO + CPO is the same frame with a CPO step.

## The functions

F1 to F13 in 30 dimensions, the scalable unimodal and multimodal functions of the GWO and WOA papers (Yao, Liu & Lin 1999). Formulas and bounds are taken from `Get_Functions_details.m`, the file distributed with the authors' GWO-family code, and the dimension from the GWO paper's Tables 1 and 2. That file's F6 is a sphere moved to -0.5 and not the step function of Yao et al. It is followed as it stands, because it is what those papers ran.

Each function is run in two forms.

1. **As published.** Twelve of the thirteen have their optimum at or next to the origin.
2. **Shifted.** The same twelve are moved by a fixed random offset of up to 80% of the half-range in every dimension, so the optimum is no longer at the origin. F8, whose optimum already lies near a corner, is unchanged.

The shifted form is there because GWO-type updates are known to be drawn towards the origin, which flatters them on functions whose optimum sits there (Niu, Niu & Chang 2019; Kudela 2022; Camacho-Villalón, Dorigo & Stützle 2023). If the two forms disagree, the shifted form is the fairer reading, and the write-up says so.

Not run: the ten fixed-dimension functions F14 to F23, the six composite functions of the GWO paper, and the CEC 2014, 2017 and 2020 suites of the CPO paper, which need the official data files.

## Design

- Population 30 and 15,000 fitness calls per run, which is the 30 agents and 500 iterations of the GWO and WOA papers. Every optimiser gets the same number of calls, so a hybrid, which evaluates twice per member, runs half as many generations.
- 30 independent runs per function and form. Within a run the five optimisers share one uniform random start. Seeds come from sha256 of (function, form, run, optimiser).
- F7 adds uniform noise to every evaluation, as published. Each optimiser has its own seeded noise stream.

## Reported

- Per function and form: mean, standard deviation and median of the best value found, per optimiser.
- Ranks: within each run the five are ranked (1 is best, ties share a rank), and the ranks are averaged over runs and then over the thirteen functions.
- GWO + CPO against each of the other four, per function: a paired Wilcoxon signed-rank test over the 30 runs, two-sided, with Holm's correction over the thirteen functions. The verdict per function is "better", "worse" or "no clear difference" at 5%. Runs are independent, so these p-values are not exploratory.
- The summary is a count per competitor and form: better on so many of thirteen, worse on so many, no clear difference on the rest. No single overall winner is declared.

## Checks before the run

- Every function returns its published optimum value at its published optimum point, and hand-computed values at a second point.
- Every shifted function returns the same optimum value at the moved point.
- Each optimiser in continuous form spends exactly its budget, never evaluates a point outside the bounds, is deterministic, improves on its start, and beats random search with the same budget.

The script is `benchmark_functions_demo.py`, its checks are in `check_benchmark_functions.py`, and results go to `results/benchmark_functions/`.

## Addendum, 2026-11-23: one sensitivity run

Written after the registered run and before the run it describes.

In the registered run PSO alone was weak: a median of 2.8 on the sphere as published and 12,800 on the shifted sphere. PSO's update does not depend on where the optimum is, so the drop comes from the velocity limit and the walls. A limit of half the range lets particles reach a wall in one step, and a clipped particle keeps its velocity. Half the range is the loose end of what the literature uses.

GWO + PSO contains the same PSO step. To check that the comparison of GWO + CPO with GWO + PSO does not hinge on this one setting, the whole run is repeated once with a velocity limit of 10% of the range, the tight end of the usual 10 to 20%. Nothing else changes.

- The registered run stays the main result.
- The sensitivity run is reported next to it, with its ranks and its counts, whichever way it falls. Its files go to `results/benchmark_functions/pso_vmax_0.1/`.
- No further setting is tried.
