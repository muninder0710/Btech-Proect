# Amendment 12: the base paper's own nine datasets

Decided on 2026-11-23, before any optimiser had run on any of the nine datasets. The students asked for the method to be run on the datasets that the base paper itself used.

## What this is and is not

- It is a replication on the base paper's own ground: its nine datasets and its protocol.
- It tests no new variant. The base method and the proposed model are exactly those of Amendment 6, with every setting unchanged. The stop rule of Amendment 8 forbids trying further variants for a "beats the base paper" claim, and none is tried here.
- The analysis below is fixed before the run, and the outcome is reported whichever way it falls.
- If the proposed model is not higher than the base method here, the write-up must say that the arrhythmia result did not carry over to the base paper's gene-expression data. That sentence is not optional.

## Data

The nine datasets of Khaseeb et al. 2025, Table 1: SRBCT, Brain_Tumor1, Brain_Tumor2, Leukemia1, Leukemia2, DLBCL, Prostate_Tumor, 11_Tumors and 14_Tumors. They are the GEMS collection of Statnikov et al. 2005.

- Source: the authors' own files, from the Internet Archive's copies of gems-system.org, which is offline. `fetch_gems.py` downloads them and checks each against a recorded checksum. `data/gems/SOURCES.txt` lists every URL.
- Each file has exactly the instances, genes and classes of Khaseeb et al.'s Table 1. Three of the nine (11_Tumors, DLBCL, SRBCT) were also compared with the copies on OpenML and are identical.
- The values are used as distributed. Each gene is min-max scaled on the training rows only, and a gene that is constant on the training rows is dropped. No other filtering is done.

## Protocol

The base paper's, exactly as Amendment 5 applied it to Arrhythmia (`base_protocol_demo.py`): a random 80/20 split, 30 runs, a population of 10, 2,000 fitness calls, 5-NN, and one starting population shared by the arms that use the same start.

The paper does not say which rows the fitness is scored on, so both readings of Amendment 5 run.

- **Clean.** The fitness is scored inside the 80% rows, by 3-fold cross-validation. The 20% rows are used once, to report. This is the primary reading.
- **Paper-style.** The fitness is scored on the 20% rows that are then reported.

Four arms run on every split and in both readings.

1. **Base as published.** IBGWO4, accuracy fitness, random start.
2. **Hybrid alone.** GWO-CPO, accuracy fitness, random start.
3. **Base with smart start.** IBGWO4, accuracy fitness, the informed start of Amendment 6.
4. **Proposed.** GWO-CPO, imbalance-aware fitness, the informed start. This is the model of Amendment 6.

The informed start is computed from the 80% rows only, in both readings.

## Reported

Accuracy (the base paper's ACA), macro-F1 and the number of genes kept, as the mean over the 30 runs for every dataset, reading and arm.

Three comparisons, each an arm minus the base as published:

- **Hybrid alone** (arm 2), on accuracy. This is the question of Amendment 5.
- **Start alone** (arm 3), on accuracy.
- **Proposed model** (arm 4), on macro-F1. This is the claim of Amendment 6.

For each comparison and reading:

- Per dataset: the mean difference over the 30 runs with a bootstrap interval. The splits of one dataset share patients, so these p-values are exploratory.
- Across the nine datasets: a Wilcoxon signed-rank test on the nine per-dataset mean differences, two-sided at 5%. The datasets are independent, so this is the test that counts. The verdict is "higher", "lower" or "no clear difference".

The claim of Amendment 6 **replicates on the base paper's datasets** if, in the clean reading, the proposed model is "higher" than the base as published on macro-F1 across the nine datasets. Otherwise it does not replicate.

Also reported, with no pass or fail: how close the base method comes to the base paper's own printed result. The paper prints an average accuracy of 0.95 over the nine datasets for IBGWO4, with five datasets at 1.00. Our base arm is reported next to that in both readings.

## Checks before the run

- Every dataset loads with the size of Table 1, without missing values, and with its recorded checksum.
- The job of `base_datasets_demo.py`, run on Arrhythmia, reproduces the saved rows of Amendment 5 exactly for the two arms that both scripts run.
- The paper-style form of the imbalance-aware fitness equals the same score computed a second way with scikit-learn.

## Practicalities

- The run saves each finished dataset and run, and can be stopped and started again. Jobs are ordered run by run, so a stopped run holds the same number of runs for every dataset.
- A summary made from an unfinished run is labelled partial. Only the complete run goes into the write-up.

The script is `base_datasets_demo.py`, its checks are in `check_base_datasets.py`, and results go to `results/base_datasets/`.
