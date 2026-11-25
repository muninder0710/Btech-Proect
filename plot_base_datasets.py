"""Graphs and a table of the run on the base paper's nine datasets (Amendment 12).

Reads results/base_datasets/ (base_datasets_demo.py, complete run only) and writes there:

    base_datasets_table.csv         per dataset: the base method in both readings, and all four arms in the clean one
    base_datasets_two_tests.png     the base method as published, paper-style test against clean test
    base_datasets_clean.png         in the clean test: the smart start added to the base method, and the proposed model

Run from code/:  .venv/bin/python plot_base_datasets.py
"""
import sys

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from base_datasets_demo import ARMS, BASE, COMPARISONS, DATASETS, OUT
from base_protocol_demo import N_RUNS
from plot_optimizers import AXIS, INK, INK_2, MUTED, SURFACE, START_COLOR, _heading, _quiet

PRINTED_MEAN = 0.95  # Khaseeb et al. 2025, p. 21: IBGWO4-V "achieved an average ACA value of 0.95 across all datasets"
CLEAN_COLOR = "#2a78d6"
MEAN_ROW = "Mean of the nine"
ROWS = ["11_Tumors", "14_Tumors", "Brain_Tumor1", "Brain_Tumor2", "Leukemia1", "Leukemia2", "DLBCL", "Prostate_Tumor",
        "SRBCT"]  # the order of the base paper's Table 1
assert sorted(ROWS) == sorted(DATASETS)


def _label(name: str) -> str:
    if name == MEAN_ROW:
        return name
    patients, genes, _ = DATASETS[name]
    return f"{name}  ({patients} patients, {genes:,} genes)"


def _with_mean(table: pd.DataFrame) -> pd.DataFrame:
    table = table.loc[ROWS]
    table.loc[MEAN_ROW] = table.mean()
    return table


def _panel(ax, table: pd.DataFrame, before: str, after: str, after_color: str) -> None:
    """One dumbbell per dataset on `ax`: a ring at column `before` and a dot at column `after`, with the change."""
    for y, name in enumerate(table.index):
        y += name == MEAN_ROW  # a gap above the mean row
        a, b = table.at[name, before], table.at[name, after]
        ax.plot([a, b], [y, y], color=AXIS, linewidth=2, solid_capstyle="round", zorder=2)
        ax.plot(a, y, "o", ms=8, mfc=SURFACE, mec=MUTED, mew=1.8, zorder=3, clip_on=False)
        ax.plot(b, y, "o", ms=9.5, mfc=after_color, mec=SURFACE, mew=1.5, zorder=4, clip_on=False)
        ax.text(1.035, y, f"{b - a:+.3f}".replace("-", "−"), transform=ax.get_yaxis_transform(), va="center",
                color=INK_2, fontsize=9)
    ax.text(1.035, -1, "change", transform=ax.get_yaxis_transform(), va="center", color=MUTED, fontsize=9)
    ax.set_ylim(len(table) + 0.6, -1.6)
    _quiet(ax, "x")
    ax.tick_params(axis="y", pad=8)


def _name_rows(ax, table: pd.DataFrame) -> None:
    ax.set_yticks([y + (name == MEAN_ROW) for y, name in enumerate(table.index)], [_label(name) for name in table.index],
                  color=INK)
    ax.get_yticklabels()[-1].set_fontweight("bold")


def _legend(fig, labels: list[str], after_color: str) -> None:
    height = fig.get_figheight()
    fig.legend(handles=[Line2D([], [], linestyle="", marker="o", ms=8, mfc=SURFACE, mec=MUTED, mew=1.8),
                        Line2D([], [], linestyle="", marker="o", ms=9.5, mfc=after_color, mec=SURFACE, mew=1.5)],
               labels=labels, loc="upper left", bbox_to_anchor=(0.014, 1 - 0.8 / height), ncols=2, handlelength=1.2,
               handletextpad=0.5, columnspacing=1.6, borderaxespad=0)


def two_tests(accuracy: pd.DataFrame, path) -> None:
    """The base method as published: accuracy when the search is scored on the reported rows, and when it is not."""
    table = _with_mean(accuracy)
    head, foot = 1.45, 0.65
    height = head + (len(table) + 2) * 0.33 + foot
    fig = plt.figure(figsize=(11.5, height))
    ax = fig.add_gridspec(1, 1, left=0.3, right=0.93, top=1 - head / height, bottom=foot / height).subplots()
    _panel(ax, table, "paper", "clean", CLEAN_COLOR)
    _name_rows(ax, table)
    mean_y = len(table)
    ax.plot([PRINTED_MEAN, PRINTED_MEAN], [mean_y - 0.32, mean_y + 0.32], color=INK, linewidth=1.5, zorder=5)
    ax.annotate(f"printed in the paper: {PRINTED_MEAN:.2f}", (PRINTED_MEAN, mean_y - 0.32), xytext=(0, 5),
                textcoords="offset points", ha="center", va="bottom", fontsize=8.5, color=INK_2)
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("accuracy on the 20% of patients that are reported")
    _heading(fig, "The base paper's method on its own nine datasets, tested two ways",
             f"GWO + PSO as published, mean of {N_RUNS} random 80/20 splits per dataset.")
    _legend(fig, ["paper-style test: the search is scored on the patients that are reported",
                  "clean test: the search never sees those patients"], CLEAN_COLOR)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def clean_comparisons(means: pd.DataFrame, across: pd.DataFrame, path) -> None:
    """Clean test only: the smart start added to the base method (accuracy), and the proposed model (macro-F1)."""
    panels = [("start alone", "The base method with the smart start added", "accuracy"),
              ("proposed model", "Our model: GWO + CPO with the smart start", "macro-F1")]
    head, foot = 2.0, 0.65
    rows = len(ROWS) + 1
    height = head + (rows + 2) * 0.33 + foot
    fig = plt.figure(figsize=(12.5, height))
    axes = fig.add_gridspec(1, 2, left=0.27, right=0.94, top=1 - head / height, bottom=foot / height,
                            wspace=0.22).subplots(sharey=True)
    for ax, (comparison, title, metric_name) in zip(axes, panels):
        arm, metric = COMPARISONS[comparison]
        table = _with_mean(means[metric][[BASE, arm]])
        _panel(ax, table, BASE, arm, START_COLOR["informed"])
        verdict = across.loc[(comparison, metric)]
        ax.annotate(title, (0, 1), xycoords="axes fraction", xytext=(0, 40), textcoords="offset points",
                    fontsize=10.5, fontweight="bold")
        ax.annotate(f"Higher on {int(verdict.higher_in)} of {int(verdict.datasets)} datasets.\n"
                    f"Across the datasets: {verdict.verdict} (p = {verdict.p:.3f}).", (0, 1), xycoords="axes fraction",
                    xytext=(0, 8), textcoords="offset points", fontsize=9, color=INK_2, linespacing=1.5)
        ax.set_xlim(0.3, 1.0)
        ax.set_xlabel(f"{metric_name} on the reported patients (higher is better)")
    _name_rows(axes[0], _with_mean(means["acc"][[BASE]]))
    _heading(fig, "Clean test on the base paper's nine datasets: what the smart start and our model add",
             f"Mean of {N_RUNS} random 80/20 splits per dataset. The search never sees the reported patients.")
    _legend(fig, ["the base method as published (GWO + PSO, random start)", "the method named above each panel"],
            START_COLOR["informed"])
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> int:
    runs = pd.read_csv(OUT / "base_datasets_runs.csv", float_precision="round_trip")
    assert runs.groupby(["dataset", "reading", "arm"]).size().eq(N_RUNS).all(), "the run is not complete"
    across = pd.read_csv(OUT / "base_datasets_across.csv")
    across = across[across.reading == "clean"].set_index(["comparison", "metric"])

    base = runs[runs.arm == BASE].pivot_table(index="dataset", columns="reading", values="acc")
    clean = runs[runs.reading == "clean"].pivot_table(index="dataset", columns="arm", values=["acc", "macro_f1", "n_features"])
    table = pd.DataFrame({"accuracy / paper-style test / base as published": base["paper"]})
    for metric, name in [("acc", "accuracy"), ("macro_f1", "macro-F1"), ("n_features", "genes kept")]:
        for arm in ARMS:
            table[f"{name} / clean test / {arm}"] = clean[metric][arm]
    _with_mean(table).rename_axis("dataset").to_csv(OUT / "base_datasets_table.csv", float_format="%.3f")

    two_tests(base, OUT / "base_datasets_two_tests.png")
    clean_comparisons(clean, across, OUT / "base_datasets_clean.png")
    print(f"table and graphs written to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
