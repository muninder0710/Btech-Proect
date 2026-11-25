"""Graphs and a table of the benchmark-function side check of Amendment 11. Not an arrhythmia result.

Reads results/benchmark_functions/ (benchmark_functions_demo.py) and its sensitivity run in pso_vmax_0.1/, and
writes to results/benchmark_functions/:

    benchmark_medians.csv   median best value per function and form, one column per optimiser (also in pso_vmax_0.1/)
    benchmark_ranks.png     mean rank of the five optimisers, as published against shifted, for both PSO limits
    benchmark_sphere.png    the sphere function as the example: best value found, as published against shifted

Run from code/:  .venv/bin/python plot_benchmark_functions.py
"""
import sys

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from benchmark_functions_demo import BUDGET, DIM, FORMS, FUNCTIONS, OPTIMISERS, OUT, PROPOSED, PSO_VMAX, RUNS
from plot_optimizers import AXIS, INK, INK_2, MUTED, SURFACE, _heading, _quiet

SENSITIVITY = OUT / "pso_vmax_0.1"
FORM_COLOR = {"published": MUTED, "shifted": "#eb6834"}
FORM_LABEL = {"published": "as published, optimum at the origin", "shifted": "optimum moved away from the origin"}
LABEL = {"PSO": "PSO", "GWO": "GWO", "CPO": "CPO", "GWO-PSO": "GWO + PSO", "GWO-CPO": "GWO + CPO (ours)"}
assert list(LABEL) == list(OPTIMISERS) and list(FORM_LABEL) == list(FORMS)


def _legend(fig) -> None:
    """The two forms, on a line of their own under the heading."""
    height = fig.get_figheight()
    fig.legend(handles=[Line2D([], [], linestyle="", marker="o", ms=8, mfc=SURFACE, mec=FORM_COLOR["published"], mew=1.8),
                        Line2D([], [], linestyle="", marker="o", ms=9.5, mfc=FORM_COLOR["shifted"], mec=SURFACE, mew=1.5)],
               labels=list(FORM_LABEL.values()), loc="upper left", bbox_to_anchor=(0.014, 1 - 0.8 / height), ncols=2,
               handlelength=1.2, handletextpad=0.5, columnspacing=1.6, borderaxespad=0)


def _rows(ax, table: pd.DataFrame, change) -> None:
    """One dumbbell per optimiser on `ax`: a ring for the published form and a dot for the shifted one."""
    for y, method in enumerate(LABEL):
        published, shifted = table.at[method, "published"], table.at[method, "shifted"]
        ax.plot([published, shifted], [y, y], color=AXIS, linewidth=2, solid_capstyle="round", zorder=2)
        ax.plot(published, y, "o", ms=8, mfc=SURFACE, mec=FORM_COLOR["published"], mew=1.8, zorder=3, clip_on=False)
        ax.plot(shifted, y, "o", ms=9.5, mfc=FORM_COLOR["shifted"], mec=SURFACE, mew=1.5, zorder=4, clip_on=False)
        if change:
            ax.text(1.035, y, change(published, shifted), transform=ax.get_yaxis_transform(), va="center",
                    color=INK_2, fontsize=9)
    ax.set_ylim(len(LABEL) - 0.4, -0.6)
    _quiet(ax, "x")
    ax.tick_params(axis="y", pad=8)


def _name_rows(ax) -> None:
    ax.set_yticks(range(len(LABEL)), list(LABEL.values()), color=INK)
    for tick, method in zip(ax.get_yticklabels(), LABEL):
        tick.set_fontweight("bold" if method == PROPOSED else "normal")


def rank_figure(ranks: dict[str, pd.DataFrame], path) -> None:
    """Mean rank over the functions, one panel per PSO velocity limit."""
    head, foot = 1.95, 0.65
    height = head + len(LABEL) * 0.36 + foot
    fig = plt.figure(figsize=(11.5, height))
    axes = fig.add_gridspec(1, 2, left=0.16, right=0.94, top=1 - head / height, bottom=foot / height,
                            wspace=0.22).subplots(sharey=True)
    for ax, (limit, table) in zip(axes, ranks.items()):
        _rows(ax, table, lambda published, shifted: f"{shifted - published:+.2f}".replace("-", "−"))
        ax.text(1.035, -0.75, "change", transform=ax.get_yaxis_transform(), va="center", color=MUTED, fontsize=9)
        ax.annotate(f"PSO velocity limit: {limit}", (0, 1), xycoords="axes fraction", xytext=(0, 14),
                    textcoords="offset points", fontsize=10.5, fontweight="bold")
        ax.set_xlim(1, len(LABEL))
        ax.set_xticks(range(1, len(LABEL) + 1))
        ax.set_xlabel(f"mean rank (1 is best of {len(LABEL)})")
    _name_rows(axes[0])
    _heading(fig, f"Rank on {len(FUNCTIONS)} benchmark functions, as published and with the optimum moved",
             f"Mean rank over F1 to F{len(FUNCTIONS)} in {DIM} dimensions, {RUNS} runs each, {BUDGET:,} fitness calls per run.")
    _legend(fig)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def sphere_figure(medians: pd.DataFrame, path) -> None:
    """The sphere function as the example: median best value of each optimiser in both forms, on a log axis."""
    head, foot = 1.45, 0.65
    height = head + len(LABEL) * 0.36 + foot
    fig = plt.figure(figsize=(11.5, height))
    ax = fig.add_gridspec(1, 1, left=0.16, right=0.96, top=1 - head / height, bottom=foot / height).subplots()
    _rows(ax, medians, None)
    ax.set_xscale("log")
    ax.set_xlabel("median best value found (lower is better; the optimum is 0)")
    _name_rows(ax)
    _heading(fig, "The sphere function, F1: the same optimisers, with the optimum in two places",
             f"Median over {RUNS} runs in {DIM} dimensions. Both forms are the same function. Only the place of its minimum differs.")
    _legend(fig)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> int:
    ranks, medians = {}, {}
    for limit, folder in [(f"{PSO_VMAX:.0%} of the range (registered run)", OUT),
                          ("10% of the range (sensitivity run)", SENSITIVITY)]:
        ranks[limit] = pd.read_csv(folder / "benchmark_ranks.csv").set_index("method")
        summary = pd.read_csv(folder / "benchmark_summary.csv", float_precision="round_trip")
        medians[limit] = summary.pivot(index=["function", "form"], columns="method", values="median")[list(OPTIMISERS)]
        order = [(fn.name, form) for fn in FUNCTIONS for form in FORMS]
        medians[limit].loc[order].to_csv(folder / "benchmark_medians.csv", float_format="%.4g")
    rank_figure(ranks, OUT / "benchmark_ranks.png")
    registered = next(iter(medians.values()))
    sphere_figure(registered.loc["F1"].T, OUT / "benchmark_sphere.png")
    print(f"table and graphs written to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
