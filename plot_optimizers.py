"""Table and graphs of the optimiser comparison of Amendments 9 and 10. They are comparison rows: no claim is tested.

Reads results/more_optimizers/all_optimizers_rows.csv and all_optimizers_curves.csv (more_optimizers_demo.py)
and writes next to them, for one fitness kind:

    table_<fitness>_fitness.csv                          5-NN means per optimiser, task and start
    scores_macro_f1_<fitness>_fitness.png                5-NN macro-F1 per optimiser, random start against informed start
    scores_accuracy_<fitness>_fitness.png                the same for accuracy
    features_<fitness>_fitness.png                       number of selected attributes per optimiser and start
    convergence_all_<classes>_<fitness>_fitness.png      mean best-so-far fitness against fitness calls, a panel per optimiser
    convergence_hybrids_<classes>_<fitness>_fitness.png  the same for the hybrids in one panel per start

Run from code/:  .venv/bin/python plot_optimizers.py [imbalance|accuracy]     (default: the proposed model's fitness)

Colours come from the project's chart palette. The two starts keep one colour each in every graph. In the hybrid
graph a hybrid takes the hue and marker of its second optimiser, and grey wolf alone is the neutral line. Those
four hues stay apart on every pair, for readers with and without colour-vision deficiency (checked with the
dataviz validator).
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

from arrfs.experiment import FITNESS_KINDS, PROPOSED, Settings  # noqa: E402
from arrfs.optimizers import CURVE_POINTS  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "more_optimizers"

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
START_COLOR = {"random": MUTED, "informed": "#6250d6"}
SECOND = {"BPSO": ("#2a78d6", "o"), "BCPO": ("#008300", "^"), "BCS": ("#eda100", "D"), "BWOA": ("#e87ba4", "v")}
ALONE = (INK_2, "s")  # grey wolf without a second optimiser, in the hybrid graph

TASKS = {"arrhythmia-multiclass": ("12 classes", ""), "arrhythmia-binary": ("2 classes", " (normal or arrhythmia)")}
STARTS = {"random": "random start", "informed": "informed start (mutual information)"}
FITNESS = {"imbalance": "imbalance-aware", "accuracy": "accuracy"}
GROUPS = {False: "Single optimisers, oldest first", True: "Grey wolf, then a second optimiser"}  # Optimiser.hybrid -> caption


@dataclass(frozen=True)
class Optimiser:
    method: str       # its name in the result files
    label: str        # its name on a graph
    year: int         # year of the paper that introduced it; single optimisers are drawn oldest first
    second: str = ""  # a hybrid's second optimiser, a key of SECOND; empty for a single optimiser

    @property
    def hybrid(self) -> bool:
        return bool(self.second)


OPTIMISERS = (
    Optimiser("GA", "Genetic algorithm (GA)", 1975),
    Optimiser("BPSO", "Particle swarm (PSO)", 1995),
    Optimiser("BDE", "Differential evolution (DE)", 1997),
    Optimiser("BCS", "Cuckoo search", 2009),
    Optimiser("BBA", "Bat algorithm", 2010),
    Optimiser("BGWO", "Grey wolf (GWO)", 2014),
    Optimiser("BDA", "Dragonfly", 2016),
    Optimiser("BSCA", "Sine cosine", 2016),
    Optimiser("BWOA", "Whale", 2016),
    Optimiser("BSSA", "Salp swarm", 2017),
    Optimiser("BHHO", "Harris hawks", 2019),
    Optimiser("BMPA", "Marine predators", 2020),
    Optimiser("BCPO", "Crested porcupine (CPO)", 2024),
    Optimiser("IBGWO4", "GWO + PSO (base paper)", 2025, "BPSO"),
    Optimiser("GWO-CPO", "GWO + CPO (proposed)", 2026, "BCPO"),
    Optimiser("GWO-CS", "GWO + cuckoo search", 2026, "BCS"),
    Optimiser("GWO-WOA", "GWO + whale", 2026, "BWOA"),
)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.edgecolor": AXIS, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def _signed(digits: int):
    return lambda change: f"{change:+.{digits}f}".replace("-", "−")


def _spread(panel: pd.DataFrame) -> str:
    gap = panel.max() - panel.min()
    return "\n".join(f"Best minus worst optimiser, {start} start: {gap[start]:.3f}" for start in STARTS)


def _range(panel: pd.DataFrame) -> str:
    return "\n".join(f"{start.capitalize()} start: {panel[start].min():.0f} to {panel[start].max():.0f} attributes"
                     for start in STARTS)


def _heading(fig, title: str, *lines: str) -> None:
    """A bold title and grey lines under it, at fixed distances in inches from the top-left corner."""
    height = fig.get_figheight()
    fig.text(0.02, 1 - 0.34 / height, title, fontsize=13, fontweight="bold")
    for n, line in enumerate(lines):
        fig.text(0.02, 1 - (0.62 + 0.23 * n) / height, line, color=INK_2)


def _quiet(ax, grid: str) -> None:
    ax.grid(axis=grid)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=3, color=AXIS)


def dumbbell(optimisers, values: pd.DataFrame, title: str, subtitle: str, xlabel: str, note, change, step: float,
             path: Path) -> None:
    """One panel per task: every optimiser's mean from the random start (ring) and the informed start (dot).

    `values` is indexed by (task, method) with one column per start. The panels get the same axis span, so a
    given length is the same change in both. A group caption takes the free row above its group."""
    row, caption_row = {}, {}
    for opt in optimisers:
        caption_row.setdefault(opt.hybrid, len(row) + len(caption_row))
        row[opt.method] = len(row) + len(caption_row)
    head, pitch, foot = 1.72, 0.305, 0.63  # inches above the axes, per row, and below the axes
    height = head + (len(row) + len(caption_row)) * pitch + foot
    low = {task: np.floor(values.loc[task].to_numpy().min() / step) * step for task in TASKS}
    span = max(np.ceil(values.loc[task].to_numpy().max() / step) * step - low[task] for task in TASKS)
    fig = plt.figure(figsize=(11.5, height))
    axes = fig.add_gridspec(1, 2, left=0.235, right=0.94, top=1 - head / height, bottom=foot / height,
                            wspace=0.27).subplots(sharey=True)
    for ax, (task, (classes, gloss)) in zip(axes, TASKS.items()):
        panel = values.loc[task]
        for opt in optimisers:
            y, random, informed = row[opt.method], panel.at[opt.method, "random"], panel.at[opt.method, "informed"]
            ax.plot([random, informed], [y, y], color=AXIS, linewidth=2, solid_capstyle="round", zorder=2)
            ax.plot(random, y, "o", ms=8, mfc=SURFACE, mec=START_COLOR["random"], mew=1.8, zorder=3, clip_on=False)
            ax.plot(informed, y, "o", ms=9.5, mfc=START_COLOR["informed"], mec=SURFACE, mew=1.5, zorder=4, clip_on=False)
            ax.text(1.035, y, change(informed - random), transform=ax.get_yaxis_transform(), va="center",
                    color=INK_2, fontsize=9)
        ax.text(1.035, caption_row[False], "change", transform=ax.get_yaxis_transform(), va="center", color=MUTED,
                fontsize=9)
        ax.annotate(classes + gloss, (0, 1), xycoords="axes fraction", xytext=(0, 43), textcoords="offset points",
                    fontsize=11, fontweight="bold")
        ax.annotate(note(panel), (0, 1), xycoords="axes fraction", xytext=(0, 10), textcoords="offset points",
                    fontsize=9, color=INK_2, linespacing=1.5)
        ax.set_xlim(low[task], low[task] + span)
        ax.xaxis.set_major_locator(MultipleLocator(step))
        ax.set_xlabel(xlabel)
        _quiet(ax, "x")
        ax.tick_params(axis="y", pad=8)
    axes[0].set_ylim(max(row.values()) + 0.6, min(caption_row.values()) - 0.6)
    axes[0].set_yticks(list(row.values()), [opt.label for opt in optimisers], color=INK)
    for tick, opt in zip(axes[0].get_yticklabels(), optimisers):
        tick.set_fontweight("bold" if opt.method == PROPOSED[0] else "normal")
    for hybrid, y in caption_row.items():
        axes[0].annotate(GROUPS[hybrid], (0, y), xycoords=axes[0].get_yaxis_transform(), xytext=(-8, 0),
                         textcoords="offset points", ha="right", va="center", color=MUTED, fontsize=9)
    _heading(fig, title, subtitle)
    fig.legend(handles=[Line2D([], [], linestyle="", marker="o", ms=8, mfc=SURFACE, mec=START_COLOR["random"], mew=1.8),
                        Line2D([], [], linestyle="", marker="o", ms=9.5, mfc=START_COLOR["informed"], mec=SURFACE, mew=1.5)],
               labels=list(STARTS.values()), loc="upper right", bbox_to_anchor=(0.985, 1 - 0.09 / height), ncols=2,
               handlelength=1.2, handletextpad=0.5, columnspacing=1.6, borderaxespad=0)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def convergence_all(optimisers, curves: pd.Series, task: str, lines: tuple[str, ...], path: Path) -> None:
    """Mean best-so-far fitness against fitness calls, one small panel per optimiser and one line per start.

    `curves` is indexed by (task, method, start, point) and holds one fitness kind. Every panel has the same
    axes, so heights compare across panels. The last cell of the grid holds the legend."""
    budget = Settings().budget
    calls = np.arange(1, CURVE_POINTS + 1) * budget / CURVE_POINTS
    columns = 6
    rows = -(-(len(optimisers) + 1) // columns)
    head, cell = 1.3, 1.95
    height = head + rows * cell + 0.55
    fig, axes = plt.subplots(rows, columns, figsize=(13.5, height), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.055, right=0.99, top=1 - head / height, bottom=0.55 / height, wspace=0.08,
                        hspace=0.34)
    cells = list(axes.flat)
    for ax, opt in zip(cells, optimisers):
        for start in STARTS:
            ax.plot(calls, curves.loc[(task, opt.method, start)].to_numpy(), color=START_COLOR[start], linewidth=1.7,
                    solid_capstyle="round")
        ax.set_title(opt.label, loc="left", fontsize=9.5, color=INK, pad=5,
                     fontweight="bold" if opt.method == PROPOSED[0] else "normal")
        _quiet(ax, "y")
        ax.tick_params(labelbottom=True)
        ax.set_xticks([0, budget // 2, budget])
        for tick, side in zip(ax.get_xticklabels(), ("left", "center", "right")):
            tick.set_horizontalalignment(side)
        ax.set_xlim(0, budget * 1.02)
    for ax in cells[len(optimisers):]:
        ax.set_axis_off()
    cells[-1].legend(handles=[Line2D([], [], color=START_COLOR[start], linewidth=1.7) for start in STARTS],
                     labels=[label.split(" (")[0] for label in STARTS.values()], loc="center left", handlelength=2.2)
    fig.text(0.012, 1 - (head + rows * cell / 2) / height, "best fitness so far (lower is better)", rotation=90,
             va="center", color=INK_2)
    fig.text(0.5, 0.1 / height, "fitness calls", ha="center", color=INK_2)
    _heading(fig, f"How fast each optimiser improves, {TASKS[task][0]}", *lines)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def convergence_hybrids(optimisers, curves: pd.Series, task: str, lines: tuple[str, ...], path: Path) -> None:
    """The hybrids and grey wolf alone in one panel per start, so that the hybrids can be compared directly."""
    budget = Settings().budget
    calls = np.arange(1, CURVE_POINTS + 1) * budget / CURVE_POINTS
    shown = [(opt.label, *SECOND[opt.second], 3, opt.method) for opt in optimisers if opt.hybrid]
    shown.append(("Grey wolf alone", *ALONE, 2, "BGWO"))
    head = 1.55
    height = head + 3.3 + 0.6
    fig, axes = plt.subplots(1, 2, figsize=(11.5, height), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.075, right=0.985, top=1 - head / height, bottom=0.6 / height, wspace=0.06)
    for ax, start in zip(axes, STARTS):
        for n, (label, hue, marker, zorder, method) in enumerate(shown):
            ax.plot(calls, curves.loc[(task, method, start)].to_numpy(), color=hue, linewidth=1.7, marker=marker,
                    ms=6, mfc=hue, mec=SURFACE, mew=1, markevery=(3 + 3 * n, 16), solid_capstyle="round",
                    zorder=zorder, label=label)
        ax.set_title(f"{start.capitalize()} start", loc="left", fontsize=10.5, fontweight="bold", pad=8)
        _quiet(ax, "y")
        ax.xaxis.set_major_locator(MultipleLocator(500))
        ax.set_xlim(0, budget * 1.01)
        ax.set_xlabel("fitness calls")
    axes[1].legend(loc="upper right", borderaxespad=0.2, handlelength=2.6, labelcolor=INK)
    axes[0].set_ylabel("best fitness so far (lower is better)")
    _heading(fig, f"Grey wolf followed by a second optimiser, {TASKS[task][0]}", *lines,
             "A hybrid has the colour and marker of its second optimiser.")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def table(optimisers, means: pd.DataFrame) -> pd.DataFrame:
    """One row per optimiser: macro-F1, accuracy and attribute count for each task and start."""
    columns = {f"{classes} / {name} / {start} start": means.loc[task][(metric, start)]
               for task, (classes, _) in TASKS.items()
               for metric, name in [("macro_f1", "macro-F1"), ("acc", "accuracy"), ("n_features", "attributes")]
               for start in STARTS}
    out = pd.DataFrame(columns).loc[[opt.method for opt in optimisers]]
    out.insert(0, "optimiser", [opt.label for opt in optimisers])
    out.insert(1, "year", [opt.year for opt in optimisers])
    return out.rename_axis("method").reset_index()


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    fitness = argv[0] if argv else PROPOSED[1]
    if len(argv) > 1 or fitness not in FITNESS_KINDS:
        print(__doc__)
        return 2
    rows = pd.read_csv(OUT / "all_optimizers_rows.csv", float_precision="round_trip")
    unknown = set(rows.method) - {opt.method for opt in OPTIMISERS}
    assert not unknown, f"no label for {sorted(unknown)}"
    optimisers = [opt for opt in OPTIMISERS if opt.method in set(rows.method)]
    knn = rows[(rows.classifier == "KNN") & (rows.fitness == fitness)]
    means = knn.groupby(["task", "method", "start"])[["macro_f1", "acc", "n_features"]].mean().unstack("start")
    assert len(means) == len(TASKS) * len(optimisers) and not means.isna().any().any()
    folds = knn.groupby(["repeat", "fold"]).ngroups
    count = f"{len(optimisers)} optimisers"
    searched = f"Mean of {folds} folds. The search used the {FITNESS[fitness]} fitness."

    table(optimisers, means).to_csv(OUT / f"table_{fitness}_fitness.csv", index=False, float_format="%.3f")
    for metric, name in [("macro_f1", "macro-F1"), ("acc", "accuracy")]:
        dumbbell(optimisers, means[metric], f"{count.capitalize()}, random start against informed start",
                 f"5-nearest-neighbour {name} on the held-out fold. {searched}",
                 f"{name} (higher is better)", _spread, _signed(3), 0.05,
                 OUT / f"scores_{name.replace('-', '_').lower()}_{fitness}_fitness.png")
    dumbbell(optimisers, means["n_features"], "How many attributes each optimiser keeps",
             f"Selected attributes out of the dataset's 279. {searched}",
             "selected attributes (fewer is simpler)", _range, _signed(0), 20, OUT / f"features_{fitness}_fitness.png")

    curves = pd.read_csv(OUT / "all_optimizers_curves.csv", float_precision="round_trip")
    curves = curves[curves.fitness == fitness].set_index(["task", "method", "start", "point"]).best_fitness.sort_index()
    lines = (f"Mean over {folds} folds of the best {FITNESS[fitness]} fitness found so far, on training patients only. "
             f"Every optimiser gets {Settings().budget:,} fitness calls.",)
    for task, (classes, _) in TASKS.items():
        stem = f"{classes.replace(' ', '_')}_{fitness}_fitness.png"
        convergence_all(optimisers, curves, task, lines, OUT / f"convergence_all_{stem}")
        convergence_hybrids(optimisers, curves, task, lines, OUT / f"convergence_hybrids_{stem}")

    pd.set_option("display.width", 200)
    print(means.round(3).to_string())
    print(f"table and graphs for {count} written to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
