"""Answers, claims and figures of Experiment 13, exactly as fixed in design/amendment-13-what-matters.md.

    .venv/bin/python analyze_what_matters.py                         # results/what_matters/
    .venv/bin/python analyze_what_matters.py results/what_matters_smoke

Writes into <dir>/summary/: means, comparisons and claims as CSV, and figures 1-10 as PNG.
Figures are numbered in the paper's reading order. Amendment 13 lists the same ten figures in another
order; paper number (amendment number): 1 (1), 2 (4), 3 (3), 4 (2), 5 (8), 6 (10), 7 (5), 8 (6), 9 (7), 10 (9).
"""
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from analyze import _paired, holm  # noqa: E402
from what_matters_demo import CLASSIFIERS, FILTERS, MISSING, SELECTORS, WRAPPERS  # noqa: E402

M0, M1 = MISSING
GUIDED = ("GWO-CPO guided", "BPSO guided", "GA guided")
MARGIN = 0.03
BOOT = 2000
TASKS = ("two-class", "twelve-class")
# Reference palette of the dataviz skill: categorical slots in fixed order, a blue ramp, a blue<->red diverging pair.
SERIES = dict(zip(CLASSIFIERS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUES = LinearSegmentedColormap.from_list("blues", ["#f4f8fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
DIVERGE = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#f0a3a2", "#b42f2f"])
PHATAI = {  # Phatai et al. 2026, Table II, two classes, accuracy, 5 x 5 cross-validation (their folds, not ours)
    ("random forest", M0, "none"): 0.7991, ("random forest", M1, "none"): 0.7995,
    ("random forest", M1, "ReliefF"): 0.8221, ("SVM", M1, "none"): 0.7499, ("SVM", M1, "ReliefF"): 0.7814,
}

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.dpi": 200,
                     "legend.frameon": False})


def wide(rows, metric="macro_f1"):
    return rows.pivot_table(index=["repeat", "fold"], columns=["missing", "selector", "classifier"], values=metric)


def diff(w, a, b):
    d = (w[a] - w[b]).to_numpy()
    assert not np.isnan(d).any()
    return _paired(d)


def verdict(r, adjusted):
    if adjusted < 0.05:
        return "higher" if r["mean_diff"] > 0 else "lower"
    return "no clear difference"


def q1(task, w):
    out = []
    for sel in SELECTORS:
        block = [dict(question="Q1 indicators", task=task, selector=sel, classifier=c, a=M1, b=M0,
                      **diff(w, (M1, sel, c), (M0, sel, c))) for c in CLASSIFIERS]
        for r, p in zip(block, holm(np.array([r["p_exploratory"] for r in block]))):
            r.update(p_holm=p, claim=sel == "none", verdict=verdict(r, p) if sel == "none" else "")
        out += block
    return out


def q2(task, w):
    out = []
    for c in CLASSIFIERS:
        block = [dict(question="Q2 selection", task=task, selector=s, classifier=c, a=s, b="none",
                      **diff(w, (M0, s, c), (M0, "none", c))) for s in SELECTORS[1:]]
        for r, p in zip(block, holm(np.array([r["p_exploratory"] for r in block]))):
            r.update(p_holm=p, claim=True, verdict=verdict(r, p))
        out += block
    return out


def q3(task, w):
    out = []
    for c in CLASSIFIERS:
        block = [dict(question="Q3 optimisers", task=task, selector=f"{a} - {b}", classifier=c, a=a, b=b,
                      **diff(w, (M0, a, c), (M0, b, c))) for a, b in combinations(GUIDED, 2)]
        same = all(-MARGIN < r["ci_low"] and r["ci_high"] < MARGIN for r in block)
        for r in block:
            r.update(claim=True, verdict="interchangeable" if same else "not shown interchangeable")
        out += block
    r = dict(question="Q3 replication", task=task, selector="GWO-CPO guided - IBGWO4 as published",
             classifier="5-NN", a="GWO-CPO guided", b="IBGWO4 as published",
             **diff(w, (M0, "GWO-CPO guided", "5-NN"), (M0, "IBGWO4 as published", "5-NN")))
    r.update(claim=True, verdict="replicates" if r["mean_diff"] > 0 and r["p_exploratory"] < 0.05 else "does not replicate")
    return out + [r]


def shares(Y):
    """Variance shares of a balanced fold x M x S x C array, after removing each fold's mean."""
    Y = Y - Y.mean(axis=(1, 2, 3), keepdims=True)
    total = (Y ** 2).sum()
    nf, nm, ns, nc = Y.shape
    m, s, c = Y.mean((0, 2, 3)), Y.mean((0, 1, 3)), Y.mean((0, 1, 2))
    mc = Y.mean((0, 2)) - m[:, None] - c[None, :]
    sc = Y.mean((0, 1)) - s[:, None] - c[None, :]
    ss = {"missing values": nf * ns * nc * (m ** 2).sum(), "selection": nf * nm * nc * (s ** 2).sum(),
          "classifier": nf * nm * ns * (c ** 2).sum(), "missing x classifier": nf * ns * (mc ** 2).sum(),
          "selection x classifier": nf * nm * (sc ** 2).sum()}
    ss["rest (other interactions, fold noise)"] = total - sum(ss.values())
    return {k: v / total for k, v in ss.items()}


def q4(task, rows):
    w = wide(rows)
    Y = np.stack([w[[(m, s, c) for s in SELECTORS for c in CLASSIFIERS]].to_numpy().reshape(len(w), len(SELECTORS),
                  len(CLASSIFIERS)) for m in MISSING], axis=1)
    point = shares(Y)
    rng = np.random.default_rng(0)
    boot = pd.DataFrame([shares(Y[rng.integers(0, len(Y), len(Y))]) for _ in range(BOOT)])
    table = [dict(task=task, part=k, share=v, ci_low=boot[k].quantile(0.025), ci_high=boot[k].quantile(0.975))
             for k, v in point.items()]
    claims = []
    for a, b in combinations(["missing values", "selection", "classifier"], 2):
        d = boot[a] - boot[b]
        lo, hi = d.quantile(0.025), d.quantile(0.975)
        claims.append(dict(question="Q4 knobs", task=task, selector=f"{a} vs {b}", mean_diff=point[a] - point[b],
                           ci_low=lo, ci_high=hi, claim=True,
                           verdict=f"{a} matters more" if lo > 0 else f"{b} matters more" if hi < 0 else "no clear order"))
    return table, claims


def q5(task, rows):
    r = rows[(rows.missing == M1) & (rows.selector != "none")].copy()
    r["excess"] = r.n_indicators_kept - r.n_features * r.n_indicator_columns / r.n_columns
    out = []
    for sel in SELECTORS[1:]:
        per_fold = r[r.selector == sel].groupby(["repeat", "fold"]).excess.mean().to_numpy()
        p = _paired(per_fold)
        out.append(dict(question="Q5 indicators picked", task=task, selector=sel, **p, claim=True,
                        verdict="more than chance" if p["mean_diff"] > 0 and p["p_exploratory"] < 0.05 else "not shown"))
    return out


# ---------- figures ----------

def save(fig, out, name):
    fig.savefig(out / name, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def fig_heatmap(rows, out):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)
    for i, task in enumerate(TASKS):
        means = rows[rows.task == task].groupby(["missing", "selector", "classifier"]).macro_f1.mean()
        lo, hi = means.min(), means.max()
        for j, m in enumerate(MISSING):
            ax = axes[i, j]
            grid = means[m].unstack("classifier").loc[list(SELECTORS), list(CLASSIFIERS)]
            ax.imshow(grid.to_numpy(), cmap=BLUES, vmin=lo, vmax=hi, aspect="auto")
            for (r, c), v in np.ndenumerate(grid.to_numpy()):
                ax.text(c, r, f"{v:.3f}", ha="center", va="center", fontsize=8,
                        color="white" if v > lo + 0.6 * (hi - lo) else INK)
            ax.set_xticks(range(len(CLASSIFIERS)), CLASSIFIERS)
            ax.set_yticks(range(len(SELECTORS)), SELECTORS)
            ax.set_title(f"{task}, {m}", color=INK, loc="left")
            ax.spines[:].set_visible(False)
            ax.tick_params(length=0)
    fig.suptitle("Figure 1. Mean macro-F1 over 25 test folds", x=0.01, ha="left", color=INK)
    save(fig, out, "fig01_heatmap_macro_f1.png")


def dot_whisker(ax, table, labels, title):
    offsets = np.linspace(-0.27, 0.27, len(CLASSIFIERS))
    for k, c in enumerate(CLASSIFIERS):
        t = table[table.classifier == c].set_index("selector").loc[labels]
        y = np.arange(len(labels)) + offsets[k]
        ax.hlines(y, t.ci_low, t.ci_high, color=SERIES[c], lw=2)
        ax.plot(t.mean_diff, y, "o", ms=6, color=SERIES[c], mec=SURFACE, mew=1.5, label=c)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, color=INK, loc="left")
    ax.set_xlabel("difference in macro-F1 (95% bootstrap interval)")


def fig_effects(comp, out, question, labels, name, title):
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.45 * len(labels) + 1.8), sharey=True, constrained_layout=True)
    for ax, task in zip(axes, TASKS):
        dot_whisker(ax, comp[(comp.question == question) & (comp.task == task)], labels, task)
    axes[1].legend(loc="lower right")
    fig.suptitle(title, x=0.01, ha="left", color=INK)
    save(fig, out, name)


def fig_shares(table, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3), sharey=True, constrained_layout=True)
    for ax, task in zip(axes, TASKS):
        t = table[table.task == task]
        y = np.arange(len(t))
        ax.barh(y, t.share, color="#86b6ef", height=0.6)
        ax.hlines(y, t.ci_low, t.ci_high, color=INK, lw=1.5)
        ax.set_yticks(y, t.part)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_title(task, color=INK, loc="left")
        ax.set_xlabel("share of the variation in macro-F1 (95% interval over folds)")
    fig.suptitle("Figure 2. Which knob moves the score", x=0.01, ha="left", color=INK)
    save(fig, out, "fig02_variance_shares.png")


def fig_folds(rows, out):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    rng = np.random.default_rng(1)
    for ax, task in zip(axes, TASKS):
        labels, x = [], 0
        for c in CLASSIFIERS:
            for sel in ("none", "GWO-CPO guided"):
                for m in MISSING:
                    v = rows[(rows.task == task) & (rows.classifier == c) & (rows.selector == sel) & (rows.missing == m)]
                    ax.boxplot(v.macro_f1, positions=[x], widths=0.6, showfliers=False,
                               medianprops=dict(color=INK), boxprops=dict(color=INK2), whiskerprops=dict(color=INK2),
                               capprops=dict(color=INK2))
                    ax.plot(x + rng.uniform(-0.18, 0.18, len(v)), v.macro_f1, "o", ms=3, alpha=0.7,
                            color=SERIES[c])
                    short = {"random forest": "RF", "XGBoost": "XGB"}.get(c, c)
                    labels.append(f"{short}\n{'all' if sel == 'none' else 'GWO-CPO'}\n{'+ind' if m == M1 else 'med'}")
                    x += 1
            x += 0.5
        ax.set_xticks([i + 0.5 * (i // 4) for i in range(len(labels))], labels, fontsize=6.5)
        ax.set_ylabel("macro-F1 per test fold")
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_title(task, color=INK, loc="left")
    fig.suptitle("Figure 7. Spread over the 25 test folds (all = no selection, med = median, +ind = with indicators)",
                 x=0.01, ha="left", color=INK)
    save(fig, out, "fig07_fold_spread.png")


def fig_size(rows, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, task in zip(axes, TASKS):
        means = rows[rows.task == task].groupby(["missing", "selector", "classifier"])[["n_features", "macro_f1"]].mean()
        for c in CLASSIFIERS:
            t = means.xs(c, level="classifier")
            ax.plot(t.n_features, t.macro_f1, "o", ms=7, color=SERIES[c], mec=SURFACE, mew=1.5, label=c)
        ax.set_xscale("log")
        ax.set_xlabel("mean number of features kept (log scale)")
        ax.set_ylabel("mean macro-F1")
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_title(task, color=INK, loc="left")
    axes[0].legend()
    fig.suptitle("Figure 8. Features kept against score, one dot per pipeline", x=0.01, ha="left", color=INK)
    save(fig, out, "fig08_size_vs_score.png")


def fig_recall(rows, out, class_names):
    r = rows[(rows.task == "twelve-class") & (rows.selector == "none") & (rows.missing == M0)]
    cols = [c for c in rows.columns if c.startswith("recall_") and r[c].notna().any()]
    grid = r.groupby("classifier")[cols].mean().loc[list(CLASSIFIERS)]
    fig, ax = plt.subplots(figsize=(11, 3), constrained_layout=True)
    ax.imshow(grid.to_numpy(), cmap=BLUES, vmin=0, vmax=1, aspect="auto")
    for (i, j), v in np.ndenumerate(grid.to_numpy()):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v > 0.6 else INK)
    ax.set_xticks(range(len(cols)), [class_names[int(c.split("_")[1])] for c in cols], rotation=30, ha="right")
    ax.set_yticks(range(len(CLASSIFIERS)), CLASSIFIERS)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Figure 9. Twelve classes: recall of each class, all features, median filling", color=INK, loc="left")
    save(fig, out, "fig09_per_class_recall.png")


def fig_indicators(rows, out):
    r = rows[(rows.missing == M1) & (rows.selector != "none")]
    attrs = sorted({int(a) for s in r.indicators_kept.dropna().astype(str) for a in s.split(";") if a and a != "nan"})
    if not attrs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True, constrained_layout=True)
    for ax, task in zip(axes, TASKS):
        t = r[r.task == task]
        kept = {a: t.groupby("selector").indicators_kept.apply(
            lambda s, a=a: s.fillna("").astype(str).str.split(";").apply(lambda xs: str(a) in xs).mean()) for a in attrs}
        grid = pd.DataFrame(kept).loc[list(SELECTORS[1:])]
        ax.imshow(grid.to_numpy(), cmap=BLUES, vmin=0, vmax=1, aspect="auto")
        for (i, j), v in np.ndenumerate(grid.to_numpy()):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v > 0.6 else INK)
        ax.set_xticks(range(len(attrs)), [f"attr {a}" for a in attrs])
        ax.set_yticks(range(len(grid)), grid.index)
        ax.spines[:].set_visible(False)
        ax.tick_params(length=0)
        ax.set_title(task, color=INK, loc="left")
    fig.suptitle("Figure 5. Share of pipelines that keep each 'value was missing' column", x=0.01, ha="left", color=INK)
    save(fig, out, "fig05_indicators_kept.png")


def fig_confusion(preds, out, class_names):
    p = preds[(preds.task == "twelve-class") & (preds.classifier == "XGBoost") & (preds.selector == "none")
              & (preds.missing == M0)]
    C = len(class_names)
    cm = np.zeros((C, C))
    np.add.at(cm, (p.y.astype(int), p.yhat.astype(int)), 1)
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
    ax.imshow(norm, cmap=BLUES, vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(norm):
        if cm[i, j]:
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > 0.6 else INK)
    ax.set_xticks(range(C), class_names, rotation=45, ha="right")
    ax.set_yticks(range(C), class_names)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class (each row sums to 1)")
    ax.spines[:].set_visible(False)
    ax.set_title("Figure 10. XGBoost, all features, median filling,\ntwelve classes, 25 test folds pooled",
                 color=INK, loc="left")
    save(fig, out, "fig10_confusion_xgboost.png")


def fig_curves(curves, out):
    fig, axes = plt.subplots(2, len(WRAPPERS), figsize=(13, 5.5), constrained_layout=True)
    for i, task in enumerate(TASKS):
        for j, sel in enumerate(WRAPPERS):
            ax = axes[i, j]
            for m, color in zip(MISSING, ["#2a78d6", "#eb6834"]):
                t = curves[(curves.task == task) & (curves.selector == sel) & (curves.missing == m)]
                mean = t.groupby("point").best_fitness.mean()
                ax.plot((mean.index + 1) / len(mean) * 100, mean.to_numpy(), lw=2, color=color, label=m)
            ax.grid(color=GRID, lw=0.8)
            ax.set_title(f"{task}: {sel}", color=INK, loc="left", fontsize=8)
            ax.set_xlabel("% of the 2,000 fitness calls")
            if j == 0:
                ax.set_ylabel("best inner fitness (lower is better)")
    axes[0, 0].legend()
    fig.suptitle("Figure 6. Mean search progress (IBGWO4 uses the accuracy fitness, so its scale differs)",
                 x=0.01, ha="left", color=INK)
    save(fig, out, "fig06_convergence.png")


def main():
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "results" / "what_matters"
    out = base / "summary"
    out.mkdir(exist_ok=True)
    rows = pd.read_csv(base / "what_matters_rows.csv")
    preds = pd.read_csv(base / "what_matters_predictions.csv")
    curves = pd.read_csv(base / "what_matters_curves.csv")
    smoke = "smoke" in base.name
    if smoke:
        names = [f"class {i}" for i in range(12)]
    else:
        from arrfs.data import load_task
        names = load_task("arrhythmia-multiclass").class_names

    means = rows.groupby(["task", "missing", "selector", "classifier"], sort=False)[
        ["macro_f1", "bacc", "acc", "n_features"]].agg(["mean", "std"])
    means.columns = [f"{a}_{b}" for a, b in means.columns]
    means.reset_index().round(4).to_csv(out / "means.csv", index=False)

    comp, share_rows = [], []
    for task in TASKS:
        t = rows[rows.task == task]
        w = wide(t)
        comp += q1(task, w) + q2(task, w) + q3(task, w)
        table, claims = q4(task, t)
        share_rows += table
        comp += claims + q5(task, t)
    comp = pd.DataFrame(comp)
    comp.round(4).to_csv(out / "comparisons.csv", index=False)
    claims = comp[comp.claim == True]  # noqa: E712
    claims.round(4).to_csv(out / "claims.csv", index=False)
    shares_df = pd.DataFrame(share_rows)
    shares_df.round(4).to_csv(out / "variance_shares.csv", index=False)

    acc = rows[rows.task == "two-class"].groupby(["classifier", "missing", "selector"]).acc.mean()
    pd.DataFrame([dict(classifier=c, missing=m, selector=s, ours_accuracy=acc[(c, m, s)], phatai_accuracy=v)
                  for (c, m, s), v in PHATAI.items()]).round(4).to_csv(out / "phatai_comparison.csv", index=False)

    fig_heatmap(rows, out)
    fig_effects(comp, out, "Q1 indicators", list(SELECTORS), "fig04_indicator_effect.png",
                "Figure 4. Adding 'value was missing' columns: with minus without")
    fig_effects(comp, out, "Q2 selection", list(SELECTORS[1:]), "fig03_selection_effect.png",
                "Figure 3. Each selector minus no selection (median filling)")
    fig_shares(shares_df, out)
    fig_folds(rows, out)
    fig_size(rows, out)
    fig_recall(rows, out, names)
    fig_indicators(rows, out)
    fig_confusion(preds, out, names)
    fig_curves(curves, out)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 60)
    show = ["question", "task", "selector", "classifier", "mean_diff", "ci_low", "ci_high", "p_exploratory", "p_holm",
            "verdict"]
    print(claims[[c for c in show if c in claims]].round(4).to_string(index=False))
    print(shares_df.round(3).to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
