"""The slide figures. One visual system: thin marks, hairline solid grid, text in ink
tokens (never in series colour), categorical hues in fixed order, diverging map for
signed weights, legend whenever there are two or more series.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402

from common import config as C  # noqa: E402

# ---- palette (reference instance, validated for CVD; slots 1-3 validate all-pairs) ----
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
ACTION_COLOR = {"answer": SERIES[0], "verify": SERIES[1], "abstain": SERIES[2]}
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
DIVERGING = LinearSegmentedColormap.from_list("kwtc_div", ["#2a78d6", "#f0efec", "#e34948"])
DPI = 200

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlecolor": INK, "text.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "legend.frameon": False, "legend.fontsize": 8,
})


def _ax(ax, title: str, xlabel: str = "", ylabel: str = "", grid_axis: str = "y"):
    ax.set_title(title, loc="left", fontweight="bold", pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.6, linestyle="-")
        ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _ink_for(fill_hex_or_rgb) -> str:
    r, g, b = to_rgb(fill_hex_or_rgb)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return INK if lum > 0.55 else "#ffffff"


def _save(fig, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


# 1 ------------------------------------------------------------------ risk-coverage
def fig_risk_coverage(curves: dict[str, list[tuple[float, float]]], areas: dict[str, float], path) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    _ax(ax, "Risk-coverage: selective accuracy vs coverage", "Coverage (share of cases answered)", "Selective accuracy")
    for i, (name, pts) in enumerate(curves.items()):
        pts = sorted(pts)
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        label = f"{name} (area {areas[name]:.3f})" if name in areas else name
        ax.plot(xs, ys, color=SERIES[i % len(SERIES)], linewidth=2, solid_capstyle="round", solid_joinstyle="round",
                marker="o", markersize=4.5, markeredgecolor=SURFACE, markeredgewidth=1.2, label=label)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left")
    return _save(fig, path)


# 2 ------------------------------------------------------------------ phase diagram
def fig_phase_diagram(ws: list[float], cs: list[float], majority: list[list[str]], utility: np.ndarray, path,
                      title="Reward design: majority action over (w, c)") -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    _ax(ax, title, "Tool cost c (per call)", "Wrong-answer penalty w", grid_axis="")
    for i, w in enumerate(ws):
        for j, c in enumerate(cs):
            col = ACTION_COLOR[majority[i][j]]
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=col, edgecolor=SURFACE, linewidth=2))
            ax.text(j + 0.5, i + 0.5, f"{utility[i][j]:+.2f}", ha="center", va="center", fontsize=8.5, color=_ink_for(col))
    ax.set_xlim(0, len(cs))
    ax.set_ylim(0, len(ws))
    ax.set_xticks([j + 0.5 for j in range(len(cs))], [f"{c:g}" for c in cs])
    ax.set_yticks([i + 0.5 for i in range(len(ws))], [f"{w:g}" for w in ws])
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.legend(handles=[Patch(facecolor=ACTION_COLOR[a], label=a) for a in C.ACTIONS], loc="upper left",
              bbox_to_anchor=(1.01, 1.0), title="majority action")
    ax.text(0, -0.9, "cell text = mean reward (utility) on test", color=INK2, fontsize=8)
    return _save(fig, path)


# 3 ------------------------------------------------------------------ grounding test
def fig_grounding(policies: list[str], parent_acc: list[float], flip_rate: list[float], stubborn: list[float], path) -> Path:
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    _ax(ax, "Grounding test: does the agent notice its evidence is gone?", "", "Rate")
    x = np.arange(len(policies))
    wd = 0.26
    series = [("Accuracy on parents", parent_acc, SERIES[0]), ("Grounding flip rate on twins", flip_rate, SERIES[1]),
              ("Stubborn rate on twins", stubborn, SERIES[2])]
    for k, (name, vals, col) in enumerate(series):
        vals = [np.nan if v is None else v for v in vals]
        bars = ax.bar(x + (k - 1) * wd, vals, width=wd - 0.03, color=col, label=name, edgecolor=SURFACE, linewidth=1)
        for b, v in zip(bars, vals):
            if v == v:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(x, policies)
    ax.set_ylim(0, 1.12)
    ax.legend(loc="upper left", ncol=3, bbox_to_anchor=(0, 1.02))
    return _save(fig, path)


# 4 ------------------------------------------------------------------ reliability
def fig_reliability(bins_raw: list[dict], bins_cal: list[dict], ece_raw: float, ece_cal: float, path) -> Path:
    fig, ax = plt.subplots(figsize=(5.6, 6.0))
    _ax(ax, "Reliability: raw confidence vs calibrated P(correct)", "Predicted probability", "Observed accuracy",
        grid_axis="both")
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1, zorder=1)
    for name, bins, ece, col in (("calibrated P(correct)", bins_cal, ece_cal, SERIES[0]),
                                 ("raw confidence", bins_raw, ece_raw, SERIES[1])):
        pts = [(b["confidence"], b["accuracy"], b["n"]) for b in bins if b["n"]]
        if not pts:
            continue
        xs, ys, ns = zip(*pts)
        ax.plot(xs, ys, color=col, linewidth=2, zorder=2)
        ax.scatter(xs, ys, s=[18 + 120 * n / max(ns) for n in ns], color=col, edgecolor=SURFACE, linewidth=1.2, zorder=3,
                   label=f"{name} (ECE {ece:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1)       # below the plot: never covers the curves
    ax.text(0.5, -0.30, "marker size = cases in bin; diagonal = perfect calibration", transform=ax.transAxes,
            ha="center", color=MUTED, fontsize=7.5)
    return _save(fig, path)


# 5 ------------------------------------------------------------------ cost-accuracy frontier
def fig_cost_frontier(ours: list[tuple[float, float, float]], baselines: dict[str, tuple[float, float]], path) -> Path:
    """ours: [(tool_calls_per_case, accuracy, c)], baselines: name -> (tool_calls, accuracy)."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    _ax(ax, "Cost-accuracy frontier (tool cost c varies)", "Tool calls per case", "Accuracy", grid_axis="both")
    pts = sorted(ours)
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=SERIES[0], linewidth=2, marker="o", markersize=5,
            markeredgecolor=SURFACE, markeredgewidth=1.2, label="ours (sweep over c)")
    merged: dict[tuple[float, float], list[float]] = {}        # identical policies share one label
    for tc, acc, c in pts:
        merged.setdefault((round(tc, 3), round(acc, 3)), []).append(c)
    for (tc, acc), cs in merged.items():
        ax.annotate("c=" + ", ".join(f"{c:g}" for c in cs), (tc, acc), textcoords="offset points", xytext=(6, 4),
                    fontsize=7.5, color=INK2)
    for i, (name, (tc, acc)) in enumerate(baselines.items()):
        ax.scatter([tc], [acc], color=SERIES[(i + 1) % len(SERIES)], s=46, edgecolor=SURFACE, linewidth=1.2, zorder=3, label=name)
        ax.annotate(name, (tc, acc), textcoords="offset points", xytext=(6, -9), fontsize=7.5, color=INK2)
    ax.legend(loc="lower right")
    return _save(fig, path)


# 6 ------------------------------------------------------------------ W heatmap
def fig_w_heatmap(W: np.ndarray, feature_names: list[str], action_names: list[str], path, reading: str = "") -> Path:
    W = np.asarray(W)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    _ax(ax, "Controller weights W (rows = actions, columns = z-scored signals)", grid_axis="")
    vmax = float(np.max(np.abs(W))) or 1.0
    im = ax.imshow(W, cmap=DIVERGING, vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            ax.text(j, i, f"{W[i, j]:+.2f}", ha="center", va="center", fontsize=8.5, color=_ink_for(DIVERGING((W[i, j] + vmax) / (2 * vmax))))
    ax.set_xticks(range(len(feature_names)), feature_names)
    ax.set_yticks(range(len(action_names)), action_names)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, colors=MUTED)
    if reading:
        ax.text(0, 1.18, reading, transform=ax.transAxes, color=INK2, fontsize=8.5, va="bottom")
    return _save(fig, path)


# 7 ------------------------------------------------------------------ action mix by case type
def fig_action_mix(mix: dict[str, dict[str, dict[str, int]]], case_types: list[str], path) -> Path:
    """mix[policy][case_type][action] = count. One panel per policy."""
    pols = list(mix)
    fig, axes = plt.subplots(1, len(pols), figsize=(3.3 * len(pols) + 1.2, 4.0), sharey=True, squeeze=False)
    pretty = {"both_right": "both\nright", "only_verify_right": "only verify\nright", "only_answer_right": "only answer\nright",
              "neither_right": "neither\nright"}
    for ax, pol in zip(axes[0], pols):
        _ax(ax, pol, "", "Share of cases" if pol == pols[0] else "")
        bottoms = np.zeros(len(case_types))
        for a in C.ACTIONS:
            vals = []
            for ct in case_types:
                tot = sum(mix[pol].get(ct, {}).values()) or 1
                vals.append(mix[pol].get(ct, {}).get(a, 0) / tot)
            ax.bar(range(len(case_types)), vals, bottom=bottoms, width=0.55, color=ACTION_COLOR[a], edgecolor=SURFACE,
                   linewidth=1.5, label=a)
            bottoms += np.asarray(vals)
        ax.set_xticks(range(len(case_types)), [pretty.get(ct, ct) for ct in case_types], fontsize=7.5)
        for k, ct in enumerate(case_types):
            n = sum(mix[pol].get(ct, {}).values())
            ax.text(k, 1.02, f"n={n}", ha="center", va="bottom", fontsize=7, color=MUTED)
        ax.set_ylim(0, 1.1)
    axes[0][-1].legend(handles=[Patch(facecolor=ACTION_COLOR[a], label=a) for a in C.ACTIONS], loc="upper left",
                       bbox_to_anchor=(1.01, 1.0), title="action")
    fig.suptitle("Action mix by case type (what verify could have fixed)", x=0.01, ha="left", fontsize=11, fontweight="bold")
    return _save(fig, path)
