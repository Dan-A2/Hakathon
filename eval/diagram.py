"""The pipeline diagram for slides and the report.

    python -m eval.diagram            # -> art/report/pipeline.png
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from common import config as C  # noqa: E402

INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"
BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#c3c2b7"
W, H = 2.55, 1.5


def _box(ax, x, y, title, body, edge, badge=None, fill="white"):
    ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.02,rounding_size=0.12", linewidth=1.8,
                                edgecolor=edge, facecolor=fill, zorder=2))
    ax.text(x + 0.12, y + H - 0.17, title, fontsize=10.5, fontweight="bold", color=INK, va="top", zorder=3)
    ax.text(x + 0.12, y + H - 0.52, textwrap.fill(body, 29), fontsize=8.4, color=INK2, va="top", zorder=3, linespacing=1.25)
    if badge:
        ax.text(x + W / 2, y - 0.12, badge, fontsize=8, color=edge, ha="center", va="top", fontweight="bold", zorder=3)


def _arrow(ax, x0, y0, x1, y1, color=MUTED, label=None, lx=None, ly=None, ha="center"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=1,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8, mutation_scale=16, shrinkA=0, shrinkB=0))
    if label:
        ax.text(lx if lx is not None else (x0 + x1) / 2, ly if ly is not None else (y0 + y1) / 2, label, fontsize=8.6,
                color=color, ha=ha, va="center", fontweight="bold", zorder=3,
                bbox=dict(boxstyle="round,pad=0.15", facecolor=SURFACE, edgecolor="none"))


def draw(out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(15.5, 7.4))
    fig.patch.set_facecolor(SURFACE)
    ax.set_xlim(0, 15.1)
    ax.set_ylim(-0.2, 7.0)
    ax.axis("off")
    xs = [0.3, 3.3, 6.3, 9.3, 12.3]
    top, bot = 4.8, 1.6

    ax.text(0.3, 6.75, "How the agent handles one claim", fontsize=15, fontweight="bold", color=INK, va="center")
    # top row
    _box(ax, xs[0], top, "1. A claim + evidence", "A scientific claim and the 3 most relevant abstracts a search engine finds for it", GREY)
    _box(ax, xs[1], top, "2. Ask the model twice", "Each answer gives a verdict (supported / refuted / not enough evidence), how sure it is, and its sources", GREY, "2 model calls")
    _box(ax, xs[2], top, "3. Measure its doubt", "Six numbers: how sure it says it is, whether its two answers agree, how far apart they are, ...", GREY, "no model call")
    _box(ax, xs[3], top, "4. Decide", "Answer now, check first, or say \"I don't know\". Learned from thousands of practice cases", BLUE, "no model call")
    _box(ax, xs[4], top, "Final output", "A verdict, the exact sentence it is based on, and an honest probability of being right. Or \"I don't know\"", GREEN, fill="#f2fbf7")
    # bottom row (the check)
    _box(ax, xs[0], bot, "5. Look it up", f"Search the library, open abstracts, read them sentence by sentence (at most {C.DEFAULT_K} lookups)", ORANGE, "1 model call per step")
    _box(ax, xs[1], bot, "6. No quote, no verdict", "A verdict must quote a sentence the agent actually opened. Otherwise: \"not enough evidence\"", ORANGE, "no model call")
    _box(ax, xs[2], bot, "7. Blind reviewer", "A fresh model call sees only the claim and the quote, and says what the quote really shows", ORANGE, "1 model call")
    _box(ax, xs[3], bot, "8. Decide again", "Use the checked verdict, keep the first answer, or say \"I don't know\"", BLUE, "no model call")

    for i in range(3):   # top row arrows
        _arrow(ax, xs[i] + W, top + H / 2, xs[i + 1], top + H / 2)
    _arrow(ax, xs[3] + W, top + H / 2, xs[4], top + H / 2, BLUE, "answer or\n\"I don't know\"", lx=(xs[3] + W + xs[4]) / 2, ly=top + H + 0.32)
    # check path: down from Decide, left, down to Look it up
    yb = 3.7
    ax.plot([xs[3] + W / 2, xs[3] + W / 2, xs[0] + W / 2], [top, yb, yb], color=ORANGE, lw=1.8, zorder=1, solid_capstyle="round")
    _arrow(ax, xs[0] + W / 2, yb, xs[0] + W / 2, bot + H, ORANGE)
    ax.text((xs[0] + xs[3]) / 2 + W / 2, yb + 0.2, "check first", fontsize=9.5, color=ORANGE, ha="center", fontweight="bold")
    for i in range(3):   # bottom row arrows
        _arrow(ax, xs[i] + W, bot + H / 2, xs[i + 1], bot + H / 2, ORANGE)
    _arrow(ax, xs[3] + W, bot + H * 0.75, xs[4] + 0.6, top, BLUE, "send the\nfinal answer", lx=xs[4] + 0.75, ly=2.75, ha="left")
    # behind the scenes
    ax.add_patch(FancyBboxPatch((0.3, -0.05), 14.55, 0.95, boxstyle="round,pad=0.02,rounding_size=0.1", linewidth=1.2,
                                edgecolor=GREY, facecolor="#f3f3f0", linestyle=(0, (4, 3)), zorder=0))
    ax.text(0.5, 0.42, "Behind the scenes:", fontsize=9.5, fontweight="bold", color=INK, va="center")
    ax.text(2.55, 0.42, textwrap.fill("every option is run once for every claim and saved. Learning and every comparison replay those saved runs, so "
                                      "strategies differ only in their decisions, never in luck. The answer key is seen only by the scorer, never by the agent.", 140),
            fontsize=9, color=INK2, va="center")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(draw(C.ART_DIR / "report" / "pipeline.png"))
