#!/usr/bin/env python3
"""Run from the repo root:  python src/figs/plots.py

"""
import json
import os
import re
import sys

# Run from the repo root, so src/ is not on the path by default and the
# sibling modules in it would not import.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BLUE, GREEN, VERM, PURP = "#0072B2", "#009E73", "#D55E00", "#8C5AA8"
INK, MUTED, GRID = "#1a1a1a", "#5a5a5a", "#d8d8d8"
mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "grid.color": GRID, "grid.linewidth": 0.5, "figure.dpi": 200,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})
R = lambda f: pd.read_csv(f"results/{f}.csv")
os.makedirs("figures", exist_ok=True)   # not in the repo, so may not exist
ITEMS = {i["id"]: i for i in json.load(open("data/items_v2.json"))}
CONDS = ["T0_no_evidence", "T1_explicit", "T2_implicit",
         "T3_counterfact", "T4_conflict"]
CLAB = ["T0\nno event", "T1\nstated", "T2\ninferred",
        "T3\nno rival", "T4\nconflict"]
QM = ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B",
      "Qwen/Qwen3.5-9B", "Qwen/Qwen2.5-7B-Instruct"]
QL = ["Qwen3.5-0.8B", "Qwen3.5-2B", "Qwen3.5-4B", "Qwen3.5-9B", "Qwen2.5-7B"]
GM = ["google/gemma-3-1b-it", "google/gemma-3-4b-it",
      "google/gemma-3-12b-it", "google/gemma-3-27b-it"]
GL = ["Gemma-3-1b", "Gemma-3-4b", "Gemma-3-12b", "Gemma-3-27b"]


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * max(0, c - h), 100 * min(1, c + h)


def acc(df, model, cond, parsed=True):
    s = df[(df.model == model) & (df.condition == cond)]
    if parsed and "parsed" in s:
        s = s[s.parsed]
    return wilson(int(s.is_correct.sum()), len(s))


def chance(ax, y=50):
    ax.axhline(y, color=MUTED, lw=0.7, ls=(0, (4, 3)), zorder=1)


def panel(ax, letter, title):
    ax.text(-0.155, 1.07, letter, transform=ax.transAxes, fontsize=10.5,
            fontweight="bold", va="bottom", ha="left")
    ax.set_title(title, loc="left", pad=5, fontsize=8.0)
    ax.tick_params(labelsize=6.4)
    ax.spines[["top", "right"]].set_visible(False)


# ===================================================================== fig 2
def fig_main():
    """Figure 2, grouped bars with every value labelled.

    Points with offset markers made the reader work out which series was
    which; bars with the number printed on them do not. Shades within a
    family encode scale, so the ordering is readable without the legend.
    """
    probe = R("probe_chat_prefill__phase1_main")
    inc = R("probe_chat_prefill__phase6_incidental")
    qgen = pd.concat([R("generate_thinkoff__phase1_gen_ladder"),
                      R("generate_thinkoff__phase1_gen_rescored")])
    gen = pd.concat([R("generate_thinkoff__phase5_gen_req_rescored"),
                     R("generate_thinkoff__phase5xl_req")])
    ginc = pd.concat([R("generate_thinkoff__phase5_gen_inc"),
                      R("generate_thinkoff__phase5xl_inc")])
    for d in (inc, ginc):
        d["base"] = d.id.str.replace("_finc", "", regex=False)

    fig = plt.figure(figsize=(7.0, 5.4))
    gs = fig.add_gridspec(2, 2, hspace=0.75, wspace=0.30,
                          top=0.88, bottom=0.09, left=0.085, right=0.985)

    def bars(ax, models, labels, data, base_col, letter, title):
        """Grouped bars, one group per condition, with 95% intervals."""
        shades = [0.34 + 0.22 * i for i in range(len(models))]
        w = 0.78 / len(models)
        for j, (m, lab) in enumerate(zip(models, labels)):
            xs, ys, lo, hi = [], [], [], []
            for i, c in enumerate(CONDS):
                acc_, l, h = acc(data, m, c)
                xs.append(i - 0.39 + w * (j + 0.5))
                ys.append(acc_); lo.append(acc_ - l); hi.append(h - acc_)
            ax.bar(xs, ys, width=w * 0.90, color=base_col, alpha=shades[j],
                   zorder=3, label=lab, linewidth=0)
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=INK,
                        elinewidth=0.6, capsize=0, zorder=4)
            for x, y, h in zip(xs, ys, hi):
                ax.text(x, y + h + 2.0, f"{y:.0f}", fontsize=4.7,
                        ha="center", va="bottom", color=INK, zorder=5)
        chance(ax)
        ax.set_xticks(range(5)); ax.set_xticklabels(CLAB, fontsize=6.4)
        # T2 is the condition the paper turns on; mark it on the axis rather
        # than with a floating label that would sit on the T3 bars.
        ax.get_xticklabels()[2].set_color(VERM)
        ax.get_xticklabels()[2].set_fontweight("bold")
        ax.set_ylim(0, 122); ax.set_yticks([0, 25, 50, 75, 100])
        ax.tick_params(labelsize=6.4)
        ax.yaxis.grid(True, zorder=0)
        ax.set_title(title, fontsize=8.0, pad=5)
        ax.text(-0.155, 1.06, letter, transform=ax.transAxes, fontsize=10.5,
                fontweight="bold", va="bottom")
        ax.spines[["top", "right"]].set_visible(False)

    QG, QGL = QM[1:], QL[1:]
    ax = fig.add_subplot(gs[0, 0])
    bars(ax, QG, QGL, qgen, BLUE, "a", "Qwen, free-text generation")
    ax.set_ylabel("accuracy (\%)", fontsize=7.5)
    h1, l1 = ax.get_legend_handles_labels()
    ax2 = fig.add_subplot(gs[0, 1])
    bars(ax2, GM, GL, gen, GREEN, "b", "Gemma, free-text generation")
    h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="upper center", ncol=8, frameon=False,
               fontsize=6.4, handlelength=1.1, handletextpad=0.4,
               columnspacing=1.0, bbox_to_anchor=(0.53, 1.005))
    ax.text(0.0, 114, "every T2 interval lies below chance",
            fontsize=6.4, color=VERM, ha="left", va="center")

    # ---- (c) framing, paired slopes with both ends labelled
    ax = fig.add_subplot(gs[1, 0])
    ax.text(-0.155, 1.06, "c", transform=ax.transAxes, fontsize=10.5,
            fontweight="bold", va="bottom")
    ax.set_title("T2 under two framings of the stated preference",
                 fontsize=8.0, pad=5)
    rows = ([(l, probe, inc, m, BLUE) for l, m in zip(QL, QM)]
            + [(l, gen, ginc, m, GREEN) for l, m in zip(GL, GM)])
    ends = []
    for lab, A, B, m, col in rows:
        a0 = acc(A, m, "T2_implicit")[0]; b0 = acc(B, m, "T2_implicit")[0]
        ax.plot([0, 1], [a0, b0], "-o", ms=3, lw=1.0, color=col, alpha=0.85,
                zorder=3)
        ends.append([b0, lab, col])
    ends.sort(key=lambda r: r[0])
    for i in range(1, len(ends)):
        if ends[i][0] - ends[i - 1][0] < 5.0:
            ends[i][0] = ends[i - 1][0] + 5.0
    for y, lab, col in ends:
        ax.annotate(lab, (1.06, y), fontsize=6.0, va="center", color=col)
    chance(ax)
    ax.set_xlim(-0.15, 1.72); ax.set_xticks([0, 1])
    ax.set_xticklabels(["requirement", "incidental"], fontsize=6.8)
    ax.set_ylim(0, 104); ax.set_ylabel("accuracy (\%)", fontsize=7.5)
    ax.tick_params(labelsize=6.4); ax.yaxis.grid(True, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    # ---- (d) interventions, bars with the end value on each
    ax = fig.add_subplot(gs[1, 1])
    ax.text(-0.155, 1.06, "d", transform=ax.transAxes, fontsize=10.5,
            fontweight="bold", va="bottom")
    ax.set_title("What moves T2, and what does not", fontsize=8.0, pad=5)
    bars_d = [("30$\\times$ tokens", 3.8, 5.5, MUTED),
              ("step by step", 9.2, 7.5, MUTED),
              ("user's voice", 9.2, 24.5, VERM),
              ("name the event", 9.2, 37.5, VERM),
              ("casual wording", 9.2, 45.0, VERM),
              ("natural chat", 9.2, 61.3, VERM),
              ("no rival option", 9.2, 74.1, VERM)]
    for i, (lab, b0, b1, col) in enumerate(bars_d):
        ax.bar(i, b1, width=0.62, color=col, alpha=0.90, zorder=3,
               linewidth=0)
        ax.plot([i - 0.31, i + 0.31], [b0, b0], color=INK, lw=0.8, zorder=5)
        ax.text(i, b1 + 2.5, f"{b1:.1f}", fontsize=5.6, ha="center",
                va="bottom", color=INK, zorder=6,
                bbox=dict(fc="white", ec="none", pad=0.5))
        ax.text(i, b1 / 2, f"{b1-b0:+.1f}", fontsize=5.6, ha="center",
                va="center", color="white", fontweight="bold", zorder=5)
    chance(ax)
    ax.set_xticks(range(len(bars_d)))
    ax.set_xticklabels([b[0] for b in bars_d], fontsize=6.0, rotation=38,
                       ha="right")
    ax.set_ylim(0, 92); ax.set_ylabel("T2 accuracy (\%)", fontsize=7.5)
    ax.tick_params(labelsize=6.4); ax.yaxis.grid(True, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.5, 84, "baseline shown as a rule on each bar", fontsize=5.8,
            color=INK)

    fig.savefig("figures/fig2_main.pdf"); fig.savefig("figures/fig2_main.png", dpi=180)
    print("wrote figures/fig2_main.pdf + png")


# ===================================================================== fig 3
def fig_controls():
    t = R("probe_chat_prefill__phase2_turns")
    p = R("probe_chat_prefill__phase2_pos")
    tf = R("generate_thinkon__phase3_t2_full_rescored")
    probe = R("probe_chat_prefill__phase1_main")
    keep = ["Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B", "Qwen/Qwen2.5-7B-Instruct"]
    kl = ["Qwen3.5-4B", "Qwen3.5-9B", "Qwen2.5-7B"]
    C3 = [BLUE, GREEN, VERM]

    fig = plt.figure(figsize=(7.0, 4.4))
    gs = fig.add_gridspec(2, 3, hspace=0.6, wspace=0.35)

    # (a) length
    ax = fig.add_subplot(gs[0, 0])
    panel(ax, "a", "Conversation length")
    for c, col, ls in zip(["T0_no_evidence", "T1_explicit", "T2_implicit"],
                          C3, ["-", "-", "-"]):
        ys = [100 * t[(t.condition == c) & (t.n_turns == n)
                      & t.model.isin(keep)].is_correct.mean()
              for n in (4, 10, 20, 50)]
        ax.plot([4, 10, 20, 50], ys, "-o", ms=3, lw=1.2, color=col,
                label=c.split("_")[0], zorder=3)
        # only T2 is labelled. T0 and T1 sit on top of each other at the
        # ceiling, and three labels per endpoint were unreadable.
        if c == "T2_implicit":
            for xx, yy in ((4, ys[0]), (50, ys[-1])):
                ax.annotate(f"{yy:.1f}", (xx, yy), (0, 8),
                            textcoords="offset points", fontsize=6.0,
                            ha="center", color=col, zorder=5)
    chance(ax); ax.set_xscale("log"); ax.set_xticks([4, 10, 20, 50])
    ax.set_xticklabels([4, 10, 20, 50]); ax.set_xlabel("turns")
    ax.set_ylabel("accuracy (%)"); ax.set_ylim(-4, 104)
    ax.yaxis.grid(True, zorder=0)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(0.02, 0.60),
              handletextpad=0.4, labelspacing=0.2, fontsize=6.4)
    ax.text(4.4, 86, "T0 and T1 hold at ceiling", fontsize=5.8, color=INK)

    # (b) distance
    ax = fig.add_subplot(gs[0, 1])
    panel(ax, "b", "Event distance, length fixed")
    for c, col in zip(["T0_no_evidence", "T1_explicit", "T2_implicit"], C3):
        ys = [100 * p[(p.condition == c) & (p.event_pos == q)
                      & p.model.isin(keep)].is_correct.mean()
              for q in (0.3, 0.6, 0.9)]
        ax.plot([14, 8, 2], ys, "-o", ms=3, lw=1.2, color=col, zorder=3)
        if c == "T2_implicit":
            for xx, yy in ((14, ys[0]), (2, ys[-1])):
                ax.annotate(f"{yy:.1f}", (xx, yy), (0, 8),
                            textcoords="offset points", fontsize=6.0,
                            ha="center", color=col, zorder=5)
    chance(ax); ax.set_xlabel("turns before the question")
    ax.set_xticks([2, 8, 14]); ax.invert_xaxis()
    ax.set_ylim(-4, 104); ax.yaxis.grid(True, zorder=0)

    # (c) confidence
    ax = fig.add_subplot(gs[0, 2])
    panel(ax, "c", "Probability on the correct option")
    for c, col in zip(["T0_no_evidence", "T1_explicit", "T2_implicit"], C3):
        v = probe[(probe.condition == c)
                  & (probe.model == "Qwen/Qwen3.5-9B")].p_correct.values
        xs = np.sort(v); ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(xs, ys, lw=1.4, color=col, zorder=3)
    ax.axvline(0.5, color=MUTED, lw=0.7, ls=(0, (4, 3)))
    ax.set_xlabel("p(correct)"); ax.set_ylabel("cumulative fraction")
    ax.set_xlim(0, 1); ax.grid(True, zorder=0)
    ax.legend(handles=[Line2D([], [], color=c, lw=1.4, label=l)
                       for c, l in zip(C3, ["T0", "T1", "T2"])],
              loc="center left", bbox_to_anchor=(0.02, 0.62), frameon=False,
              handletextpad=0.5, labelspacing=0.2, fontsize=6.8)

    # (d) where the trace names its answer
    ax = fig.add_subplot(gs[1, 0])
    panel(ax, "d", "Where the trace names an option")
    # Only tags whose content RESOLVES to a displayed option. The first tag in
    # a trace is usually the model restating the required output format
    # (``<answer>OPTION</answer>``), which sits at ~4% of the text and is not a
    # choice; counting it put the median at 4% instead of 24%.
    import sys as _s
    _s.path.insert(0, ".")
    from parsing import _letter_ref, _match
    TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S | re.I)
    pos = []
    for r in tf[tf.parsed].itertuples():
        txt = str(r.raw); L = len(txt); opts = [r.opt_a, r.opt_b]
        for m in TAG.finditer(txt):
            if _match(m.group(1), opts) or _letter_ref(m.group(1), opts):
                pos.append(m.start() / L); break
    ax.hist(pos, bins=20, color=BLUE, alpha=0.85, zorder=3)
    ax.axvline(np.median(pos), color=VERM, lw=1.2)
    ax.annotate(f"median {100*np.median(pos):.0f}%",
                (np.median(pos) + 0.03, ax.get_ylim()[1] * 0.85),
                fontsize=7, color=VERM)
    ax.set_xlabel("position in trace"); ax.set_ylabel("traces")
    ax.set_xlim(0, 1); ax.yaxis.grid(True, zorder=0)
    print(f"    (d) n={len(pos)} traces, median commit "
          f"{100*np.median(pos):.0f}% of trace")

    # (e) deliberation, three arms
    ax = fig.add_subplot(gs[1, 1])
    panel(ax, "e", "Deliberation, Gemma-3-12b")
    vals = [9.2, 7.5, 37.5]; labs = ["none", "neutral", "directed"]
    cols = [MUTED, MUTED, VERM]
    for i, (v, col) in enumerate(zip(vals, cols)):
        ax.bar(i, v, width=0.55, color=col, alpha=0.9, zorder=3, linewidth=0)
        ax.annotate(f"{v:.1f}", (i, v + 2.5), ha="center", fontsize=6.6,
                    color=INK, zorder=5)
        if i:
            ax.annotate(f"{v-vals[0]:+.1f}", (i, v / 2), ha="center",
                        va="center", fontsize=6.0, color="white",
                        fontweight="bold", zorder=5)
    chance(ax); ax.set_xticks(range(3)); ax.set_xticklabels(labs)
    ax.set_ylabel("T2 accuracy (%)"); ax.set_ylim(0, 104)
    ax.yaxis.grid(True, zorder=0)

    # (f) does reasoning predict the answer
    ax = fig.add_subplot(gs[1, 2])
    panel(ax, "f", "Reasoning does not predict the answer")
    j = R("generate_thinkon__phase3_t2_full_judged")
    both = j[j.derives_consequence & j.links_to_choice]
    rest = j[~(j.derives_consequence & j.links_to_choice)]
    for i, (g, lab) in enumerate([(both, "derives and\nlinks"),
                                  (rest, "neither")]):
        a, lo, hi = wilson(int(g.model_answered_correct.sum()), len(g))
        ax.errorbar(i, a, yerr=[[a - lo], [hi - a]], fmt="o", ms=5, lw=1.0,
                    color=BLUE, zorder=3)
        ax.annotate(f"{a:.1f}%\nn={len(g)}", (i + 0.12, a), fontsize=7,
                    va="center", color=INK)
    ax.set_xlim(-0.4, 1.6); ax.set_xticks([0, 1])
    ax.set_xticklabels(["derives and\nlinks", "neither"])
    ax.set_ylim(0, 30); ax.set_ylabel("accuracy (%)")
    ax.yaxis.grid(True, zorder=0)
    fig.savefig("figures/fig3_controls.pdf"); fig.savefig("figures/fig3_controls.png", dpi=170)
    print("wrote figures/fig3_controls.pdf + png")


# ===================================================================== fig 1
def fig_teaser():
    """Figure 1, drawn as a diagram rather than as plots.

    Geometry is set from measured text widths. The widest line in the right
    panel ends at x=0.844, so the tinted blocks stop at 0.966 rather than
    running to the frame edge, and the slack is spent on the gutter between
    the two panels and on the outer margins instead.
    """
    from matplotlib.patches import FancyBboxPatch
    fig = plt.figure(figsize=(7.2, 3.55))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    TINT_B, TINT_V, TINT_G, TINT_N = "#eaf1f7", "#fdf0e8", "#e9f5f1", "#f2f4f5"
    LX0, LX1 = 0.018, 0.400          # left frame
    RX0, RX1 = 0.462, 0.982          # right frame, 0.062 gutter between
    LPAD, RPAD = 0.018, 0.018        # inner padding

    def frame(x0, y0, x1, y1, col):
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                     boxstyle="round,pad=0.006,rounding_size=0.022",
                     fc="white", ec=col, lw=1.9, zorder=1))

    def head(x, txt, sub, ulen):
        ax.text(x, 0.905, txt, fontsize=9.2, fontweight="bold", color=INK)
        ax.plot([x, x + ulen], [0.884, 0.884], color=INK, lw=1.0, zorder=3)
        ax.text(x, 0.845, sub, fontsize=7.0, color=INK, style="italic")

    def bubble(x0, y0, x1, y1, txt, fc, ec, fs=6.6, style="italic", tc=INK):
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                     boxstyle="round,pad=0.004,rounding_size=0.014",
                     fc=fc, ec=ec, lw=0.9, zorder=2))
        ax.text(x0 + 0.013, (y0 + y1) / 2, txt, fontsize=fs, va="center",
                style=style, color=tc, zorder=3)

    # ================= LEFT: the problem =================
    frame(LX0, 0.03, LX1, 0.965, BLUE)
    head(LX0 + LPAD, "The problem", "a preference the user never restates",
         0.096)
    bx0, bx1 = LX0 + LPAD, LX1 - LPAD

    bubble(bx0, 0.740, bx1, 0.818,
           "user: For my way of getting around,\nwhat works for me is transit.",
           TINT_B, BLUE)
    ax.text(bx0 + 0.008, 0.707, "turn 1, the stated preference", fontsize=6.0,
            color=BLUE)
    
    bubble(bx0, 0.594, bx1, 0.680,
           "user: The strike has taken out almost the\nwhole timetable "
           "for the foreseeable future.",
           TINT_V, VERM)
    ax.text(bx0 + 0.008, 0.561, "turn 2, an event that removes transit entirely",
            fontsize=6.0, color=VERM)
    bubble(bx0, 0.482, bx1, 0.536, "user: [ two unrelated turns ]",
           TINT_N, GRID, fs=6.4)
    bubble(bx0, 0.368, bx1, 0.446,
           "Which should you suggest?\nA. cycling      B. transit",
           "#f0f3f6", INK, fs=6.6, style="normal")

    ax.plot([(bx0 + bx1) / 2, (bx0 + bx1) / 2], [0.347, 0.325], color=INK,
            lw=0.9, zorder=2)
    ax.plot([(bx0 + bx1) / 2], [0.322], marker="v", ms=4, color=INK, zorder=2)

    for y, num, lab, col in ((0.300, "100%", "when the user states it (T1)",
                              BLUE),
                             (0.232, "3.8%", "when an event implies it (T2)",
                              VERM)):
        ax.text(bx0 + 0.098, y, num, fontsize=14.5, fontweight="bold",
                color=col, va="center", ha="right", zorder=3)
        ax.text(bx0 + 0.112, y, lab, fontsize=6.6, color=INK, va="center",
                zorder=3)
    ax.text(bx0 + 0.098, 0.163, "93.3%", fontsize=10.5, fontweight="bold",
            color=GREEN, va="center", ha="right", zorder=3)
    ax.text(bx0 + 0.112, 0.163, "a human annotator, same items",
            fontsize=6.4, color=INK, va="center", zorder=3)
    ax.text(bx0, 0.076,
            "Qwen3.5-9B. Across nine models T2 runs from 2.2% to 13.3%,\n"
            "every interval below the 50% chance baseline. The human\n"
            "figure covers items whose premise the annotator endorses.",
            fontsize=6.0, color=INK, va="center")

    # ================= RIGHT: the findings =================
    frame(RX0, 0.03, RX1, 0.965, VERM)
    head(RX0 + RPAD, "What we find",
         "the model derives the consequence, then sets it aside", 0.112)

    def block(y0, y1, n, title, tint, lines):
        x0, x1 = RX0 + RPAD, RX1 - RPAD
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                     boxstyle="round,pad=0.004,rounding_size=0.014",
                     fc=tint, ec="none", zorder=2))
        ax.add_patch(FancyBboxPatch((x0 + 0.0005, y0), 0.006, y1 - y0,
                     boxstyle="square,pad=0", fc=INK, ec="none", zorder=3))
        ax.text(x0 + 0.018, y1 - 0.038, f"{n}.  {title}", fontsize=7.8,
                fontweight="bold", color=INK, va="center", zorder=3)
        yy = y1 - 0.090
        for big, col, rest in lines:
            ax.text(x0 + 0.076, yy, big, fontsize=9.0, fontweight="bold",
                    color=col, va="center", ha="right", zorder=3)
            ax.text(x0 + 0.088, yy, rest, fontsize=7.0, color=INK,
                    va="center", zorder=3)
            yy -= 0.050

    block(0.587, 0.815, 1, "Not memory, and not more computation", TINT_N,
          [("2.5%", VERM, "T2 in a 50-turn conversation, while T0 and T1 hold"),
           ("+1.7", INK, "points from a reasoning mode spending 36x the tokens"),
           ("9.4%", INK, "correct when the trace derives it, 8.9% when it does not")])
    block(0.324, 0.552, 2, "It is the weight on the stated preference", TINT_G,
          [("66.7%", GREEN, "once that preference is taken off the ballot (T3)"),
           ("+27.5", GREEN, "points from wording turn 1 as a habit, not a rule"),
           ("24%", INK, "of the way through the trace, the answer is already fixed")])
    block(0.061, 0.289, 3, "Part of it is the exam framing, and it cuts both ways",
          TINT_V,
          [("40.0%", VERM, "T2 once the same task is asked as ordinary chat"),
           ("61.3%", VERM, "for Gemma-3-12b, the first cell to clear chance"),
           ("10.3%", BLUE, "of T1 answers then override the user, against 0.0% before")])

    fig.savefig("figures/fig1_teaser.pdf"); fig.savefig("figures/fig1_teaser.png", dpi=180)
    print("wrote figures/fig1_teaser.pdf + png")


# ================================================================ fig tradeoff

ROSE, SAND = "#B4656F", "#E0A458"


def fig_tradeoff():
    """The two directions of failure, one panel per model.
    """
    QMn, GMn = "Qwen/Qwen3.5-9B", "google/gemma-3-12b-it"
    exam = {QMn: R("generate_thinkoff__phase1_gen_rescored"),
            GMn: R("generate_thinkoff__phase5_gen_req_rescored")}
    fp = R("generate_thinkoff__phase7_ask_firstperson")
    nat = {QMn: R("generate_thinkoff__phase7_natural_qwen_judged"),
           GMn: R("generate_thinkoff__phase7_natural_gemma_judged")}

    def err(model, kind, cond):
        """Wrong share over committed replies, then the same counting a
        declined reply as wrong."""
        if kind == "exam":
            d = exam[model][exam[model].model == model]
        elif kind == "fp":
            d = fp[fp.model == model]
        else:
            d = nat[model]
        s = d[d.condition == cond]
        if kind == "nat":
            com = s[s.judge_pick != "no_recommendation"]
            wrong = int((~com.judge_correct).sum())
        else:
            com = s[s.parsed] if "parsed" in s else s
            wrong = int((~com.is_correct).sum())
        dec = len(s) - len(com)
        return 100 * wrong / len(com), 100 * (wrong + dec) / len(s)

    FRAMES = (("exam", "exam\nquestion"), ("fp", "first-person\nask"),
              ("nat", "natural\nchat"))
    SERIES = (("T2_implicit", ROSE, "o",
               "keeps the option the event ruled out"),
              ("T1_explicit", SAND, "s",
               "does not follow what the user just asked"))
    x = np.arange(3)

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.85), sharey=True,
                             gridspec_kw=dict(wspace=0.10))
    for ax, (m, mlab) in zip(axes, ((QMn, "Qwen3.5-9B"),
                                    (GMn, "Gemma-3-12b"))):
        ax.set_title(mlab, loc="left", pad=6, fontsize=8.5,
                     fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", zorder=0); ax.set_axisbelow(True)
        for cond, col, mk, _ in SERIES:
            v = [err(m, k, cond) for k, _ in FRAMES]
            lo = [a for a, _ in v]
            hi = [b for _, b in v]
            ax.plot(x, hi, ls=(0, (3, 2)), lw=1.0, color=col, zorder=2)
            ax.plot(x, lo, "-", lw=1.6, color=col, zorder=3)
            ax.plot(x, lo, mk, ms=5.0, color=col, mec="white", mew=0.8,
                    zorder=4)
            for xi, (a, b) in zip(x, v):
                # The committed figure sits beside its marker. The dashed
                # figure is only written where it differs.
                dy = 6.5 if cond == "T1_explicit" else -8.5
                ax.text(xi, a + dy, f"{a:.0f}", fontsize=6.8, color=INK,
                        ha="center", va="center")
                if b - a > 2:
                    ax.text(xi + 0.07, b + 5.5, f"{b:.0f}", fontsize=6.6,
                            color=col, ha="center", va="center")
        ax.set_xticks(x)
        ax.set_xticklabels([l for _, l in FRAMES], fontsize=7.0)
        ax.set_xlim(-0.35, 2.35); ax.set_ylim(-6, 106)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.tick_params(axis="x", length=0, pad=4)
    axes[0].set_ylabel("share of answers in error (%)", labelpad=3)

    handles = [Line2D([], [], color=c, lw=1.6, marker=mk, ms=5.0,
                      mec="white", mew=0.8, label=lab)
               for _, c, mk, lab in SERIES]
    handles.append(Line2D([], [], color=INK, lw=1.0, ls=(0, (3, 2)),
                          label="a declined reply counted as an error"))
    fig.legend(handles=handles, loc="lower center", ncol=1, frameon=False,
               fontsize=7.2, bbox_to_anchor=(0.55, -0.20),
               handlelength=2.0, handletextpad=0.6, labelspacing=0.35)

    fig.savefig("figures/fig_tradeoff.pdf")
    fig.savefig("figures/fig_tradeoff.png", dpi=180)
    print("wrote figures/fig_tradeoff.pdf + png")


if __name__ == "__main__":
    fig_teaser(); fig_main(); fig_controls(); fig_tradeoff()
