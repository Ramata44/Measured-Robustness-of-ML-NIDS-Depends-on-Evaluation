# make_paper_figures.py -- Figures 1-3 of the manuscript, drawn from the values
# below (ten-seed results of the deduplicated run, Model 4, 30 features).
# Output: fig1.pdf, fig2.pdf, fig3.pdf in figures_paper/ (next to this file, or
# in the notebook's working folder when pasted into a cell).
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DPI = 1600
try:                                   # run as a script
    BASE = os.path.dirname(os.path.abspath(__file__))
except NameError:                      # run in a notebook cell
    BASE = os.getcwd()
OUT = os.path.join(BASE, "figures_paper")
os.makedirs(OUT, exist_ok=True)

DATASETS = [("cic", "CSE-CIC-IDS2018"), ("unsw", "UNSW-NB15")]
ACTS = ["RELU", "ELU", "TANH"]
EPS = [0.1, 0.3, 0.5]

BLUE, LIGHT_BLUE = "#3b6fb0", "#6fa3d8"
RED, ORANGE = "#c0504d", "#e2915f"

FIG1 = {   # mean, SD over ten seeds; order ReLU, ELU, Tanh
    "cic": {
        "PGD": ([0.9991, 1.0, 0.9302], [0.0012, 0.0002, 0.1393]),
        "BIM": ([0.9992, 1.0, 0.9474], [0.0008, 0.0002, 0.1649]),
        "HSJ": ([0.9948, 1.0, 1.0], [0.0111, 0.0002, 0.0]),
        "SPSA": ([0.9056, 0.7819, 0.9468], [0.1807, 0.3392, 0.1647]),
    },
    "unsw": {
        "PGD": ([0.8422, 0.8498, 0.934], [0.0306, 0.0179, 0.0126]),
        "BIM": ([0.8409, 0.8498, 0.9342], [0.0342, 0.0182, 0.0121]),
        "HSJ": ([0.8142, 0.8425, 0.9054], [0.0382, 0.023, 0.0179]),
        "SPSA": ([0.8292, 0.8363, 0.9272], [0.0355, 0.022, 0.0135]),
    },
}

FIG2 = {   # (dataset, detector, level): (means, SDs) at eps 0.1, 0.3, 0.5
    ("cic", "standard", 2): ([0.958, 0.8108, 0.5454], [0.0124, 0.0601, 0.156]),
    ("cic", "standard", 0): ([0.4064, 0.0003, 0.0], [0.1467, 0.0002, 0.0]),
    ("cic", "pgd_at", 2): ([0.9888, 0.9864, 0.9841], [0.0004, 0.0055, 0.0072]),
    ("cic", "pgd_at", 0): ([0.9855, 0.9424, 0.3928], [0.0062, 0.0118, 0.1118]),
    ("cic", "ca_at", 2): ([0.9883, 0.9848, 0.9792], [0.0056, 0.008, 0.0118]),
    ("cic", "ca_at", 0): ([0.4893, 0.0218, 0.0001], [0.1956, 0.045, 0.0003]),
    ("unsw", "standard", 2): ([0.9547, 0.8951, 0.8137], [0.0147, 0.0447, 0.0718]),
    ("unsw", "standard", 0): ([0.5149, 0.1144, 0.0384], [0.0391, 0.0137, 0.005]),
    ("unsw", "pgd_at", 2): ([0.9753, 0.975, 0.9741], [0.0016, 0.0016, 0.002]),
    ("unsw", "pgd_at", 0): ([0.9587, 0.8879, 0.6543], [0.0027, 0.0091, 0.0183]),
    ("unsw", "ca_at", 2): ([0.9729, 0.9711, 0.9696], [0.0163, 0.0169, 0.0172]),
    ("unsw", "ca_at", 0): ([0.4986, 0.125, 0.0386], [0.0392, 0.0201, 0.0037]),
}

FIG3 = {   # mean |attribution| over ten seeds: feature: (ReLU, ELU, Tanh)
    "cic": {
        "Fwd Seg Size Min": (1.7532, 3.0854, 1.6773),
        "Init Fwd Win Bytes": (1.5389, 1.9395, 1.2059),
        "Bwd IAT Total": (0.5198, 1.2670, 0.5163),
        "Bwd Packet Length Max": (0.4690, 2.9056, 2.2460),
        "Init Bwd Win Bytes": (0.4635, 0.7474, 0.4801),
        "Bwd Packet Length Std": (0.4531, 3.2600, 2.1715),
        "RST Flag Count": (0.4224, 1.0827, 0.3901),
        "Fwd RST Flags": (0.3885, 1.2527, 0.4976),
        "Flow IAT Std": (0.1749, 1.6403, 0.9458),
        "Fwd IAT Max": (0.1531, 0.9784, 0.5909),
    },
    "unsw": {
        "dload": (1.0444, 1.5852, 0.5096),
        "ackdat": (0.9443, 0.9374, 0.8025),
        "tcprtt": (0.6594, 0.5878, 0.4308),
        "dmean": (0.4373, 0.4653, 0.2737),
        "synack": (0.2756, 0.2790, 0.2506),
        "service_smtp": (0.2132, 0.3014, 0.1844),
        "service_-": (0.1913, 0.0739, 0.0892),
        "proto_tcp": (0.1737, 0.1794, 0.4843),
        "swin": (0.1607, 0.2207, 0.3421),
        "state_FIN": (0.1600, 0.2317, 0.4020),
    },
}


def err(mean, sd):
    """Mean +- SD, truncated to [0, 1]."""
    return [np.minimum(sd, mean), np.minimum(sd, 1 - np.asarray(mean))]


def save(fig, name):
    path = os.path.join(OUT, name + ".pdf")
    fig.savefig(path, format="pdf", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("written:", path)


# Figure 1: mean ASR at eps = 0.3 (C0) of the four primary attacks
bars = [("PGD", "PGD (WB)", BLUE), ("BIM", "BIM (WB)", LIGHT_BLUE),
        ("HSJ", "HSJ (BB)", RED), ("SPSA", "SPSA (BB)", ORANGE)]
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
x, w = np.arange(3), 0.2
for ax, (ds, title) in zip(axes, DATASETS):
    for i, (key, label, c) in enumerate(bars):
        mean, sd = map(np.asarray, FIG1[ds][key])
        ax.bar(x + (i - 1.5) * w, mean, w, yerr=err(mean, sd), capsize=3,
               color=c, label=label, error_kw=dict(lw=1))
    ax.set_xticks(x)
    ax.set_xticklabels(ACTS)
    ax.set_title(title)
    ax.set_xlabel("Activation")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
axes[0].set_ylabel(r"Mean ASR at $\epsilon = 0.3$")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.93))
save(fig, "fig1")

# Figure 2: robust detection TPR x (1 - ASR); solid C2, dashed C0
SERIES = [("standard", "Standard", BLUE), ("pgd_at", "PGD-AT", RED),
          ("ca_at", "SS-AT", ORANGE)]
LEVELS = [(2, "-", "source-side"), (0, "--", "unconstrained")]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for ax, (ds, title) in zip(axes, DATASETS):
    for key, _, c in SERIES:
        for lv, ls, _ in LEVELS:
            mean, sd = map(np.asarray, FIG2[(ds, key, lv)])
            ax.errorbar(EPS, mean, yerr=err(mean, sd), color=c, ls=ls,
                        marker="o", ms=4, capsize=3, lw=1.5)
    ax.set_title(title)
    ax.set_xticks(EPS)
    ax.set_ylim(-0.02, 1.03)
    ax.set_xlabel(r"Attack budget $\epsilon$")
    ax.grid(alpha=0.3)
axes[0].set_ylabel(r"Robust detection, TPR $\times$ (1 $-$ ASR)")
handles = [Line2D([], [], color=c, ls=ls, marker="o", ms=4, lw=1.5)
           for _, _, c in SERIES for _, ls, _ in LEVELS]
labels = [f"{lab}, {nm}" for _, lab, _ in SERIES for _, _, nm in LEVELS]
fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.86))
save(fig, "fig2")

# Figure 3: top-8 features by mean |attribution|, one row per ranking activation
ABBREV = [(" Packet Length ", " PktLen "), ("Seg Size", "Seg Sz"),
          ("Flag Count", "Flag Cnt"), ("Win Bytes", "Win B")]
fig, axes = plt.subplots(3, 2, figsize=(11, 12.6))
for r, ref in enumerate(ACTS):
    for ax, (ds, title) in zip(axes[r], DATASETS):
        feats = sorted(FIG3[ds], key=lambda f: -FIG3[ds][f][r])[:8]
        x = np.arange(8)
        for j, (a, c) in enumerate(zip(ACTS, [BLUE, RED, ORANGE])):
            ax.bar(x + (j - 1) * 0.8 / 3, [FIG3[ds][f][j] for f in feats],
                   0.8 / 3, label=a, color=c)
        ticks = []
        for f in feats:
            for old, new in ABBREV:
                f = f.replace(old, new)
            ticks.append(f)
        ax.set_xticks(x)
        ax.set_xticklabels(ticks, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Mean |attribution|")
        ax.set_title(f"{title} (ranked by {ref})")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
fig.tight_layout()
save(fig, "fig3")
