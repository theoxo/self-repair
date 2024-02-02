import json
import argparse
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sys

def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", required=True, type=str,
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
    )
    parser.add_argument(
        '--mode',
        required=True,
        type=str,
        choices=['pass@t', 'pass@k', 'pass@k-simple']
    )
    return parser

args = build_args().parse_args()

fig = plt.figure(figsize=(4, 3))
ax = plt.gca()

ax.set_ylim(0.0, 1.0)
if args.mode == 'pass@t':
    ax.set_ylabel("Mean pass rate")
    ax.set_xlabel("Mean number of tokens generated")
    ax.set_xlim(0.0, 10000)
if args.mode == 'pass@k':
    ax.set_ylabel("Mean pass rate")
    ax.set_xlabel("Mean number of samples")
    ax.set_xlim(0.0, 50)
if args.mode == 'pass@k-simple':
    ax.set_ylabel("Mean pass rate")
    ax.set_xlabel(r"Number of programs sampled")
    ax.set_xlim(0.0, 50)
ax.grid(True)

with open(args.input, "r") as f:
    krns = json.load(f)

# marker based on n
# ns = [1, 3, 5, 10]
markers = {1: "^", 3: "<", 5: "v", 10: ">"}

red = "#B94700"
yellow = "#F0B323"
green = "#49C5B1"
blue = "#009CDE"
purple = "#151F6D"
dark_gray = "#515555"

# color based on k
# ks = [1, 2, 5, 10, 25]
colors = {1: red, 2: yellow, 5: green, 10: blue, 25: purple}

baseline_krns = [f'({k}, 0)' for k in range(1, 51)]
baseline_ys = [krns[krn][0][0] for krn in baseline_krns]
baseline_ts = [krns[krn][1][0] for krn in baseline_krns]
baseline_ss = [krns[krn][2][0] for krn in baseline_krns]
baseline_xs = list(range(1, 51))
y_errs = [krns[krn][0][1] for krn in baseline_krns]
baseline_handle = ax.plot(
    baseline_ts if args.mode == 'pass@t' else baseline_ss if args.mode == 'pass@k' else baseline_xs,
    baseline_ys,
    markersize=3,
    color=dark_gray,
    zorder=4,
)[0]
plt.fill_between(
    baseline_ts if args.mode == 'pass@t' else baseline_ss if args.mode == 'pass@k' else baseline_xs,
    [y - y_err for y, y_err in zip(baseline_ys, y_errs)],
    [y + y_err for y, y_err in zip(baseline_ys, y_errs)],
    color=dark_gray,
    alpha=0.2,
    zorder=4,
)

for krn in krns:
    k, n = krn.split(",")
    k = k.strip()[1:]
    n = n.strip()[:-1]
    k = int(k)
    n = int(n)
    if n == 0: continue
    color = colors[k]
    marker = markers[n]
    ax.errorbar(krns[krn][1][0] if args.mode == 'pass@t' else krns[krn][2][0] if args.mode == 'pass@k' else k+k*n,
                krns[krn][0][0], color=color, marker=marker, zorder=5, linestyle="None", yerr=krns[krn][0][1], capsize=3, elinewidth=0.5,
                xerr=krns[krn][1][1] if args.mode == 'pass@t' else 0)


legend_elements = [
    Line2D([0], [0], color=colors[k], label=fr"$n_p = {k}$") for k in colors
] + [
    Line2D([0], [0], color=dark_gray, marker=markers[n], label=fr"$n_{{fr}}={n}$", linestyle="None")
    for n in markers
]
plt.rc('font', size=12)  
leg = plt.legend(loc="best", ncol=2, handles=legend_elements, prop={"size": 9})
leg.set_zorder(3)
# rotate x-axis ticks
#plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

fig.tight_layout()
fig.savefig(args.output, bbox_inches="tight", dpi=600)
