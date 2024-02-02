import argparse
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

red = "#B94700"
yellow = "#F0B323"
green = "#49C5B1"
dark_grey = "#898D8D"
light_grey = "#C7C9C7"
blue = "#009CDE"
purple = "#151F6D"

colors = [(0.1, 0.1, 0.1, 0.9), red, yellow, green, (0.9, 0.9, 0.9)]
bounds = [0, 0.65, 1.0, 1.35, 1.7]
cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list('', colors, N=256)
norm = plt.Normalize(vmin=0.70, vmax=1.30)

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

with open(args.input, "r") as f:
    krns = json.load(f)

baseline_krns = [f'({k}, 0)' for k in range(1, 51)]
baseline_ys = [np.mean(krns[krn][0][0]) for krn in baseline_krns]
if args.mode == 'pass@t':
    baseline_xs = [np.mean(krns[krn][1][0]) for krn in baseline_krns]
if args.mode == 'pass@k':
    baseline_xs = [np.mean(krns[krn][2][0]) for krn in baseline_krns]
if args.mode == 'pass@k-simple':
    baseline_xs = list(range(1, 51))

def interp(x): 
    return np.interp(x, baseline_xs, baseline_ys)


heatmap_data = {}
for krn in krns:
    k, n = krn.split(",")
    k = k.strip()[1:]
    n = n.strip()[:-1]
    k = int(k)
    n = int(n)
    if n == 0: continue  # skip baseline
    assert (k, n) not in heatmap_data.keys()
    if args.mode == 'pass@t':
        heatmap_data[(k, n)] = np.mean(krns[krn][0][0]) / interp(np.mean(krns[krn][1][0]))
        if np.mean(krns[krn][1][0]) > max(baseline_xs):
            heatmap_data[(k, n)] = 0
    if args.mode == 'pass@k':
        heatmap_data[(k, n)] = np.mean(krns[krn][0][0]) / interp(np.mean(krns[krn][2][0]))
        if np.mean(krns[krn][2][0]) > max(baseline_xs):
            heatmap_data[(k, n)] = 0
    if args.mode == 'pass@k-simple':
        heatmap_data[(k, n)] = np.mean(krns[krn][0][0]) / interp(k + k * n)
        if k + k*n > max(baseline_xs):
            heatmap_data[(k, n)] = 0

ks = sorted(list(set([k[0] for k in heatmap_data.keys()])), reverse=False)
ns = sorted(list(set([k[1] for k in heatmap_data.keys()])), reverse=True)

data = np.zeros((len(ns), len(ks)))
for i, n in enumerate(ns):
    for j, k in enumerate(ks):
        data[i, j] = heatmap_data[(k, n)]

# generate heatmap from `data`
heatmap = ax.imshow(data, interpolation="nearest", aspect="auto", cmap=cmap, norm=norm)
# show ks and ns as ticks
ax.set_xticks(np.arange(len(ks)), labels=ks)
ax.set_yticks(np.arange(len(ns)), labels=ns)
# rotate x-axis ticks
#plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
# Loop over data dimensions and create text annotations.
for i in range(len(ns)):
    for j in range(len(ks)):
        if data[i, j] > 0.0:
            text = ax.text(
                j,
                i,
                f"{data[i, j]:.2f}",
                ha="center",
                va="center",
                color="w",
                fontweight='bold',
            )
        else:
            text = ax.text(
                j,
                i,
                'O.O.B.',
                ha="center",
                va="center",
                color="w",
                fontweight='bold',
            )
# set y-axis label
ax.set_ylabel("Feedback-repairs " + r"$(n_{{fr}})$" )
ax.set_xlabel("Initial programs " + r"$(n_p)$")
plt.rc('font', size=12)  
fig.tight_layout()
fig.savefig(args.output, bbox_inches="tight", dpi=600)
