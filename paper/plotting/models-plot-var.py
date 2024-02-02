import json
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import argparse

def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--inputs',
        required=True,
        nargs='+',
    )
    parser.add_argument(
        '--labels',
        required=True,
        nargs='+',
    )
    parser.add_argument(
        '--output',
        required=True,
    )
    parser.add_argument(
        '--mode',
        required=True,
        type=str,
        choices=['pass@t', 'pass@k', 'pass@k-simple']
    )
    parser.add_argument(
        '--xmax',
        required=False,
        type=int,
        default=10000,
    )
    parser.add_argument(
        '--legend',
        required=False,
        default='full',
        choices=['full', 'none', 'partial']
    )
    return parser

args = build_args().parse_args()

assert len(args.inputs) == len(args.labels), f'Number of inputs ({len(args.inputs)}) must match number of labels ({len(args.labels)})'

plt.rc('font', size=12)  

red = "#B94700"
yellow = "#F0B323"
green = "#49C5B1"
blue = "#009CDE"
purple = "#151F6D"
dark_gray = "#515555"
pink = "#51827D"
#soft_lavender = "#A68DF2"
soft_lavender = "#8A7DDB"
taupe = "#A89F91"
#olive_green = "#6B8E23"
olive_green = "#4B6920"
bright_green = "#00AA00"

color_map = {(x, y): c for x, y, c in 
             [('GPT-4', 'baseline', purple),
              ('GPT-4', 'repair', green),
              ('GPT-3.5', 'baseline', dark_gray),
              ('GPT-3.5', 'repair', red),
              ('GPT-3.5', 'GPT-4', blue),
              ('Code Llama', 'baseline', olive_green),
              ('Code Llama', 'repair', soft_lavender),
              ('Code Llama', 'GPT-3.5', yellow),
              ('Code Llama', 'GPT-4', bright_green),
             ]}

fig = plt.figure(figsize=(4, 3))
ax = plt.gca()

if args.mode == 'pass@t':
    ax.set_xlabel("Mean number of tokens generated")
if args.mode == 'pass@k':
    ax.set_xlabel("Mean number of samples")
if args.mode == 'pass@k-simple':
    ax.set_xlabel(r"Number of programs sampled")
ax.set_ylabel("Mean pass rate")
handles = []

plotted_baselines = set()

for inp, label in zip(args.inputs, args.labels):
    with open(inp, "r") as f:
        krns = json.load(f)
    
    if label.split('+')[0] not in plotted_baselines:
        # plot baseline first
        label = label.split('+')[0]
        plotted_baselines.add(label)
        baseline_krns = [f"({k}, 0)" for k in range(1, 51)]
        if args.mode == 'pass@t':
            baseline_xs = [krns[krn][1][0] for krn in baseline_krns]
        if args.mode == 'pass@k':
            baseline_xs = [krns[krn][2][0] for krn in baseline_krns]
        if args.mode == 'pass@k-simple':
            baseline_xs = [k for k in range(1, 51)]
        baseline_ys = [krns[krn][0][0] for krn in baseline_krns]
        ys_vars = [krns[krn][0][1] for krn in baseline_krns]

        handles.append(
            ax.plot(
                baseline_xs,
                baseline_ys,
                marker=",",
                label=rf"$M_P =$ {label} (no repair)",
                color=color_map[(label, 'baseline')],
                zorder=4,
            )[0]
        )
        # also plot a shaded region for all points with the maximum variance (max_var)
        ax.fill_between(
            baseline_xs,
            [baseline_ys[i] - ys_vars[i] for i in range(len(baseline_ys))],
            [baseline_ys[i] + ys_vars[i] for i in range(len(baseline_ys))],
            color=color_map[(label, 'baseline')],
            alpha=0.2,
        )
    if '+' not in label:
        # plot repair
        xs = []
        ys = []
        ys_vars = []
        for k in range(1, 26):
            krn = f"({k}, 1)"
            if args.mode == 'pass@t':
                xs.append(krns[krn][1][0])
            if args.mode == 'pass@k':
                xs.append(krns[krn][2][0])
            if args.mode == 'pass@k-simple':
                xs.append(k+k*1)
            ys.append(krns[krn][0][0])
            ys_vars.append(krns[krn][0][1])
        handles.append(
            ax.plot(xs, ys, marker=",", label=rf"$M_P = M_F =$ {label}", color=color_map[(label, 'repair')], zorder=5)[
                0
            ]
        )
        # also plot a shaded region for all points with the variance
        ax.fill_between(
            xs, [ys[i] - ys_vars[i] for i in range(len(ys))], [ys[i] + ys_vars[i] for i in range(len(ys))], color=color_map[(label, 'repair')], alpha=0.2,)
    else:
        # for M_P + M_F, we only plot the repair
        mp, mf = label.split('+')
        xs = []
        ys = []
        ys_vars = []
        for k in range(1, 26):
            krn = f"({k}, 1)"
            if args.mode == 'pass@t':
                xs.append(krns[krn][1][0])
            if args.mode == 'pass@k':
                xs.append(krns[krn][2][0])
            if args.mode == 'pass@k-simple':
                xs.append(k+k*1)
            ys.append(krns[krn][0][0])
            ys_vars.append(krns[krn][0][1])
        handles.append(
            ax.plot(xs, ys, marker=",", label=rf"$M_P =$ {mp}; $M_F =$ {mf}", color=color_map[(mp, mf)], zorder=5)[
                0
            ]
        )
        # also plot a shaded region for all points with the variance
        ax.fill_between(
            xs, [ys[i] - ys_vars[i] for i in range(len(ys))], [ys[i] + ys_vars[i] for i in range(len(ys))], color=color_map[(mp, mf)], alpha=0.2,)


ax.set_ylim(0.0, 1.0)
if args.mode == 'pass@t':
    ax.set_xlim(0.0, args.xmax)
if args.mode == 'pass@k' or args.mode == 'pass@k-simple':
    ax.set_xlim(0.0, 50)
# set x axis ticks to 10, 20, 30, 40, 50
ax.set_xticks([10, 20, 30, 40, 50])
# set y axis ticks to 0.2, 0.4, 0.6, 0.8, 1.0
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.grid(True)

fig.savefig(args.output, bbox_inches="tight", dpi=600)
#plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
if args.legend != 'none':
    def label_order(label):
        if "no repair" in label:
            return (0, label)
        elif "M_P = M_F" in label:
            return (1, label)
        else:
            return (2, label)
    if args.legend == 'full':
        handles.append(Line2D([], [], marker='None', color=color_map[('Code Llama', 'GPT-4')], label=r"$M_P =$ Code Llama; $M_F =$ GPT-4"))
    handles.sort(key=lambda x: label_order(x.get_label()))
    fig = plt.figure(figsize=(2, 3))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    leg = ax.legend(loc="center", handles=handles, frameon=False, fontsize=16)
    for line in leg.get_lines():
        line.set_linewidth(12.0)
    #leg = plt.legend(loc="center left", handles=handles, prop={"size": 9}, bbox_to_anchor=(1.0, 0.5))
    leg.set_zorder(3)
    dt = args.output.split('.')[-1]
    name = '.'.join(args.output.split('.')[:-1]) + "-legend." + dt
    fig.savefig(name, bbox_inches="tight", dpi=600)
