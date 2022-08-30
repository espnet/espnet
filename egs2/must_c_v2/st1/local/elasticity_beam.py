import argparse
import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

from espnet2.utils.types import str2triple_str

def get_parser():
    parser = argparse.ArgumentParser(
        description="Calculate the diagonality of the self-attention weights."
    )
    parser.add_argument(
        "--src",
        type=str2triple_str,
        help="path to results",
        action="append",

    )
    parser.add_argument(
        "--dst",
        type=str,
        help="path to results"
    )
    return parser

def read_src(src):
    lines = open(src, "r").readlines()
    data = [l.split() for l in lines]
    data = [(l[2], l[9]) for l in data]
    return [float(l[0]) for l in data], [float(l[1]) for l in data]

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    
    
    step = 1
    # import pdb;pdb.set_trace()
    x_labels = ['', '5', '10', '20', '30', '40', '50', '100', '200']
    x = np.array([i * step for i in range(8)])   

    #define subplots
    fig, ax = plt.subplots(2, 2, figsize=(9,6))
    fig.tight_layout()

    #create subplots
    ax[0, 0].set_title('Beam Size Elasticity of MT BLEU')
    ax[0, 1].set_title('Beam Size Elasticity of MT Length Ratio')
    ax[0, 1].yaxis.set_label_position("right")
    ax[0, 1].yaxis.tick_right()
    ax[1, 0].set_title('Beam Size Elasticity of ST BLEU')
    ax[1, 1].set_title('Beam Size Elasticity of ST Length Ratio')
    ax[1, 1].yaxis.set_label_position("right")
    ax[1, 1].yaxis.tick_right()

    ax[0, 0].set(xlabel=None, ylabel="MT BLEU")
    ax[0, 0].set_xticks([])
    ax[0, 1].set_xticks([])
    ax[0, 1].set(xlabel=None, ylabel="Length Ratio")
    ax[1, 0].set(xlabel='Beam Size', ylabel= "ST BLEU")
    ax[1, 1].set(xlabel='Beam Size', ylabel="Length Ratio")

    ax[1, 0].set_xticklabels(x_labels)
    ax[1, 1].set_xticklabels(x_labels)

    colors = {'B':'red', 'JT':'green', 'JL': 'blue'}
    labels = {'B':'Baseline (Output Sync)', 'JT':'Joint (Input Sync)', 'JL':'Joint (Output Sync)'}
    for i, src in enumerate(args.src):
        pth, task, model = src
        bleu, ratio = read_src(pth)
        # import pdb;pdb.set_trace()
        c = colors[model]
        l = labels[model]
        if task == "MT":
            y = bleu
            ax[0, 0].plot(x, y, color=c, label=l, linewidth=2, marker='o', markerfacecolor="None")
            y = ratio
            ax[0, 1].plot(x, y, color=c, label=labels, linewidth=2, marker='o', markerfacecolor="None")
        else:
            y = bleu
            ax[1, 0].plot(x, y, color=c, label=l, linewidth=2, marker='o', markerfacecolor="None")
            y = ratio
            ax[1, 1].plot(x, y, color=c, label=l, linewidth=2, marker='o', markerfacecolor="None")

    # plt.legend()
    ax.flatten()[-1].legend(loc='lower right')
    # handles, labels = ax[1][0].get_legend_handles_labels()
    # fig.legend(handles, labels, ncol=3, loc='lower center')
    # fig.subplots_adjust(bottom=0.2)

    plt.show()
    plt.savefig(args.dst)