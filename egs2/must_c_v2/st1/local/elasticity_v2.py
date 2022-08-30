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

    
    
    step = 0.2
    # import pdb;pdb.set_trace()
    x = np.array([i * step for i in range(8)])    

    #define subplots
    fig, ax = plt.subplots(2, 2, figsize=(9,6))
    fig.tight_layout()

    #create subplots
    ax[0, 0].set_title('MT BLEU Elasticity')
    ax[0, 1].set_title('MT Length Ratio Elasticity')
    ax[0, 1].yaxis.set_label_position("right")
    ax[0, 1].yaxis.tick_right()
    ax[1, 0].set_title('ST BLEU Elasticity')
    ax[1, 1].set_title('ST Length Ratio Elasticity')
    ax[1, 1].yaxis.set_label_position("right")
    ax[1, 1].yaxis.tick_right()

    ax[0, 0].set(xlabel=None, ylabel="BLEU")
    ax[0, 0].set_xticks([])
    ax[0, 1].set_xticks([])
    ax[0, 1].set(xlabel=None, ylabel="Length Ratio")
    ax[1, 0].set(xlabel='Length Penalty', ylabel="BLEU")
    ax[1, 1].set(xlabel='Length Penalty', ylabel="Length Ratio")

    colors = {'B':'red', 'JT':'green', 'JL': 'blue'}
    labels = {'B':'Baseline', 'JT':'Joint (Time Sync)', 'JL':'Joint (Label Sync)'}
    ax_00_2nd = ax[0, 0].twinx()
    ax_10_2nd = ax[1, 0].twinx()
    for i, src in enumerate(args.src):
        pth, task, model = src
        bleu, ratio = read_src(pth)
        # import pdb;pdb.set_trace()
        c = colors[model]
        l = labels[model]
        if task == "MT":
            y = bleu
            ax[0, 0].plot(x, y, color=c, label=l)
            y = ratio
            ax_00_2nd.plot(x, y, color=c, linestyle=':', label=l)
        else:
            y = bleu
            ax[1, 0].plot(x, y, color=c, label=l)
            y = ratio
            ax_10_2nd.plot(x, y, color=c, linestyle=':', label=l)

    # plt.legend()
    ax.flatten()[-2].legend(loc='upper center', bbox_to_anchor=(0.7, -0.25), ncol=3)


    plt.show()
    plt.savefig(args.dst)