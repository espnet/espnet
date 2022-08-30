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

    plt.rcParams["figure.figsize"] = (9,3.0)
    x = [i for i in range(1,7)]
    y_proposed = [0.892300, 0.831333, 0.780579, 0.838241, 0.869596, 0.837247]
    # e_proposed = [[0.049730,0.080444,0.003945,0.073136], [0.068184,0.073516,0.071845,0.042562], [0.081950,0.088862,0.062866,0.063304],\
                # [0.063866,0.070170,0.064630,0.069853], [0.066741,0.051543,0.073998,0.062330], [0.070470,0.068899,0.075186,0.063695]]
    # e_proposed = [sum(l)/len(l) for l in e_proposed]
    y_baseline = [0.731465, 0.746485, 0.786913, 0.784196, 0.766593, 0.738538]
    # e_baseline = [[0.116359,0.081406,0.091446,0.084511], [0.097186,0.085955,0.074285,0.094891], [0.079609,0.094538,0.084452,0.072140],\
                # [0.072543,0.071802,0.082437,0.078143], [0.076045,0.069411,0.077664,0.090006], [0.088222,0.086848,0.076313,0.079692]]
    # e_baseline = [sum(l)/len(l) for l in e_baseline]


    # plt.errorbar(x, y_baseline, e_baseline, linestyle='-', marker='D', capsize=5, color='r', label="Baseline")
    # plt.errorbar(x, y_proposed, e_proposed, linestyle='-', marker='D', capsize=5, color='b', label="Joint CTC/Attn")
    plt.errorbar(x, y_baseline, linestyle='-', marker='d', color='r', label="Baseline")
    plt.errorbar(x, y_proposed, linestyle='-', marker='d', color='b', label="Joint CTC/Attn")
    # plt.legend()

    plt.legend(loc='lower center', ncol=2, fontsize=14)
    plt.title('Layer-wise Monotonicity of ST Decoder Source Attention', fontsize=19)
    plt.xlabel('ST Decoder Layer', fontsize=19)
    plt.ylabel("Monotonicity", fontsize=19)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.show()
    plt.savefig(args.dst)