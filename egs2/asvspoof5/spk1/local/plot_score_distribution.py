# plot_score_distribution.py
# This score_file takes in a score_file containing SASV scores
# and plots the distribution of the scores.
# The output is a histogram of the scores. Each class is
# represented by a different color.
# The input score_file is assumed to be in the format:
# <enrollment_id> <test_id> <score> <class>

import sys

import matplotlib.pyplot as plt
import numpy as np
from a_dcf import a_dcf
from dataclasses import dataclass

@dataclass
class SASVCostModel:
    "Class describing SASV-DCF's relevant costs"
    Pspf: float = 0.05
    Pnontrg: float = 0.0095
    Ptrg: float = 0.9405
    Cmiss: float = 1
    Cfa_asv: float = 10
    Cfa_cm: float = 10

def plot_distribution(score_file, metadata_file):
    # compute the a-dcf
    adcf_results = a_dcf.calculate_a_dcf(score_file, cost_model=SASVCostModel())
    adcf = adcf_results["min_a_dcf"]
    threshold = adcf_results["min_a_dcf_thresh"]
    # Read
    data = np.loadtxt(score_file, dtype="str")
    scores = data[:, 2].astype(float)
    classes = data[:, 3]
    # Plot
    plt.hist(
        scores[classes == "target"], bins=100, alpha=0.3, label="target", color="r"
    )
    plt.hist(
        scores[classes == "nontarget"],
        bins=100,
        alpha=0.3,
        label="nontarget",
        color="g",
    )
    plt.hist(scores[classes == "spoof"], bins=100, alpha=0.3, label="spoof", color="b")
    plt.axvline(x=threshold, color="k", linestyle="--", label="accept threshold")
    plt.legend(loc="upper right")
    plt.xlabel("SASV score")
    plt.ylabel("Frequency")
    plt.title("Distribution of SASV scores for a-DCF: {:.3f}".format(adcf))
    # save the plot
    plt.savefig("distribution.png")

    # the metadata file has the following format:
    # <speaker_id> <utt_id> <sex> <space> <spoofingtype> <spoofOrbonafide>
    metadata = np.loadtxt(metadata_file, dtype="str")
    spoofingtype_dict = {}
    for row in metadata:
        utt_id = row[1]
        spoofingtype = row[-2]
        spoof = row[-1]
        # if spoof == "spoof":
        spoofingtype_dict[utt_id] = spoofingtype

    # print(spoofingtype_dict)
    # make sure it is a new figure
    plt.figure()
    # the spoofingtypes are: A09, A10, A11, A12, A13, A14, A15, A16
    # we will plot the distribution of scores for each spoofingtype, on a single plot
    for spoofingtype in ["A09", "A10", "A11", "A12", "A13", "A14", "A15", "A16"]:
        spoofingtype_scores = scores[np.array([spoofingtype_dict[utt_id] == spoofingtype for utt_id in data[:, 1]])]
        plt.hist(spoofingtype_scores, bins=100, alpha=0.3, label=spoofingtype)
    # add the accept threshold
    plt.axvline(x=threshold, color="k", linestyle="--", label="accept threshold")
    plt.legend(loc="upper right")
    plt.xlabel("SASV score")
    plt.ylabel("Frequency")
    plt.title("Distribution of SASV scores for a-DCF: {:.3f} with A12".format(adcf))
    # save the plot
    plt.savefig("distribution_spoofingtypes.png")



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python plot_distribution.py <score_file> <metadata_file>")
        sys.exit(1)
    plot_distribution(sys.argv[1], sys.argv[2])
