# plot_distribution.py
# This file takes in a file containing SASV scores
# and plots the distribution of the scores.
# The output is a histogram of the scores. Each class is
# represented by a different color.
# The input file is assumed to be in the format:
# <enrollment_id> <test_id> <score> <class>

import sys
import matplotlib.pyplot as plt
import numpy as np

def plot_distribution(file):
    # Read
    data = np.loadtxt(file, dtype='str')
    scores = data[:,2].astype(float)
    classes = data[:,3]
    # Plot
    plt.hist(scores[classes == 'target'], bins=100, alpha=0.3, label='target', color='r')
    plt.hist(scores[classes == 'nontarget'], bins=100, alpha=0.3, label='nontarget', color='g')	
    plt.hist(scores[classes == 'spoof'], bins=100, alpha=0.3, label='spoof', color='b')
    plt.legend(loc='upper right')
    plt.xlabel('SASV score')
    plt.ylabel('Frequency')
    plt.title('Distribution of SASV scores')
    # save the plot
    plt.savefig('distribution.png')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python plot_distribution.py <file>")
        sys.exit(1)
    plot_distribution(sys.argv[1])
