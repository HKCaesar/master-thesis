#!/usr/bin/env python3

import os
import os.path
import sys
import numpy as np
import matplotlib.pyplot as plt

from features_common import match_angle, base_plot

def outlier_frequency_plot(path, angles, threshold):
    f, ax = base_plot()
    w = 80 # window size
    running_avg = np.convolve(np.array(np.abs(angles) > threshold, dtype=np.float), np.ones((w,))/w, mode="valid")
    # offset by half window size
    ax.plot(np.arange(w/2-1, angles.size - w/2), 100 * running_avg)
    ax.set_xlabel("Match number (by distance)")
    ax.set_ylabel("Outlier fraction (%)")
    f.savefig(path, bbox_inches='tight')
    plt.close(f)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        path = "."
    else:
        path = sys.argv[1]

    # Produce outlier plots for all directories containing outlier_threshold.txt
    for root, subdirs, files in os.walk(path):
        if "matches.txt" in files:
            shape = np.loadtxt(os.path.join(root, "shape.txt"))
            matches = np.loadtxt(os.path.join(root, "matches.txt"), comments="#")
            threshold = np.loadtxt(os.path.join(root, "outlier_threshold.txt"))

            if threshold.size == 1:
                print("outlier_analysis.py: " + root)
                # Compute matches angles
                angles = match_angle(matches, shape)
                outlier_frequency_plot(os.path.join(root, "plot_outliers.pdf"), angles, threshold)
            else:
                print("outlier_analysis.py: " + root + " --- empty outlier_threshold.txt")
