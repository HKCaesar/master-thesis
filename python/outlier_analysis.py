#!/usr/bin/env python3

import os
import os.path
import sys
import numpy as np
import matplotlib.pyplot as plt

from features_analysis import match_angle, features_figsize

def outlier_frequency_plot(path, angles, threshold):
    f = plt.figure(figsize=features_figsize)
    ax = f.add_subplot(111)
    ax.plot(100 * np.cumsum(np.abs(angles) > threshold) / angles.size)
    ax.set_xlabel("Match number (by distance)")
    ax.set_ylabel("Outlier fraction (%)")
    f.savefig(path, bbox_inches='tight')

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: ./outlier_analysis.py <dir>")
        sys.exit(-1)

    # Produce outlier plots for all directories containing outlier_threshold.txt
    path = sys.argv[1]
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
