#!/usr/bin/env python3

import sys
import os
import os.path
import shutil
import numpy as np
import matplotlib.pyplot as plt

from view_angle import __file__ as va_filename

features_figsize = (8,4)

def match_angle(matches, shape):
    """Angle of the line connecting two matches
    when putting the two images side by side"""

    x1, y1, x2, y2 = matches[:,0], matches[:,1], matches[:,2], matches[:,3]
    return (180.0/np.pi)*np.arctan((y2 - y1) / (x2 - x1 + shape[1]))

def distances_plot(path, sorted_matches):
    f = plt.figure(figsize=features_figsize)
    ax = f.add_subplot(111)
    ax.plot(sorted_matches[:,4])
    ax.set_xlabel("Match number (by distance)")
    ax.set_ylabel("Distance")
    f.savefig(path, bbox_inches='tight')
    plt.close(f)

def angle_spread_plot(path, sorted_angles):
    f = plt.figure(figsize=features_figsize)
    ax = f.add_subplot(111)
    ax.plot(angles)
    ax.set_ylim([-3, 3])
    f.savefig(path, bbox_inches='tight')
    plt.close(f)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: ./features_analysis.py <dir>")
        sys.exit(-1)

    # Produce plots for all directories containing matches.txt
    path = sys.argv[1]
    for root, subdirs, files in os.walk(path):
        if "matches.txt" in files:
            print("features_analysis.py: " + root)
            shape = np.loadtxt(os.path.join(root, "shape.txt"))
            matches = np.loadtxt(os.path.join(root, "matches.txt"), comments="#")

            # Distances plot
            distances_plot(os.path.join(root, "plot_distances.pdf"), matches)

            # Angle spread plot
            angles = match_angle(matches, shape)
            angle_spread_plot(os.path.join(root, "plot_angle_spread.pdf"), angles)

            # Copy view_angle_script.py
            shutil.copy(va_filename, root)

            # Create empty outlier_threshold.txt unless it exists
            otf = os.path.join(root, "outlier_threshold.txt")
            if not os.path.isfile(otf):
                f = open(otf, "w")
                f.write(" ")
                f.close()
