#!/usr/bin/env python3

import sys
import os
import os.path
import shutil
import numpy as np
import matplotlib.pyplot as plt

from features_common import *
from features_common import __file__ as features_common_filename
from view_angle import __file__ as view_angle_filename
from outlier_analysis import __file__ as outlier_analysis_filename

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

            # Spatial coverage plot
            spatial_coverage_plot(os.path.join(root, "plot_spatial_coverage.pdf"), matches, shape)

            # Copy scripts
            shutil.copy(features_common_filename, root)
            shutil.copy(view_angle_filename, root)
            shutil.copy(outlier_analysis_filename, root)

            # Create empty outlier_threshold.txt unless it exists
            otf = os.path.join(root, "outlier_threshold.txt")
            if not os.path.isfile(otf):
                f = open(otf, "w")
                f.write(" ")
                f.close()
