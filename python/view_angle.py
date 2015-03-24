#!/usr/bin/env python3

import sys
import os
import os.path
import numpy as np
import matplotlib.pyplot as plt

from features_common import match_angle, base_plot

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        path = sys.argv[1]
    else:
        path = "."

    shape = np.loadtxt(os.path.join(path, "shape.txt"))
    matches = np.loadtxt(os.path.join(path, "matches.txt"), comments="#")

    angles = match_angle(matches, shape)

    f, ax = base_plot()
    ax.vlines(np.arange(angles.size), 0, angles)
    ax.set_ylim([-80, 80])
    ax.set_xlabel("Feature number (by distance)")
    ax.set_ylabel("Angle")
    plt.show(f)
    plt.close(f)
