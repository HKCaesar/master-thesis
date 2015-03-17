#!/usr/bin/env python3

import sys
import os
import os.path
import numpy as np
import matplotlib.pyplot as plt

def match_angle(matches, shape):
    """Angle of the line connecting two matches
    when putting the two images side by side"""

    x1, y1, x2, y2 = matches[:,0], matches[:,1], matches[:,2], matches[:,3]
    return (180.0/np.pi)*np.arctan((y2 - y1) / (x2 - x1 + shape[1]))

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        path = sys.argv[1]
    else:
        path = "."

    shape = np.loadtxt(os.path.join(path, "shape.txt"))
    matches = np.loadtxt(os.path.join(path, "matches.txt"), comments="#")

    angles = match_angle(matches, shape)

    f = plt.figure()
    ax = f.add_subplot(111)
    ax.plot(angles)
    ax.set_xlabel("Feature number (by distance)")
    ax.set_ylabel("Angle")
    plt.show(f)
    plt.close(f)
