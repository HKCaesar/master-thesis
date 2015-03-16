#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt

def angle(matches, shape):
    """Angle of the line connecting two matches
    when putting the two images side by side"""

    x1, y1, x2, y2 = matches[:,0], matches[:,1], matches[:,2], matches[:,3]
    return (y2 - y1) / (x2 - x1 + shape[1])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: ./features_analysis.py <dir>"
              "       <dir> is the result a a features analysis"
              "        should contain matches.txt and kp1.jpg")
        sys.exit(-1)

    path = sys.argv[1]

    # Load kp1.jpg to get image size
    kp1 = plt.imread(path + "/kp1.jpg");
    shape = kp1.shape

    # Load matches, keypoints and distances
    matches = np.loadtxt(path + "/matches.txt")

    # Distances histogram
    f = plt.figure(figsize=(8,4))
    ax = f.add_subplot(111)
    ax.hist(matches[:,4], bins=50, normed=True)
    f.savefig(path + "histogram_distances.pdf", bbox_inches='tight')

    # Matches angle histogram
    angles = angle(matches, shape)
    f = plt.figure(figsize=(8,4))
    ax = f.add_subplot(111)
    ax.hist(180/np.pi * np.arctan(angles), bins=100)
    f.savefig(path + "/histogram_angles.pdf", bbox_inches='tight')
