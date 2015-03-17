#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt

features_figsize = (8,4)

def match_angle(matches, shape):
    """Angle of the line connecting two matches
    when putting the two images side by side"""

    x1, y1, x2, y2 = matches[:,0], matches[:,1], matches[:,2], matches[:,3]
    return (y2 - y1) / (x2 - x1 + shape[1])

def distances_plot(path, sorted_matches):
    f = plt.figure(figsize=features_figsize)
    ax = f.add_subplot(111)
    ax.plot(sorted_matches[:,4])
    ax.set_xlabel("Feature number (by distance)")
    ax.set_ylabel("Distance")
    f.savefig(path, bbox_inches='tight')

def angle_spread_plot(path, sorted_angles):
    f = plt.figure(figsize=features_figsize)
    ax = f.add_subplot(111)
    ax.plot(np.maximum.accumulate(angles) - np.minimum.accumulate(angles))
    ax.plot(angles)
    f.savefig(path, bbox_inches='tight')

def angles_histogram(path, matches, angles, threshold):
    if threshold is not None:
        angles = angles[matches[:,4] < threshold]
    if len(angles) > 0:
        f = plt.figure(figsize=(8,4))
        ax = f.add_subplot(111)
        ax.set_xlabel("Match line angle")
        ax.set_ylabel("Frequency")
        ax.hist(180/np.pi * np.arctan(angles), bins=100, range=(-30, 30))
        f.savefig(path, bbox_inches='tight')

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
    matches = np.loadtxt(path + "/matches.txt", comments="#")

    # Distances plot
    distances_plot(path + "/plot_distances.pdf", matches)

    # Angle spread plot
    angles = match_angle(matches, shape)
    angle_spread_plot(path + "/plot_angle_spread.pdf", angles)
