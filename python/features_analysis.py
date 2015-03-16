#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt

def angle(matches, shape):
    """Angle of the line connecting two matches
    when putting the two images side by side"""

    x1, y1, x2, y2 = matches[:,0], matches[:,1], matches[:,2], matches[:,3]
    return (y2 - y1) / (x2 - x1 + shape[1])

def distances_histogram(path, matches):
    f = plt.figure(figsize=(8,4))
    ax = f.add_subplot(111)
    ax.hist(matches[:,4], bins=50)
    ax.set_xlabel("Distance")
    ax.set_ylabel("Count")
    f.savefig(path, bbox_inches='tight')

def angles_histogram(path, matches, angles, threshold):
    if threshold is not None:
        angles = angles[matches[:,4] < threshold]
    if len(angles) > 0:
        f = plt.figure(figsize=(8,4))
        ax = f.add_subplot(111)
        ax.set_xlabel("Match line angle")
        ax.set_ylabel("Frequency")
        ax.hist(180/np.pi * np.arctan(angles), bins=100)
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
    matches = np.loadtxt(path + "/matches.txt")

    # Distances histogram
    distances_histogram(path + "/histogram_distances.pdf", matches)

    # Matches angle histogram
    angles = angle(matches, shape)
    angles_histogram(path + "/histogram_angles_all.pdf", matches, angles, None)
    angles_histogram(path + "/histogram_angles_10.pdf", matches, angles, 10)
    angles_histogram(path + "/histogram_angles_50.pdf", matches, angles, 50)
    angles_histogram(path + "/histogram_angles_100.pdf", matches, angles, 100)
    angles_histogram(path + "/histogram_angles_200.pdf", matches, angles, 200)
    angles_histogram(path + "/histogram_angles_300.pdf", matches, angles, 300)
