#!/usr/bin/env python3

import sys
import os
import os.path
import shutil
import numpy as np
from scipy.stats import poisson, chi2, chisquare
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

def divide_count(matches, image_shape, count):
    # Number of points in grid cells
    areas = np.zeros((count, count))
    cell_size = shape / count
    for x1, y1, x2, y2, dist in matches:
        # Euclidian division to get cell coordinate
        # Only consider image 1 keypoints
        areas[y1 // cell_size[0], x1 // cell_size[1]] += 1
    return areas

def spatial_distribution_plot(path, matches, image_shape, count):
    # Number of observations
    N = matches.shape[0]

    # Expected mean under uniformity hypothesis
    mu = N / count**2

    # Select a number of bins heuristically such that the
    # last (open-ended) one expects to have 0.5 elements
    bins = list(np.arange(poisson.ppf(1 - 0.5/N, mu=mu)+1)) + [1e6]
    counts, bins = np.histogram(divide_count(matches, image_shape, count).flatten(), bins=bins)

    # All areas should be non zero for test to be valid
    # (ideally more than ~5)
    # assert(np.all(counts > 1))

    # Compare expected count with chi2 test
    expected_counts = poisson.pmf(bins[:-1], mu=mu)*N

    # Mean is estimated from the data, so the number of degrees of freedom is
    df = (len(bins)-1) - 1 - 1

    # print(list(zip(counts, [int(10*round(x, 1))/10 for x in expected_counts])))

    # Return p-value
    chi2sum = np.sum((counts - expected_counts)**2 / expected_counts)
    return 1.0 - chi2.cdf(chi2sum, df=df)

    # p = zeros(N)
    # for m in range(10, N):
    #     ...:     p[m] = spatial_distribution_plot("", matches[:m,:], shape, int(np.sqrt(m)))

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
