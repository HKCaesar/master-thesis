import numpy as np
from scipy.stats import poisson, chi2, chisquare
import matplotlib.pyplot as plt

def divide_count(matches, image_shape, count):
    """
    Divide the image in a regular grid and returns the number of
    matches per grid cell
    """
    # Number of points in grid cells
    areas = np.zeros((count, count))
    cell_size = image_shape / count
    for x1, y1, x2, y2, dist in matches:
        # Euclidian division to get cell coordinate
        # Only consider image 1 keypoints
        areas[y1 // cell_size[0], x1 // cell_size[1]] += 1
    return areas

def spatial_chi2_plot(path, matches, image_shape, count):
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

    # To make a plot out of it
    # p = zeros(N)
    # for m in range(10, N):
    #     ...:     p[m] = spatial_distribution_plot("", matches[:m,:], shape, int(np.sqrt(m)))

