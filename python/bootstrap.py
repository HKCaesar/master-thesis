#!/usr/bin/env python3

import sys
import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from project import Project
from features_common import base_plot
from IPython import embed

def extract_cameras(bootstrap):
    number_of_samples, number_of_cameras, six = bootstrap.externals.shape
    assert number_of_samples == bootstrap.number_of_samples
    assert six == 6
    for cam_number in range(number_of_cameras):
        yield bootstrap.externals[:, cam_number, :]

def covariance_externals(bootstrap):
    "List of covariance matrices of external orientation of each camera"
    return [np.cov(cam, rowvar=0) for cam in extract_cameras(bootstrap)]

def covariance_internals(bootstrap):
    "Covariance matrix of internal orientation"
    number_of_samples, four = bootstrap.internals.shape
    assert number_of_samples == bootstrap.number_of_samples
    assert four == 4
    return np.cov(bootstrap.internals, rowvar=0)

def plot_covariance_matrix(S):
    "Make a covariance matrix plot"
    f, ax = base_plot()
    ax.imshow(S, interpolation="nearest", cmap="gray")
    return f, ax

def plot_scatter(X, Y):
    f, ax = base_plot(figsize=(8,4))
    ax.plot(X, Y, '.')
    return f, ax

def plot_distribution(X):
    f, ax = base_plot(figsize=(8,4))
    ax.hist(X)
    return f, ax

def main():
    if len(sys.argv) < 2:
        print("Usage: ./bootstrap.py <project_dir>")
        sys.exit(-1)

    project_dir = sys.argv[1]
    project = Project(join(project_dir, "project.json"))

    for (bootstrap_number, bootstrap) in enumerate(project.bootstraps):
        boot_dir = os.path.abspath(join(project_dir, "bootstrap{}-{}".format(bootstrap_number, type(bootstrap.base_model).__name__)))
        os.makedirs(boot_dir, exist_ok=True)

        # Covariance matrices
        ext_covs = covariance_externals(bootstrap)
        int_cov = covariance_internals(bootstrap)

        # X, Y, Z and angles
        for (cam_number, S) in enumerate(ext_covs):
            f, ax = plot_covariance_matrix(S[:3, :3])
            f.savefig(join(boot_dir, "covariance-cam{}-pos.pdf".format(cam_number)), bbox_inches='tight')
            f, ax = plot_covariance_matrix(S[3:, 3:])
            f.savefig(join(boot_dir, "covariance-cam{}-angles.pdf".format(cam_number)), bbox_inches='tight')

        # Interior
        f, ax = plot_covariance_matrix(int_cov)
        f.savefig(join(boot_dir, "covariance-interior.pdf"), bbox_inches='tight')

        # Scatter and distribution plots
        for (cam_number, cam) in enumerate(extract_cameras(bootstrap)):
            f, ax = plot_scatter(cam[:,0], cam[:, 1])
            f.savefig(join(boot_dir, "cam{}-xy.pdf".format(cam_number)), bbox_inches='tight')
            f, ax = plot_distribution(cam[:,2])
            f.savefig(join(boot_dir, "cam{}-z.pdf".format(cam_number)), bbox_inches='tight')

if __name__ == "__main__":
    main()
