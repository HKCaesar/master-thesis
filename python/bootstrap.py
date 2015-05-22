#!/usr/bin/env python3

import sys
import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from project import Project
from features_common import base_plot
from IPython import embed

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

    # For each bootstraped model
    for (bootstrap_number, bootstrap) in enumerate(project.bootstraps):
        boot_dir = os.path.abspath(join(project_dir, "bootstrap{}-{}".format(bootstrap_number, type(bootstrap.base_model).__name__)))
        os.makedirs(boot_dir, exist_ok=True)

        # Interior
        f, ax = plot_covariance_matrix(np.cov(bootstrap.internals, rowvar=0))
        f.savefig(join(boot_dir, "covariance-interior.pdf"), bbox_inches='tight')

        # For each camera
        for (cam_number, cam) in enumerate(bootstrap.extract_cameras()):
            # Scatter and distribution plots
            f, ax = plot_scatter(cam[:,0], cam[:, 1])
            f.savefig(join(boot_dir, "cam{}-xy.pdf".format(cam_number)), bbox_inches='tight')
            f, ax = plot_distribution(cam[:,2])
            f.savefig(join(boot_dir, "cam{}-z.pdf".format(cam_number)), bbox_inches='tight')

            # X, Y, Z and angles covariances matrices
            S = np.cov(cam, rowvar=0)
            f, ax = plot_covariance_matrix(S[:3, :3])
            f.savefig(join(boot_dir, "covariance-cam{}-pos.pdf".format(cam_number)), bbox_inches='tight')
            f, ax = plot_covariance_matrix(S[3:, 3:])
            f.savefig(join(boot_dir, "covariance-cam{}-angles.pdf".format(cam_number)), bbox_inches='tight')

if __name__ == "__main__":
    main()
