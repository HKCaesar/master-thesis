#!/usr/bin/env python3

import sys
import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from project import Project
from features_common import base_plot
from IPython import embed


def blob(ax, x, y, area, color):
    if area > 0:
        hs = np.sqrt(area) / 2
        xcorners = np.array([x - hs, x + hs, x + hs, x - hs])
        ycorners = np.array([y - hs, y - hs, y + hs, y + hs])
        ax.fill(xcorners, ycorners, color=color, edgecolor=color)

def plot_covariance_matrix(S, labels):
    "Plot a covariance matrix with Hinton-ish diagram"
    covariance_matrix_font_size = 18
    f, ax = base_plot(figsize=(4,4))
    ax.invert_yaxis()
    # ax.imshow(S, interpolation="nearest", cmap="gray")
    n = S.shape[0]
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_xaxis().tick_top()
    ax.get_xaxis().set_ticks_position('none')
    ax.get_yaxis().set_ticks_position('none')
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, fontsize=covariance_matrix_font_size)
    ax.set_yticklabels(labels, fontsize=covariance_matrix_font_size)

    def format_cov(c):
        if np.allclose(c, 0.0):
            return "0"
        elif np.allclose(c, 1.0):
            return "1"
        elif np.allclose(c, -1.0):
            return "-1"
        else:
            return "{0:0.2f}".format(c).replace("-0", "-").lstrip("0")

    def covcolor(c):
        return [1.0 - abs(c)]*3

    def textcolor_for_cov(c):
        return "white" if abs(c) > 0.5 else "black"

    for x in range(n):
        for y in range(n):
            blob(ax, x, y, abs(S[x, y]), covcolor(S[x,y]))
            ax.text(x, y, format_cov(S[x, y]),
                    fontsize=covariance_matrix_font_size,
                    horizontalalignment="center",
                    verticalalignment="center",
                    color=textcolor_for_cov(S[x, y]))

    return f, ax

def plot_scatter(X, Y):
    f, ax = base_plot(figsize=(8,4))
    ax.plot(X, Y, '.')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    return f, ax

def plot_distribution(X):
    f, ax = base_plot(figsize=(8,4))
    ax.hist(X, normed=True, bins=X.size/6)
    ax.set_ylabel("Frequency")
    return f, ax

def savefigure(f, name):
    f.savefig(name + ".pdf", bbox_inches="tight")
    f.savefig(name + ".png", bbox_inches="tight")

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
        f, ax = plot_covariance_matrix(np.corrcoef(bootstrap.internals, rowvar=0), ["fx", "fy", "ppx", "ppy", "ps"])
        savefigure(f, join(boot_dir, "covariance-interior"))

        # For each camera
        for (cam_number, cam) in enumerate(bootstrap.extract_cameras()):
            # Scatter and distribution plots
            f, ax = plot_scatter(cam[:,0], cam[:, 1])
            savefigure(f, join(boot_dir, "cam{}-xy".format(cam_number)))
            f, ax = plot_distribution(cam[:,2])
            ax.set_xlabel("Z")
            savefigure(f, join(boot_dir, "cam{}-z".format(cam_number)))

            # X, Y, Z and angles covariances matrices
            S = np.corrcoef(cam, rowvar=0)
            f, ax = plot_covariance_matrix(S[:3, :3], ["X", "Y", "Z"])
            savefigure(f, join(boot_dir, "covariance-cam{}-pos".format(cam_number)))
            f, ax = plot_covariance_matrix(S[3:, 3:], ["ang1", "ang2", "ang3"])
            savefigure(f, join(boot_dir, "covariance-cam{}-angles".format(cam_number)))

if __name__ == "__main__":
    main()
