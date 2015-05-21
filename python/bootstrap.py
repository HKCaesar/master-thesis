#!/usr/bin/env python3

import sys
import os.path
import numpy as np
from project import Project
from IPython import embed

def covariance_externals(bootstrap):
    number_of_samples, number_of_cameras, six = bootstrap.externals.shape
    assert number_of_samples == bootstrap.number_of_samples
    assert six == 6

    cov_list = []
    # For each camera
    for cam_number in range(number_of_cameras):
        cam = bootstrap.externals[:, cam_number, :]

        # Covariance matrix
        cov_list.append(np.cov(cam, rowvar=0))
    return cov_list

def covariance_internals(bootstrap):
    number_of_samples, four = bootstrap.internals.shape
    assert number_of_samples == bootstrap.number_of_samples
    assert four == 4

    return np.cov(bootstrap.internals, rowvar=0)

def main():
    if len(sys.argv) < 2:
        print("Usage: ./bootstrap.py <project_dir>")
        sys.exit(-1)

    project_dir = sys.argv[1]
    project = Project(os.path.join(project_dir, "project.json"))

    for bootstrap in project.bootstraps:
        # Covariance matrices
        ext_covs = covariance_externals(bootstrap)
        int_cov = covariance_internals(bootstrap)
        embed()

if __name__ == "__main__":
    main()
