#!/usr/bin/env python3

import sys
import os.path
import numpy as np
from project import Project

def main():
    if len(sys.argv) < 2:
        print("Usage: ./bootstrap.py <project_dir>")
        sys.exit(-1)

    project_dir = sys.argv[1]
    project = Project(os.path.join(project_dir, "project.json"))

if __name__ == "__main__":
    main()
