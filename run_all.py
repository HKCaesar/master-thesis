#!/usr/bin/env python3

import sys
import os
import os.path

def system(command):
    if os.system(command) != 0:
        raise RuntimeError("Command failed: {}".format(command))

def do(command, project_dir):
    command = command.format(project_dir)
    log = os.path.join(project_dir, "log.txt")
    system('printf "\n$ {}\n" 2>&1 | tee -a {}'.format(command, log))
    system("time -p {} 2>&1 | tee -a {}".format(command, log))

def build():
    system("mkdir -p build")
    system("cd build; cmake ..")
    system("make -C build/ -s")

def features():
    system("./build/features_analysis ../data ../results")
    system("./python/features_analysis.py ../results/features_analysis")
    system("./python/features_table.py ../results/features_analysis")

def geosolve():
    project_dir = "../results/geosolve/model0"
    system("mkdir -p {}".format(project_dir))
    do("./build/geosolve ../data {} base", project_dir)
    do("./build/geosolve ../data {} features", project_dir)
    do("./build/geosolve ../data {} solve", project_dir)
    do("./python/orthoimage.py ../data {}", project_dir)

if __name__ == "__main__":

    command = sys.argv[1] if len(sys.argv) >= 2 else "all"

    if command == "all":
        build()
        features()
        geosolve()
    elif command == "geosolve":
        build()
        geosolve()
