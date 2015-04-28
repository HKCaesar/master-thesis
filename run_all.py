#!/usr/bin/env python3

import sys
import os
import os.path

def system(cmd):
    if os.system(cmd) != 0:
        raise RuntimeError("Command failed: {}".format(cmd))

def do(cmd, project_dir):
    cmd = cmd.format(project_dir)
    log = os.path.join(project_dir, "log.txt")
    system('printf "\n$ {}\n" 2>&1 | tee -a {}'.format(cmd, log))
    system("time -p {} 2>&1 | tee -a {}".format(cmd, log))

def build():
    system("mkdir -p build")
    system("cd build; cmake ..")
    system("make -C build/ -s")

def features_analysis():
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

def all():
    build()
    features_analysis()
    geosolve()

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) >= 2 else "all"
    try:
        fct = globals()[cmd]
        print("Running target {}".format(cmd))
        fct()
    except KeyError:
        print("Command not found: {}".format(cmd))

