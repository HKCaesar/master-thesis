#!/usr/bin/env python3

import sys
import os
import os.path

def system(cmd):
    if os.system(cmd) != 0:
        raise RuntimeError("Command failed: {}".format(cmd))

def build():
    system("mkdir -p build")
    system("cd build; cmake ..")
    system("make -C build/ -s")

def features_analysis():
    system("./build/features_analysis ../data ../results")
    system("./python/features_analysis.py ../results/features_analysis")
    system("./python/features_table.py ../results/features_analysis")

def geosolve_dir(name):
    "Run all of a given geosolve result directory"

    project_dir = os.path.join("../results/geosolve", name)

    if os.path.exists(os.path.join(project_dir, "log.txt")):
        print("WARNING: {}/log.txt already exists.".format(name))

    def do_log(cmd):
        "Run cmd and log it to project_dir/log.txt"
        log = os.path.join(project_dir, "log.txt")
        system('printf "\n$ {}\n" 2>&1 | tee -a {}'.format(cmd, log))
        system("time -p {} 2>&1 | tee -a {}".format(cmd, log))

    system("mkdir -p {}".format(project_dir))
    do_log("./build/geosolve ../data {} base_{}".format(project_dir, name))
    do_log("./build/geosolve ../data {} features".format(project_dir))
    do_log("./build/geosolve ../data {} solve".format(project_dir))
    do_log("./python/orthoimage.py ../data {}".format(project_dir))

def geosolve():
    if len(sys.argv) < 3:
        raise RuntimeError("geosolve command requires project_dir name")
    build()
    geosolve_dir(sys.argv[2])

def all():
    build()
    features_analysis()
    geosolve_all("model0")
    geosolve_all("model0_200")

# TODO add dtm.py and potree.py to run_all.py

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) >= 2 else "all"
    try:
        fct = globals()[cmd]
        print("Running target {}".format(cmd))
        fct()
    except KeyError:
        print("Command not found: {}".format(cmd))

