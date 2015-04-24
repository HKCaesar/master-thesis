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

if __name__ == "__main__":

    command = sys.argv[1] if len(sys.argv) >= 2 else "all"

    if command == "all":
        # Compile
        system("mkdir -p build")
        system("cd build; cmake ..")
        system("make -C build/ -s")

        # Features analysis

        # Geosolve
        project_dir = "../results/geosolve/model0"
        system("mkdir -p {}".format(project_dir))
        do("./build/geosolve ../data {} base", project_dir)
        do("./build/geosolve ../data {} features", project_dir)
        do("./build/geosolve ../data {} solve", project_dir)
        do("./python/orthoimage.py ../data {}", project_dir)
