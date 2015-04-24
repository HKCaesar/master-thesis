#!/usr/bin/env python3

import sys
import os

if __name__ == "__main__":

    command = sys.argv[1] if len(sys.argv) >= 2 else "all"

    if command == "all":
        # Compile
        os.system("mkdir -p build")
        os.system("cd build; cmake ..")
        os.system("make -C build/ -s")

        # Features analysis

        # Geosolve
        os.system("./build/geosolve ../data ../results/geosolve/model0 base")
        os.system("./build/geosolve ../data ../results/geosolve/model0 features")
        os.system("./build/geosolve ../data ../results/geosolve/model0 solve")
        os.system("./python/orthoimage.py ../data ../results/geosolve/model0")
