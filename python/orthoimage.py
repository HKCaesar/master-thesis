#!/usr/bin/env python3

import sys
import os.path
import json

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: ./orthoimage.py <data_root> <model.json>")
        sys.exit(-1)

    data_root = sys.argv[1]
    model_filename = sys.argv[2]

    project = json.load(open(model_filename))

    print(project["data"]["left"])
