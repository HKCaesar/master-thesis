#!/usr/bin/env python3

import sys
import os
import os.path
from os.path import join
import shutil
import stat

def system(cmd):
    if os.system(cmd) != 0:
        raise RuntimeError("Command failed: {}".format(cmd))

view_dtm_script = """
#!/bin/bash 

i3-msg workspace 2
firefox -new-window http://0.0.0.0:8000/potree/examples/dtm.xyz.html
python3 -m http.server

"""

if __name__ == "__main__":
    dtmdir = sys.argv[1] if len(sys.argv) >= 2 else "."

    if not os.path.exists(join(dtmdir, "dtm.xyz")):
        print("Error, {} does not exist.".format(join(dtmdir, "dtm.xyz")))
        sys.exit(-1)

    # Call potree converter
    potree_dir = "../PotreeConverter/build/PotreeConverter"
    system("cd {}; ./PotreeConverter {} -o {} -p --color-range 0 255 --input-format xyzrgb"
            .format(potree_dir, os.path.abspath(join(dtmdir, "dtm.xyz")), os.path.abspath(join(dtmdir, "potree"))))

    # Copy potree directories
    system("cp -r ../potree/build {}".format(join(dtmdir, "potree")))
    system("cp -r ../potree/resources {}".format(join(dtmdir, "potree")))
    system("cp -r ../potree/libs {}".format(join(dtmdir, "potree")))

    system("cd {}; ./PotreeConverter {} -o {} -p --color-range 0 255 --input-format xyzrgb"
            .format(potree_dir, os.path.abspath(join(dtmdir, "dtm.xyz")), os.path.abspath(join(dtmdir, "potree"))))

    # Create view_dtm.sh
    view_dtm_file = join(dtmdir, "view_dtm.sh")
    with open(view_dtm_file, "w") as f:
        f.write(view_dtm_script)

    # chmod +x
    os.chmod(view_dtm_file, os.stat(view_dtm_file).st_mode | stat.S_IEXEC)



