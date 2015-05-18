#!/usr/bin/env python3

import sys
import os
import os.path
from os.path import join
import stat

def system(cmd):
    if os.system(cmd) != 0:
        raise RuntimeError("Command failed: {}".format(cmd))

view_dtm_script = """
#!/bin/bash 
i3-msg workspace 2
firefox http://0.0.0.0:8000/potree/examples/dtm.xyz.html
konsole --nofork -e /bin/bash -c "python3 -m http.server"
"""

if __name__ == "__main__":
    dtmdir = sys.argv[1]

    if not os.path.exists(join(dtmdir, "dtm.xyz")):
        print("Error, {} does not exist.".format(join(dtmdir, "dtm.xyz")))
        sys.exit(-1)

    potree_dir = "../PotreeConverter/build/PotreeConverter"

    # Copy potree directories
    system("mkdir -p {}".format(join(dtmdir, "potree")))
    system("cp -r ../potree/build -t {}".format(join(dtmdir, "potree")))
    system("cp -r ../potree/resources -t {}".format(join(dtmdir, "potree")))
    system("cp -r ../potree/libs -t {}".format(join(dtmdir, "potree")))

    # Call potree converter
    system("cd {}; ./PotreeConverter {} -o {} -p --color-range 0 255 --input-format xyzrgb"
            .format(potree_dir, os.path.abspath(join(dtmdir, "dtm.xyz")), os.path.abspath(join(dtmdir, "potree"))))

    # Create view_dtm.sh
    view_dtm_file = join(dtmdir, "view_dtm.sh")
    with open(view_dtm_file, "w") as f:
        f.write(view_dtm_script)

    # chmod +x
    os.chmod(view_dtm_file, os.stat(view_dtm_file).st_mode | stat.S_IEXEC)
