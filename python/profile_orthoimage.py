#!/usr/bin/env python3

from pycallgraph import PyCallGraph
from pycallgraph import Config
from pycallgraph import GlobbingFilter
from pycallgraph.output import GraphvizOutput

from orthoimage import main

graphviz = GraphvizOutput(output_file='pycallgraph_orthoimage.png')
config = Config(max_depth=10)
config.trace_filter = GlobbingFilter(exclude=[
    'pycallgraph.*',
    'posixpath.*',
    'os.*',
    'contextlib',
])

with PyCallGraph(output=graphviz, config=config):
    main()
