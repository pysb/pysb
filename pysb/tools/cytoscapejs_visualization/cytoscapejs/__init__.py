# -*- coding:utf-8 -*-

"""
A package for rendering embedded Cytoscape.js object for IPython Notebook.

"""

from pysb.tools.cytoscapejs_visualization.cytoscapejs.viewer import *
import os
JS_LOADER_FILE = "loader.js"

def init():
    from IPython.core.display import display, Javascript

    path = os.path.abspath(os.path.dirname(__file__)) + "/" + JS_LOADER_FILE
    js_loader = open(path).read()
    return display(Javascript(js_loader))


# Load Cytoscape.js and dependent libraries
init()
