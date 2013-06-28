from pysb.examples.tyson_oscillator import model as tyson
from pysb.tools.tropicalize import tropicalize
from numpy import linspace
from nose.tools import *
import traceback
import os
import importlib

def test_slaves():
    t = linspace(0, 100, 10001)
    tropical = tropicalize(tyson, t, ignore=15, epsilon=1e-6, rho=1, verbose=False)
    assert_equal(tropical.slaves, ['s0','s1','s5'])
    