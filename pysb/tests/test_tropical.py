from pysb.examples.tyson_oscillator import model as tyson
from pysb.tools.tropicalize import tropicalize
from numpy import linspace
from nose.tools import *
import traceback
import os
import importlib

t = linspace(0, 100, 10001)
tropical = tropicalize(tyson, t, ignore=1, epsilon=0.1, rho=1, verbose=False)

def test_slaves():
    assert_equal(tropical.slaves, ['s0','s1','s5'])
    