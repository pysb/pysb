from pysb.examples.tyson_oscillator import model
from pysb.tools import tropicalize_fun
from numpy import linspace
from nose.tools import *
import traceback
import os
import importlib

t = linspace(0, 100, 10001)


def test_slaves():
    assert_equal(tropicalize_fun.find_slaves(model, t, epsilon=2, p=None), [0,1,4])
