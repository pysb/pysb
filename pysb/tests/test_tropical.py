from pysb.examples.tyson_oscillator import model
from pysb.tools.tropicalize import Tropical
from numpy import linspace
from nose.tools import *
import traceback
import os
import importlib

t = linspace(0, 100, 100)
tro = Tropical(model)
tro.tropicalize(t)

def test_slaves():
    assert_equal(tro.passengers, [0,1,4])
