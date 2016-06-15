import os
import csv
import re
import numpy as np
from collections import OrderedDict


def listdir_fullpath(d):
    """Return a list of path of files in directory

       Keyword arguments:
       d -- path to directory
    """
    return [os.path.join(d, f) for f in os.listdir(d)]


def read_pars(par_path):
    """Return a list of parameter values from csv file

       keyword arguments:
       par_path -- path to parameter file
    """
    f = open(par_path)
    data = csv.reader(f)
    param = [float(d[1]) for d in data]
    return param


def parse_name(spec):
    """Returns parsed name of species

        keyword arguments:
        spec -- species name to parse
    """
    m = spec.monomer_patterns
    lis_m = []
    name_counts = OrderedDict()
    parsed_name = ''
    for i in range(len(m)):
        tmp_1 = str(m[i]).partition('(')
        tmp_2 = re.findall(r"(?<=\').+(?=\')", str(m[i]))

        if not tmp_2:
            lis_m.append(tmp_1[0])
        else:
            lis_m.append(''.join([tmp_1[0], tmp_2[0]]))

    for name in lis_m:
        name_counts[name] = lis_m.count(name)

    for sp, counts in name_counts.items():
        if counts == 1:
            parsed_name += sp + '_'
        else:
            parsed_name += str(counts) + sp + '_'
    return parsed_name[:len(parsed_name)-1]


def _find_nearest_zero(array):
    idx = np.nanargmin(np.abs(array))
    return array[idx]
