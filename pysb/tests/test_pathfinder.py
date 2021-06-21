from pysb.pathfinder import get_path, set_path, list_programs, _path_config
import os


def test_get_set_path():
    bng_path = get_path('bng')
    assert os.path.exists(bng_path)
    set_path('bng', bng_path)


def test_list_programs():
    assert list_programs().keys() == _path_config.keys()
