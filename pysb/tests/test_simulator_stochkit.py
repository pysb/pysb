from pysb.export.stochkit import StochKitExporter
from pysb.examples import robertson


def test_stochkit_export():
    StochKitExporter(robertson.model).export()
