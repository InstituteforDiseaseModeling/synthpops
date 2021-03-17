
"""
Test config methods.
"""
import synthpops as sp
import sciris as sc


def test_version():
    sp.logger.info("Testing that version info is returned.")
    sp.version_info()


def test_metadata():
    sp.logger.info("Testing that the version is greater than 1.5.0")
    pop = sp.Pop(n=100)
    assert sc.compareversions(pop.version, '1.5.0') == 1 # to check that the version of synthpops is higher than 1.5.0


if __name__ == '__main__':
    test_version()
    test_metadata()