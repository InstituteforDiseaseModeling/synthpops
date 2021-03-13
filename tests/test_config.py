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

def test_nbrackets():
    sp.logger.info("Testing that nbrackets can be set outside of the recommended range and warning message returned.")
    nbrackets = max(min(sp.config.valid_nbracket_ranges), 2)  # make sure new nbrackets is at least 2
    sp.set_nbrackets(n=nbrackets - 1)  # testing a valid outside the range currently supported.


def test_validate_datadir():
    sp.logger.info("Testing that synthpops.datadir can be found.")
    sp.validate_datadir()
    sp.validate_datadir(verbose=False)


def test_set_datadir():
    sp.logger.info("Testing set_datadir still works in essence.")
    datadir = sp.set_datadir('not_spdatadir')
    assert datadir != sp.datadir, "Check failed. datadir set still equal to default sp.datadir"
    print("New datadir set different from synthpops default.")

    datadir = sp.set_datadir(sp.datadir)
    assert datadir == sp.datadir, "Check failed. datadir did not reset to default sp.datadir"
    print("datadir reset to synthpops default.")


if __name__ == '__main__':

    test_version()
    test_metadata()
    test_nbrackets()
    test_validate_datadir()
    test_set_datadir()

