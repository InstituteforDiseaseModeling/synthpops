
"""
Test config methods.
"""
import synthpops as sp
import sciris as sc
import settings


def test_version():
    sp.logger.info("Testing that version info is returned.")
    sp.version_info()


def test_metadata():
    sp.logger.info("Testing that the version is greater than 1.5.0")
    pop = sp.Pop(n=100)
    assert sc.compareversions(pop.version, '1.5.0') == 1 # to check that the version of synthpops is higher than 1.5.0


def test_log_level():
    """Test resetting the log level"""
    sp.logger.setLevel('DEBUG')
    pars = sc.objdict(n=settings.pop_sizes.small)
    pop = sp.Pop(**pars)
    sp.logger.setLevel('INFO')  # need to reset logger level - this changes a synthpops setting


if __name__ == '__main__':
    test_version()
    test_metadata()
    test_log_level()
