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

    nbrackets = max(min(sp.default_config.valid_nbracket_ranges), 2)  # make sure new nbrackets is at least 2
    sp.set_nbrackets(n=nbrackets - 1)  # testing a valid outside the range currently supported.
    assert nbrackets - 1 == sp.default_config.nbrackets,f'Check failed. sp.config.nbrackets not set to {nbrackets-1} outside of the official supported range.'
    print(f'Check passed. synthpops.default_config.nbrackets updated to {nbrackets-1} outside of the official supported range.')

    sp.set_nbrackets(n=nbrackets)  # resetting to the default value
    assert nbrackets == sp.default_config.nbrackets,f'Check failed. sp.default_config.nbrackets not reset to {nbrackets}.'
    print(f'Check passed. Reset default synthpops.default_config.nbrackets.')


def test_validate_datadir():
    sp.logger.info("Testing that synthpops.datadir can be found.")
    sp.validate_datadir()
    sp.validate_datadir(verbose=False)


def test_set_datadir():
    sp.logger.info("Testing set_datadir still works in essence.")
    datadir = sp.set_datadir('not_spdatadir')

    print('datadir', datadir)
    print('sp.default_config.datadir', sp.default_config.datadir)
    print()
    # assert datadir != sp.datadir, "Check failed. datadir set still equal to default sp.datadir"
    assert datadir != sp.default_datadir_path(), "Check failed. datadir set still equal to default sp.default_config.datadir"
    print("New datadir set different from synthpops default.")

    datadir = sp.set_datadir(sp.default_datadir_path())
    assert datadir == sp.default_datadir_path(), "Check failed. datadir did not reset to default sp.datadir"
    print("datadir reset to synthpops default.")

    assert datadir == sp.defaults.default_config.datadir, "Check 2 failed."
    print('datadir did reset everywhere')

    # print(sp.default_config.datadir)


def test_log_level():
    """Test resetting the log level"""
    sp.logger.setLevel('DEBUG')
    pars = sc.objdict(n=100)
    pop = sp.Pop(**pars)
    sp.logger.setLevel('INFO')  # need to reset logger level - this changes a synthpops setting


def test_set_location_defaults():
    """Testing that sp.set_location_defaults() works as expected"""
    sp.set_location_defaults('Senegal')
    assert sp.default_config.country_location == 'Senegal'
    sp.set_location_defaults('defaults')
    assert sp.default_config.country_location == 'usa'


def test_pakistan():
    sp.set_location_defaults('Pakistan')


if __name__ == '__main__':

    # test_version()
    # test_metadata()
    # test_nbrackets()
    # test_validate_datadir()
    test_set_datadir()
    # test_log_level()

    # print(sp.default_config)
    # sp.reset_default_config('country_location', 'Senegal')

    test_set_location_defaults()

    print(sp.default_config)
    print()
    test_pakistan()
    print(sp.default_config)

