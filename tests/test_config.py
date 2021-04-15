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
    assert nbrackets - 1 == sp.config.nbrackets,f'Check failed. sp.config.nbrackets not set to {nbrackets-1} outside of the official supported range.'
    print(f'Check passed. synthpops.nbrackets.config updated to {nbrackets-1} outside of the official supported range.')

    sp.set_nbrackets(n=nbrackets)  # resetting to the default value
    assert nbrackets == sp.config.nbrackets,f'Check failed. sp.config.nbrackets not reset to {nbrackets}.'
    print(f'Check passed. Reset default synthpops.config.nbrackets.')


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


def test_log_level():
    """Test resetting the log level"""
    sp.logger.setLevel('DEBUG')
    pars = sc.objdict(n=100)
    pop = sp.Pop(**pars)
    sp.logger.setLevel('INFO')  # need to reset logger level - this changes a synthpops setting


def print_default():
    global default_country
    print("\nglobal default_country in config.py is ", default_country)


def test_set_location_defaults():
    """Testing that sp.set_location_defaults() works as expected"""
    sp.set_location_defaults('Senegal')


# import synthpops as sp
# from synthpops import default_country
# sp.print_default()
# print("global:", default_country)
# sp.set_location_defaults('Senegal')
# sp.print_default()
# print("global:", default_country)


if __name__ == '__main__':

    # test_version()
    # test_metadata()
    # test_nbrackets()
    # test_validate_datadir()
    # test_set_datadir()
    # test_log_level()

    print(sp.default_country)
    print(sp.defaults_config)

    print()
    # sp.reset_defaults_config('a', 9)
    sp.reset_defaults_config('default_country', 'Senegal')

    print(sp.defaults_config)
    print(sp.defaults.defaults_config)

    test_set_location_defaults()

    print(sp.default_country)
    print(sp.config.default_country)
