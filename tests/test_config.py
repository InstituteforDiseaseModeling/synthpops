"""
Test config methods.
"""
import synthpops as sp
import sciris as sc
import tempfile
import os
import pytest


@pytest.fixture(scope="function")
def test_cleanup_set_default():
    yield
    sp.reset_settings(sp.default_data['defaults'])
    sp.set_datadir(sp.default_datadir_path())
    assert sp.settings.country_location == 'usa', 'Reset failed.'
    assert sp.settings.datadir == sp.default_datadir_path(), 'Reset failed for datadir.'


def test_version():
    sp.logger.info("Testing that version info is returned.")
    sp.version_info()
    print(sp.settings.country_location)


def test_metadata():
    sp.logger.info("Testing that the version is greater than 1.5.0")
    pop = sp.Pop(n=100)
    assert sc.compareversions(pop.version, '1.5.0') == 1 # to check that the version of synthpops is higher than 1.5.0


def test_nbrackets(test_cleanup_set_default):
    sp.logger.info("Testing that nbrackets can be set outside of the recommended range and warning message returned.")

    nbrackets = max(min(sp.settings.valid_nbracket_ranges), 2)  # make sure new nbrackets is at least 2
    sp.set_nbrackets(n=nbrackets - 1)  # testing a valid outside the range currently supported.
    assert nbrackets - 1 == sp.settings.nbrackets,f'Check failed. sp.settings.nbrackets not set to {nbrackets-1} outside of the official supported range.'
    print(f'Check passed. synthpops.settings.nbrackets updated to {nbrackets-1} outside of the official supported range.')

    sp.set_nbrackets(n=nbrackets)  # resetting to the default value
    assert nbrackets == sp.settings.nbrackets,f'Check failed. sp.settings.nbrackets not reset to {nbrackets}.'
    print(f'Check passed. Reset default synthpops.settings.nbrackets.')

    nbrackets = max(sp.settings.valid_nbracket_ranges)
    sp.set_nbrackets(n=nbrackets)  # set to different value
    assert nbrackets == sp.settings.nbrackets, f'Check failed. sp.default_config.nbrackets not reset to {nbrackets}.'
    print(f'Check passed. Reset default synthpops.default_config.nbrackets.')
    test_cleanup_set_default


def test_validate_datadir():
    sp.logger.info("Testing that synthpops.settings.datadir can be found.")
    sp.validate_datadir()
    sp.validate_datadir(verbose=False)
    sp.reset_settings_by_key('datadir', 0)
    with pytest.raises(FileNotFoundError):
        sp.validate_datadir()
    sp.set_datadir(sp.default_datadir_path())


def test_set_datadir(test_cleanup_set_default):
    sp.logger.info("Testing set_datadir still works in essence.")
    datadir = sp.set_datadir('not_spdatadir')

    assert datadir != sp.default_datadir_path(), "Check failed. datadir set still equal to default sp.setting_config.datadir"
    print("Check passed. New datadir set different from synthpops default.")

    datadir = sp.set_datadir(sp.default_datadir_path())
    assert datadir == sp.default_datadir_path() and datadir == sp.settings.datadir, "Check failed. datadir did not reset to default sp.settings.datadir"
    print("Check passed. datadir reset to synthpops default and sp.settings.datadir reset.")

    newpath = tempfile.mkdtemp()
    datadir = sp.set_datadir(newpath)
    assert datadir == sp.settings.datadir and datadir == newpath, "Check failed. datadir did not set to new path"
    print("Check passed. datadir reset to synthpops default and sp.default_config.datadir reset.")
    if os.path.exists(newpath):
        os.removedirs(newpath)
    test_cleanup_set_default


def test_log_level():
    """Test resetting the log level"""
    sp.logger.setLevel('DEBUG')
    pars = sc.objdict(n=100)
    pop = sp.Pop(**pars)
    sp.logger.setLevel('INFO')  # need to reset logger level - this changes a synthpops setting


def test_set_location_defaults():
    """Testing that sp.set_location_defaults() works as expected"""
    sp.set_location_defaults('Senegal')
    assert sp.settings.country_location == 'Senegal', f"Check failed. sp.settings.country_location did not set to 'Senegal'."
    print(f"Check passed. sp.settings.country_location set to 'Senegal'.")
    sp.set_location_defaults('defaults')
    assert sp.settings.country_location == 'usa', f"Check failed. sp.settings.country_location did not set to 'usa'."
    print(f"Check passed. sp.settings.country_location set to 'usa'.")


def test_pakistan():
    """
    Testing that we get a warning message and that default information is still
    set to usa, however users can manually set their own default location to
    use for data gaps (we're not supplying all data).
    """
    sp.set_location_defaults('Pakistan')
    assert sp.settings.country_location == 'usa', 'Check failed. Defaults are not set as expected.'
    print('Defaults still set to default usa location information.')

    # this data does not actually exist yet
    new_defaults = dict(location=None, state_location=None, country_location='Pakistan')
    sp.reset_settings(new_defaults)
    assert sp.settings.country_location == 'Pakistan', 'Check failed. Default location information did not update to Pakistan.'
    print('Defaults update to Pakistan location information.')

    # reset the global parameters
    sp.reset_settings(sp.default_data['defaults'])
    assert sp.settings.country_location == 'usa', 'Reset failed.'
    print('Reset to original defaults.')


if __name__ == '__main__':

    test_version()
    test_metadata()
    test_nbrackets()
    test_validate_datadir()
    test_set_datadir()
    test_log_level()
    test_set_location_defaults()
    test_pakistan()
