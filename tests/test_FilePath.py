import datetime
import inspect
import unittest
import utilities
import tempfile
import os
import shutil
import sys
import synthpops as sp
from synthpops import cfg
#import sciris as sc
import pytest


class TestFilePath(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rootDir = tempfile.TemporaryDirectory().name
        cls.initia_default_dir = cfg.datadir
        os.makedirs(os.path.join(cls.rootDir, "data"), exist_ok=True)
        cls.dataDir = os.path.join(cls.rootDir, "data")

    @classmethod
    def tearDownClass(cls) -> None:
        #cls.copy_output()
        cfg.set_datadir(cls.initia_default_dir, ["demographics","contact_matrices_152_countries"])
        cfg.set_location_defaults(country="default")
        for d in [cls.rootDir]:
            shutil.rmtree(d, ignore_errors=True)

    def test_FilePath_init_None(cls):
        with pytest.raises(NotImplementedError):
            file_paths = sp.config.FilePaths()

    def test_check_defaults(cls):
        # defaults ae set when config.py is loaded. check for the defaults
        # should be the same as usa
        assert sp.config.default_country == 'usa'
        assert sp.config.default_state == 'Washington'
        assert sp.config.default_location == 'seattle_metro'
        assert sp.config.default_sheet_name == 'United States of America'

    def test_FilePath_no_country(cls):
        with pytest.raises(NotImplementedError):
            file_paths = sp.config.FilePaths('seattle_metro', 'Washington')

    def test_set_location_defaults_none(cls):
        sp.config.set_location_defaults()
        assert sp.config.default_country == 'usa'
        assert sp.config.default_state == 'Washington'
        assert sp.config.default_location == 'seattle_metro'
        assert sp.config.default_sheet_name == 'United States of America'

    def test_set_location_defaults_Senegal(cls):
        sp.config.set_location_defaults('Senegal')
        assert sp.config.default_country == 'Senegal'
        assert sp.config.default_state == 'Dakar'
        assert sp.config.default_location == 'Dakar'
        assert sp.config.default_sheet_name is None

    def test_set_location_defaults_usacls(cls):
        sp.config.set_location_defaults('usa')
        assert sp.config.default_country == 'usa'
        assert sp.config.default_state == 'Washington'
        assert sp.config.default_location == 'seattle_metro'
        assert sp.config.default_sheet_name == 'United States of America'


    def test_FilePath_country_only_usa(cls):
        file_paths = sp.config.FilePaths(country='usa')
        assert file_paths is not None
        assert file_paths.country is not None
        assert file_paths.country == 'usa'
        assert file_paths.province is None
        assert file_paths.location is None
        assert file_paths.basedirs is not None
        assert len(file_paths.basedirs) == 1

    def test_FilePath_country_only_Senegal(cls):
        file_paths = sp.config.FilePaths(country='Senegal')
        assert file_paths is not None
        assert file_paths.country is not None
        assert file_paths.country == 'Senegal'
        assert file_paths.province is None
        assert file_paths.location is None
        assert file_paths.basedirs is not None
        assert len(file_paths.basedirs) == 1

    def test_FilePath_invalid_country(cls):
        with pytest.raises(FileNotFoundError):
            file_paths = sp.config.FilePaths(country='Germany')

    # Test valid country state combinations
    def test_FilePath_usa_Washington(cls):
        file_paths = sp.config.FilePaths(province='Washington', country='usa')
        assert file_paths is not None
        assert file_paths.country is not None
        assert file_paths.country == 'usa'
        assert file_paths.province is not None
        assert file_paths.province == 'Washington'
        assert file_paths.location is None
        assert file_paths.basedirs is not None
        assert len(file_paths.basedirs) == 2

    def test_FilePath_Senegal_Dakar(cls):
        file_paths = sp.config.FilePaths(province='Dakar', country='Senegal')
        assert file_paths is not None
        assert file_paths.country is not None
        assert file_paths.country == 'Senegal'
        assert file_paths.province is not None
        assert file_paths.province == 'Dakar'
        assert file_paths.location is None
        assert file_paths.basedirs is not None
        assert len(file_paths.basedirs) == 2

    # invalid state
    def test_FilePath_usa_Florida(cls):
        file_paths = sp.config.FilePaths(province='Florida', country='usa')
        # note because usa is valid and Florida is no, searches will be at USA level
        assert file_paths is not None
        assert file_paths.country is not None
        assert file_paths.country == 'usa'
        assert file_paths.province is not None
        assert file_paths.province == 'Florida'
        assert file_paths.location is None
        assert file_paths.basedirs is not None
        assert len(file_paths.basedirs) == 1

    # test valid locations
    def test_FilePath_usa_Washington_seattle_metro(cls):
        file_paths = sp.config.FilePaths(location='seattle_metro', province='Washington', country='usa')
        assert file_paths is not None
        assert file_paths.country is not None
        assert file_paths.country == 'usa'
        assert file_paths.province is not None
        assert file_paths.province == 'Washington'
        assert file_paths.location is not None
        assert file_paths.location == 'seattle_metro'
        assert file_paths.basedirs is not None
        assert len(file_paths.basedirs) == 3

    def test_set_alt_location(cls):
        current_alt_location = sp.config.alt_location
        sp.config.set_alt_location('Yakima', 'Washington', 'usa')
        assert sp.config.alt_location.country_location == 'usa'
        assert sp.config.alt_location.state_location == 'Washington'
        assert sp.config.alt_location.location == 'Yakima'
        sp.config.alt_location = current_alt_location

    def test_FilePaths_usa_Washington_yakima_with_alt_location(cls):
        current_alt_location = sp.config.alt_location
        sp.config.set_alt_location(location='seattle_metro', state_location='Washington', country_location='usa')
        file_paths = sp.config.FilePaths(location='yakima',province='Washington', country='usa')
        assert file_paths is not None
        assert file_paths.country is not None
        assert file_paths.country == 'usa'
        assert file_paths.province is not None
        assert file_paths.province == 'Washington'
        assert file_paths.location is not None
        assert file_paths.location == 'yakima'
        assert file_paths.alt_country is not None
        assert file_paths.alt_country == 'usa'
        assert file_paths.alt_province is not None
        assert file_paths.alt_province == 'Washington'
        assert file_paths.alt_location is not None
        assert file_paths.alt_location == 'seattle_metro'
        assert file_paths.basedirs is not None
        assert len(file_paths.basedirs) == 5
        sp.config.alt_location = current_alt_location


