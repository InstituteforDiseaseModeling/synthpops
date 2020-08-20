import os
import synthpops as sp
import sciris as sc
import pytest

def test_set_location_defaults_none():
    sp.config.set_location_defaults()
    assert sp.config.default_country == 'usa'
    assert sp.config.default_state == 'Washington'
    assert sp.config.default_location == 'seattle_metro'
    assert sp.config.default_sheet_name == 'United States of America'

def test_set_location_defaults_Senegal():
    sp.config.set_location_defaults('Senegal')
    assert sp.config.default_country == 'Senegal'
    assert sp.config.default_state == 'Dakar'
    assert sp.config.default_location == 'Dakar'
    assert sp.config.default_sheet_name is None

def test_set_location_defaults_usa():
    sp.config.set_location_defaults('usa')
    assert sp.config.default_country == 'usa'
    assert sp.config.default_state == 'Washington'
    assert sp.config.default_location == 'seattle_metro'
    assert sp.config.default_sheet_name == 'United States of America'

def test_FilePath_init_None():
    with pytest.raises(NotImplementedError):
        file_paths = sp.config.FilePaths()

def test_FilePath_no_country():
    with pytest.raises(NotImplementedError):
        file_paths = sp.config.FilePaths('seattle_metro', 'Washington')

def test_FilePath_country_only_usa():
    file_paths = sp.config.FilePaths(country='usa')
    assert file_paths is not None
    assert file_paths.country is not None
    assert file_paths.country == 'usa'
    assert file_paths.province is None
    assert file_paths.location is None
    assert file_paths.basedirs is not None
    assert len(file_paths.basedirs) == 1

def test_FilePath_country_only_Senegal():
    file_paths = sp.config.FilePaths(country='Senegal')
    assert file_paths is not None
    assert file_paths.country is not None
    assert file_paths.country == 'Senegal'
    assert file_paths.province is None
    assert file_paths.location is None
    assert file_paths.basedirs is not None
    assert len(file_paths.basedirs) == 1

def test_FilePath_invalid_country():
    with pytest.raises(FileNotFoundError):
        file_paths = sp.config.FilePaths(country='Germany')

# Test valid country state combinations
def test_FilePath_usa_Washington():
    file_paths = sp.config.FilePaths(province='Washington', country='usa')
    assert file_paths is not None
    assert file_paths.country is not None
    assert file_paths.country == 'usa'
    assert file_paths.province is not None
    assert file_paths.province == 'Washington'
    assert file_paths.location is None
    assert file_paths.basedirs is not None
    assert len(file_paths.basedirs) == 2

def test_FilePath_Senegal_Dakar():
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
def test_FilePath_usa_Florida():
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
def test_FilePath_usa_Washington_seattle_metro():
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
