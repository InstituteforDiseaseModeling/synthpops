import os
import synthpops as sp
import sciris as sc
import pytest

datadir = sp.datadir
location = 'seattle_metro'
state_location = 'Washington'
country_location = 'usa'


def test_get_gender_fraction_by_age_path():
    dat_file = sp.get_gender_fraction_by_age_path(datadir=datadir, location=location, state_location=state_location,
                                                  country_location=country_location)
    assert dat_file is not None


def test_read_gender_fraction_by_age():
    gender_data_file_path = os.path.join(datadir, 'demographics', 'contact_matrices_152_countries',
                                         country_location, state_location, 'age_distributions',
                                         'seattle_metro_gender_fraction_by_age_bracket.dat')
    dict = sp.read_gender_fraction_by_age_bracket(datadir, state_location=state_location,
                                                  country_location=country_location,
                                                  file_path=gender_data_file_path, use_default=True)
    assert dict is not None


def test_get_age_bracket_distr_path():
    dat_file = sp.get_age_bracket_distr_path(datadir=datadir, location=location, state_location=state_location,
                                             country_location=country_location)
    assert dat_file is not None


def test_get_household_size_distr_path():
    dat_file = sp.get_household_size_distr_path(datadir=datadir, location=location, state_location=state_location,
                                                country_location=country_location)
    assert dat_file


def test_get_head_age_brackets_path():
    dat_file = sp.get_head_age_brackets_path(datadir=datadir, state_location=state_location,
                                             country_location=country_location)
    assert dat_file is not None


def test_get_household_head_age_by_size_path():
    dat_file = sp.get_household_head_age_by_size_path(datadir=datadir, state_location=state_location,
                                                      country_location=country_location)
    assert dat_file is not None


def test_get_head_age_by_size_path():
    hha_by_size = sp.get_household_head_age_by_size_path(datadir, state_location=state_location,
                                                         country_location=country_location)
    assert hha_by_size is not None


def test_get_school_enrollment_rates():
    school_enrollement_file_path = os.path.join(datadir, 'demographics', 'contact_matrices_152_countries',
                                                country_location,
                                                state_location,
                                                'household_head_age_and_size_count.dat')
    dict = sp.get_school_enrollment_rates(datadir, location=location, state_location=state_location,
                                          country_location=country_location,
                                          file_path=school_enrollement_file_path, use_default=True)
    assert dict is not None


def test_get_contact_matrix():
    setting_code = 'H'
    sheet_name = 'United States of America'
    data_matrix = sp.get_contact_matrix(datadir, setting_code, sheet_name=sheet_name)
    assert len(data_matrix) == 16
