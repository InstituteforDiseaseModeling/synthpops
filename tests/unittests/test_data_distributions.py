import os
import pytest
import synthpops as sp
from synthpops import data_distributions as spdd

datadir = sp.settings.datadir
location = 'seattle_metro'
state_location = 'Washington'
country_location = 'usa'
# location = 'Dakar'
# state_location = 'Dakar'
# country_location = 'Senegal'


# def test_get_gender_fraction_by_age_path():
#     dat_file, file_path = spdd.get_gender_fraction_by_age_path(location=location, state_location=state_location,
#                                                   country_location=country_location)
#     assert file_path is not None


# def test_read_gender_fraction_by_age():
#     gender_data_file_path = os.path.join(datadir,
#                                          country_location, state_location, 'age_distributions',
#                                          'seattle_metro_gender_fraction_by_age_bracket.dat')
#     dict = spdd.read_gender_fraction_by_age_bracket(datadir, state_location=state_location,
#                                                   country_location=country_location,
#                                                   file_path=gender_data_file_path, use_default=True)
#     assert dict is not None


@pytest.mark.skip(reason="Path methods were removed; we're keeping this around for reference.")
@pytest.mark.parametrize("nbrackets", [None, "16", "18", "20"])
def test_get_age_bracket_distr_path(nbrackets):
    dat_file = spdd.get_age_bracket_distr_path(datadir=datadir, location=location, state_location=state_location,
                                             country_location=country_location, nbrackets=nbrackets)
    print(dat_file)
    assert dat_file is not None
    assert os.path.exists(dat_file)


@pytest.mark.skip(reason="Path methods were removed; we're keeping this around for reference.")
def test_get_household_size_distr_path():
    dat_file = spdd.get_household_size_distr_path(datadir=datadir, location=location, state_location=state_location,
                                                country_location=country_location)
    assert dat_file


@pytest.mark.skip(reason="Path methods were removed; we're keeping this around for reference.")
def test_get_head_age_brackets_path():
    dat_file = spdd.get_head_age_brackets_path(datadir=datadir, state_location=state_location,
                                             country_location=country_location)
    assert dat_file is not None


@pytest.mark.skip(reason="Path methods were removed; we're keeping this around for reference.")
def test_get_household_head_age_by_size_path():
    dat_file = spdd.get_household_head_age_by_size_path(datadir=datadir, state_location=state_location,
                                                      country_location=country_location)
    assert dat_file is not None


@pytest.mark.skip(reason="Path methods were removed; we're keeping this around for reference.")
def test_get_head_age_by_size_path():
    hha_by_size = spdd.get_household_head_age_by_size_path(datadir, state_location=state_location,
                                                         country_location=country_location)
    assert hha_by_size is not None

@pytest.mark.skip(reason="Path methods were removed; we're keeping this around for reference.")
def test_get_school_enrollment_rates():
    school_enrollement_file_path = os.path.join(datadir,
                                                country_location,
                                                state_location,
                                                'household_head_age_and_size_count.dat')
    dict = spdd.get_school_enrollment_rates(datadir, location=location, state_location=state_location,
                                          country_location=country_location,
                                          file_path=school_enrollement_file_path, use_default=True)
    assert dict is not None

@pytest.mark.skip
def test_get_contact_matrix():
    setting_code = 'H'
    sheet_name = 'United States of America'
    data_matrix = spdd.get_contact_matrix(datadir, setting_code, sheet_name=sheet_name)
    assert len(data_matrix) == 16


if __name__ == '__main__':
    # We currently have files for both Senegal and USA for this data
    # test_get_gender_fraction_by_age_path()
    # test_read_gender_fraction_by_age()
    test_get_school_enrollment_rates()
    test_get_head_age_brackets_path()
    test_get_contact_matrix()
    test_get_age_bracket_distr_path()
    # We currently only have files for USA for this data
    test_get_household_size_distr_path()
    test_get_household_head_age_by_size_path()
    test_get_head_age_by_size_path()
    # test_get_gender_fraction_by_age_path()
