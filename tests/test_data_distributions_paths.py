import os
import synthpops as sp
import sciris as sc
import pytest

datadir = sp.datadir
location = 'seattle_metro'
state_location = 'Washington'
country_location = 'usa'


def test_get_gender_fraction_by_age_path_all_variables():
    """
    Test getting the file path with all input variables
    """
    dat_file = sp.get_gender_fraction_by_age_path(location=location, state_location=state_location,
                                                  country_location=country_location)
    print(dat_file)
    assert dat_file is not None


# def test_get_gender_fraction_by_age_path_country_variable_only():
#     """
#     Test getting the file path with only country_location variable
#     """
#     dat_file = sp.get_gender_fraction_by_age_path(country_location=country_location)
#     print(dat_file)
#     assert dat_file is not None


def test_get_gender_fraction_by_age_path_country_state_variables():
    """
    Test getting the file path with both state_location and country_location variables
    """
    dat_file = sp.get_gender_fraction_by_age_path(state_location=state_location,
                                                  country_location=country_location)
    assert dat_file is not None


if __name__ == '__main__':
    test_get_gender_fraction_by_age_path_all_variables()
    # test_get_gender_fraction_by_age_path_country_variable_only()
    test_get_gender_fraction_by_age_path_country_state_variables()
