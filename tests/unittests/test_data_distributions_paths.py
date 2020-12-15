print('All deprecated')

# import os
# import unittest
# import synthpops as sp
# import sciris as sc
# import pytest

# datadir = sp.datadir
# location = 'seattle_metro'
# state_location = 'Washington'
# country_location = 'usa'


# def test_get_gender_fraction_by_age_path_all_variables():
#     """
#     Test getting the file path with all input variables
#     """
#     dat_file, file_path = sp.get_gender_fraction_by_age_path(location=location, state_location=state_location, country_location=country_location)
#     check_exists = False if dat_file is None else os.path.exists(dat_file)
#     assert os.path.basename(dat_file) == 'Washington_gender_fraction_by_age_bracket_20.dat'
#     assert file_path is not None
#     assert check_exists == True

# def test_get_gender_fraction_by_age_path_all_variables_alt_location():
#     """
#     Test getting the file path with all input variables
#     """

#     current_alt_location = sp.config.alt_location
#     sp.config.set_alt_location(location=location, state_location=state_location, country_location=country_location)
#     dat_file, file_path = sp.get_gender_fraction_by_age_path(location='Yakima', state_location=state_location, country_location=country_location)
#     sp.config.alt_location = current_alt_location
#     check_exists = False if dat_file is None else os.path.exists(dat_file)

#     assert os.path.basename(dat_file) == 'Washington_gender_fraction_by_age_bracket_20.dat'
#     assert dat_file is not None
#     assert check_exists == True

# # def test_get_gender_fraction_by_age_path_country_variable_only():
# #     """
# #     Test getting the file path with only country_location variable
# #     """
# #     dat_file = sp.get_gender_fraction_by_age_path(country_location=country_location)
# #     print(dat_file)
# #     assert dat_file is not None

# def test_get_gender_fraction_by_age_path_country_state_variables():
#     """
#     Test getting the file path with both state_location and country_location variables
#     """
#     dat_file, file_path = sp.get_gender_fraction_by_age_path(state_location=state_location,
#                                                   country_location=country_location)
#     check_exists = False if dat_file is None else os.path.exists(dat_file)
#     assert dat_file is not None
#     assert os.path.basename(dat_file) == 'Washington_gender_fraction_by_age_bracket_20.dat'
#     assert check_exists == True

# @unittest.skip("in progress")
# def test_get_gender_fraction_by_age_path_all_variables_Senegal():
#     """
#     Test getting the file path with all input variables
#     Note get_gender_fraction_by_age does not exist for senegal
#     """
#     default_country = sp.config.default_country
#     sp.config.set_location_defaults('Senegal')
#     datadir = sp.datadir
#     location = 'Dakar'
#     state_location = 'Dakar'
#     country_location = 'Senegal'
#     dat_file, file_path = sp.get_gender_fraction_by_age_path(location=location, state_location=state_location,   country_location=country_location)
#     sp.config.set_location_defaults(default_country)
#     assert file_path is not None

# def test_get_census_age_brackets_path_usa():
#     """
#     Test getting the file path with all input variables
#     """
#     #default_country = sp.config.default_country
#     #sp.config.set_location_defaults('Senegal')
#     datadir = sp.datadir
#     location = 'seattle_metro'
#     state_location = 'Washington'
#     country_location = 'usa'
#     data_file, file_path = sp.get_census_age_brackets_path(datadir, state_location=state_location,
#                                                   country_location=country_location)
#     assert file_path is not None
#     check_file = os.path.exists(file_path)
#     assert check_file == True



# def test_get_census_age_brackets_path_Senegal():
#     """
#     Test getting the file path with all input variables
#     """
#     default_country = sp.config.default_country
#     sp.config.set_location_defaults('Senegal')
#     datadir = sp.datadir
#     location = 'Dakar'
#     state_location = 'Dakar'
#     country_location = 'Senegal'
#     data_file, file_path = sp.get_census_age_brackets_path(datadir, state_location=state_location,
#                                                   country_location=country_location)
#     print(data_file)
#     print(file_path)
#     sp.config.set_location_defaults(default_country)
#     assert file_path is not None

# def test_get_school_size_brackets_path_Senegal():
#     """
#     Test getting the file path with all input variables
#     """
#     default_country = sp.config.default_country
#     sp.config.set_location_defaults('Senegal')
#     datadir = sp.datadir
#     location = 'Dakar'
#     state_location = 'Dakar'
#     country_location = 'Senegal'
#     data_file = sp.get_school_size_brackets_path(datadir, location=location, state_location=state_location,  country_location=country_location)
#     print(data_file)
#     sp.config.set_location_defaults(default_country)
#     assert data_file is not None

# def test_get_school_sizes_path_Senegal():
#     """
#     Test getting the file path with all input variables
#     """
#     default_country = sp.config.default_country
#     sp.config.set_location_defaults('Senegal')
#     datadir = sp.datadir
#     location = 'Dakar'
#     state_location = 'Dakar'
#     country_location = 'Senegal'
#     data_file = sp.get_school_sizes_path(datadir, location=location, state_location=state_location,  country_location=country_location)
#     print(data_file)
#     sp.config.set_location_defaults(default_country)
#     assert data_file is not  None

# def test_get_school_size_distr_by_brackets_path_Senegal():
#     """
#     Test getting the file path with all input variables
#     """
#     default_country = sp.config.default_country
#     sp.config.set_location_defaults('Senegal')
#     datadir = sp.datadir
#     location = 'Dakar'
#     state_location = 'Dakar'
#     country_location = 'Senegal'
#     data_file = sp.get_school_size_distr_by_brackets_path(datadir, location=location, state_location=state_location,  country_location=country_location)
#     print(data_file)
#     sp.config.set_location_defaults(default_country)
#     assert data_file is not None

# def test_get_school_size_distr_by_brackets_path_Senegal():
#     """
#     Test getting the file path with all input variables
#     """
#     default_country = sp.config.default_country
#     sp.config.set_location_defaults('Senegal')
#     sp.set_nbrackets(18)
#     datadir = sp.datadir
#     location = 'Dakar'
#     state_location = 'Dakar'
#     country_location = 'Senegal'
#     data_file = sp.get_school_size_distr_by_brackets_path(datadir, location=location, state_location=state_location,  country_location=country_location)
#     print(data_file)
#     sp.config.set_location_defaults(default_country)
#     assert data_file is not None

# if __name__ == '__main__':
#     test_get_gender_fraction_by_age_path_all_variables()
#     # test_get_gender_fraction_by_age_path_country_variable_only()
#     test_get_gender_fraction_by_age_path_country_state_variables()
#     # test_get_gender_fraction_by_age_path_all_variables_Senegal()
#     test_get_census_age_brackets_path_Senegal()
#     test_get_school_size_brackets_path_Senegal()
#     test_get_school_sizes_path_Senegal()
#     test_get_school_size_distr_by_brackets_path_Senegal()

