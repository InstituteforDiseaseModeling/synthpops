import os
import synthpops as sp
import sciris as sc
import pytest
from copy import deepcopy
from collections import Counter
from synthpops import sampling as spsamp
from synthpops import base
from random import randrange

if not sp.config.full_data_available:
    pytest.skip("Data not available, tests not possible", allow_module_level=True)

datadir = sp.datadir


def test_all(location='seattle_metro',state_location='Washington',country_location='usa',sheet_name='United States of America'):
    ''' Run all tests '''

    sc.heading('Running all tests')

    sp.validate() # Validate that data files can be found
    dropbox_path = sp.datadir

    age_bracket_distr = sp.read_age_bracket_distr(dropbox_path,location,state_location,country_location)
    gender_fraction_by_age = sp.read_gender_fraction_by_age_bracket(dropbox_path,location,state_location,country_location)
    age_brackets_filepath = sp.get_census_age_brackets_path(dropbox_path,state_location,country_location)
    age_brackets = sp.get_age_brackets_from_df(age_brackets_filepath)
    age_by_brackets_dic = sp.get_age_by_brackets_dic(age_brackets)

    ### Test selecting an age and sex for an individual ###
    a,s = sp.get_age_sex(gender_fraction_by_age,age_bracket_distr,age_brackets)
    print(a,s)

    ### Test age mixing matrix ###
    num_agebrackets = 18

    # flu-like weights. calibrated to empirical diary survey data.
    weights_dic = {'H': 4.11, 'S': 11.41, 'W': 8.07, 'C': 2.79}

    age_mixing_matrix_dic = sp.get_contact_matrix_dic(dropbox_path,sheet_name)

    ### Test sampling contacts based on age ###
    age, sex = sp.get_age_sex(gender_fraction_by_age,age_bracket_distr,age_brackets) # sample an age (and sex) from the seattle metro distribution

    n_contacts = 30
    contact_ages = sp.sample_n_contact_ages(n_contacts,age,age_brackets,age_by_brackets_dic,age_mixing_matrix_dic,weights_dic)
    print(contact_ages)


    # shut down schools
    no_schools_weights = sc.dcp(weights_dic)
    no_schools_weights['S'] = 0.1 # research shows that even with school closure, kids still have some contact with their friends from school.

    f_reduced_contacts_students = 0.5
    f_reduced_contacts_nonstudents = 0.2

    if age < 20:
        n_reduced_contacts = int(n_contacts * (1 - f_reduced_contacts_students))
    else:
        n_reduced_contacts = int(n_contacts * (1 - f_reduced_contacts_nonstudents))

    contact_ages = sp.sample_n_contact_ages(n_reduced_contacts,age,age_brackets,age_by_brackets_dic,age_mixing_matrix_dic,no_schools_weights)
    print(contact_ages)

    return


def test_n_single_ages(n_people=1e4,location='seattle_metro',state_location='Washington',country_location='usa'):

    sc.heading('Running single ages')
    sp.validate()
    datadir = sp.datadir

    age_bracket_distr = sp.read_age_bracket_distr(datadir,location,state_location,country_location)
    gender_fraction_by_age = sp.read_gender_fraction_by_age_bracket(datadir,location,state_location,country_location)
    age_brackets_filepath = sp.get_census_age_brackets_path(datadir,state_location,country_location)
    age_brackets = sp.get_age_brackets_from_df(age_brackets_filepath)

    ### Test selecting an age and sex for an individual ###
    a,s = sp.get_age_sex(gender_fraction_by_age,age_bracket_distr,age_brackets)
    print(a,s)

    n_people = int(n_people)
    ages, sexes = [], []
    for p in range(n_people):
        a,s = sp.get_age_sex(gender_fraction_by_age,age_bracket_distr,age_brackets)
        ages.append(a)
        sexes.append(s)

    return


def test_multiple_ages(n_people=1e4,location='seattle_metro',state_location='Washington',country_location='usa'):
    sc.heading('Running multiple ages')

    datadir = sp.datadir

    age_bracket_distr = sp.read_age_bracket_distr(datadir,location,state_location,country_location)
    gender_fraction_by_age = sp.read_gender_fraction_by_age_bracket(datadir,location,state_location,country_location)
    age_brackets_filepath = sp.get_census_age_brackets_path(datadir,state_location,country_location)
    age_brackets = sp.get_age_brackets_from_df(age_brackets_filepath)

    ages, sexes = sp.get_age_sex_n(gender_fraction_by_age,age_bracket_distr,age_brackets,n_people)
    print(len(ages),len(sexes))

    return


def test_resample_age():
    sc.heading('Resample age')

    single_year_age_distr = {3: 0.011913819924925495, 2: 0.01184275603726525, 4: 0.011789662328093801,
                             1: 0.011888906722929662, 0: 0.011973039831309033, 7: 0.011999586685894757,
                             9: 0.0121020983859104, 8: 0.012092296470371055, 6: 0.012001220338484648,
                             5: 0.01201224749346641, 12: 0.010614249289667433, 11: 0.010641612970548102,
                             14: 0.010787008051048376, 10: 0.010629768989271395, 13: 0.010642429796843048,
                             17: 0.010853579394086422, 16: 0.010869507506837857, 15: 0.010974061272590863,
                             19: 0.005552354154644016, 18: 0.005751659770610685, 21: 0.01045653977678589,
                             20: 0.009449188340131387, 22: 0.010311144696285617, 23: 0.010091602086337735,
                             24: 0.010719026988349208, 26: 0.016264746675619092, 25: 0.01617583505173166,
                             29: 0.01623411568955864, 28: 0.015737852629016703, 27: 0.015959028891554476,
                             32: 0.016121046468839516, 34: 0.01591573709792237, 33: 0.01563446288157531,
                             31: 0.01580995809272695, 30: 0.016102667877203245, 35: 0.013665380655347916,
                             39: 0.013636852956155635, 38: 0.013600218215144708, 37: 0.01348014474978774,
                             36: 0.013655701182070185, 40: 0.010464176367499974, 43: 0.01223074771053801,
                             42: 0.012244286565535415, 41: 0.01265192397307578, 44: 0.012184188611726125,
                             48: 0.011139140723242524, 46: 0.01142735796309662, 45: 0.011928930394555701,
                             49: 0.011764462338385826, 47: 0.011421987371048669, 50: 0.010852128588062669,
                             54: 0.012155109922356584, 52: 0.01074197948050666, 51: 0.011045716419964718,
                             53: 0.011533872438688008, 55: 0.01226284963739039, 57: 0.011377205318682745,
                             58: 0.011131054633018334, 59: 0.011569384247749956, 56: 0.011400423646957881,
                             62: 0.010593971209323587, 60: 0.010734812524070854, 63: 0.01045972572606669,
                             64: 0.010823846590219899, 61: 0.010656458420886908, 69: 0.007510758051558897,
                             66: 0.007786253262710538, 65: 0.007767119065910128, 67: 0.00798713131013629,
                             68: 0.0077655466344510446, 71: 0.005847026282611278, 70: 0.005051927240380949,
                             72: 0.004780924573851508, 73: 0.005525850020073504, 74: 0.005770367053148027,
                             77: 0.0014567073895416008, 90: 0.0014832542441273252, 76: 0.0010554821500991828,
                             92: 0.0014154576616468605, 88: 0.0016959150516989783, 91: 0.0015746775680303988,
                             100: 0.0019530316712143472, 84: 0.0017799869389475437, 87: 0.001515110469630194,
                             79: 0.0014881552018969973, 97: 0.0018273016429235683, 99: 0.0017493559528870928,
                             78: 0.0018179081405316968, 89: 0.0013424129293800571, 75: 0.0016648756524910546,
                             96: 0.0012648756524910544, 95: 0.0014897888544868879, 85: 0.001386113136159634,
                             93: 0.001586929962454579, 81: 0.0012550737369517102, 94: 0.0015085758592706313,
                             86: 0.0014595662815739096, 98: 0.001422400685153896, 83: 0.0016558905632466558,
                             80: 0.0015926477465191967, 82: 0.0013167441222200844}
    tolerance = 2  # the resampled age should be within two years
    for n in range(int(1e4)):
        random_age = int(randrange(100))
        resampled_age = sp.resample_age(single_year_age_distr, random_age)
        assert abs(random_age - resampled_age) <= tolerance


def test_generate_household_sizes(location='seattle_metro', state_location='Washington', country_location='usa'):
    sc.heading('Generate household sizes')

    datadir = sp.datadir

    Nhomes_to_sample_smooth = 100000
    household_size_distr = sp.get_household_size_distr(datadir, location, state_location, country_location)
    hh_sizes = sp.generate_household_sizes(Nhomes_to_sample_smooth, household_size_distr)
    assert len(hh_sizes) == 7


def test_generate_household_sizes_from_fixed_pop_size(location='seattle_metro', state_location='Washington',
                                                      country_location='usa'):
    datadir = sp.datadir
    household_size_distr = sp.get_household_size_distr(datadir, location, state_location, country_location)

    Nhomes = 10000
    hh_sizes = sp.generate_household_sizes_from_fixed_pop_size(Nhomes, household_size_distr)
    assert len(hh_sizes) == 7


def test_get_school_enrollment_rates_path():
    path = sp.get_school_enrollment_rates_path(datadir=datadir, location='seattle_metro', state_location='Washington',
                                               country_location='usa')
    assert path is not None


#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    datadir = sp.datadir
    location = 'seattle_metro' # for census distributions
    state_location = 'Washington' # for state wide age mixing patterns
    # location = 'portland_metro'
    # state_location = 'Oregon'
    country_location = 'usa'

    test_all(location,state_location,country_location)
    test_n_single_ages(1e4,location,state_location,country_location)
    test_multiple_ages(1e4,location,state_location,country_location)

    ages,sexes = sp.get_usa_age_sex_n(datadir,location,state_location,country_location,1e2)
    print(ages,sexes)

    # country_location = 'Algeria'
    # age_brackets_filepath = sp.get_census_age_brackets_path(sp.datadir,country_location)
    # age_brackets = sp.get_age_brackets_from_df(age_brackets_filepath)
    # print(age_brackets)
    sc.toc()





print('Done.')