import os
import synthpops as sp
import sciris as sc
import pytest
from copy import deepcopy
from collections import Counter
from synthpops import sampling as spsamp
from synthpops import base

if not sp.config.full_data_available:
    pytest.skip("Data not available, tests not possible", allow_module_level=True)


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


@pytest.mark.skip
# TODO: Not sure how to do this
def test_resample_age(state_location='Washington', country_location='usa'):
    sc.heading('Resample age')

    datadir = sp.datadir

    age_brackets = sp.get_census_age_brackets(datadir, state_location=state_location,
                                              country_location=country_location)
    age_by_brackets_dic = sp.get_age_by_brackets_dic(age_brackets)

    sheet_name = 'United States of America'
    age_mixing_matrix_dic = sp.get_contact_matrix_dic(datadir, sheet_name=sheet_name)

    n_contacts_dic = {'H': 4, 'S': 20, 'W': 20, 'C': 20}
    k = 'H'
    nc = spsamp.pt(n_contacts_dic[k])

    default_n = 10000
    popdict = sp.make_popdict(n=default_n)
    for uid in popdict:
        age = popdict[uid]['age']

    single_year_age_distr = spsamp.sample_n_contact_ages_with_matrix(nc, age, age_brackets, age_by_brackets_dic,
                                                                     age_mixing_matrix_dic[k])

    hha_brackets = sp.get_head_age_brackets(datadir, country_location=country_location)
    hha_by_size_counts = sp.get_head_age_by_size_distr(datadir, country_location=country_location)
    size = 1
    hha = sp.generate_household_head_age_by_size(hha_by_size_counts, hha_brackets, size, single_year_age_distr)
    b = age_by_brackets_dic[hha]
    contact_matrix_dic = sp.get_contact_matrix_dic(datadir, sheet_name=sheet_name)
    b_prob = contact_matrix_dic['H'][b, :]
    bi = spsamp.sample_single(b_prob)

    ai = spsamp.sample_from_range(single_year_age_distr, age_brackets[bi][0], age_brackets[bi][-1])

    sp.resample_age(single_year_age_distr, ai)


def test_generate_household_sizes(location='seattle_metro', state_location='Washington', country_location='usa'):
    sc.heading('Generate household sizes')

    datadir = sp.datadir

    Nhomes_to_sample_smooth = 100000
    household_size_distr = sp.get_household_size_distr(datadir, location, state_location, country_location)
    hh_sizes = sp.generate_household_sizes(Nhomes_to_sample_smooth, household_size_distr)
    assert len(hh_sizes) == 7



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