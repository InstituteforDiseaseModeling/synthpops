import synthpops as sp
import numpy as np
import sciris as sc
import pytest
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
    # num_agebrackets = 18

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

    single_year_age_distr = {}
    for n in range(101):
        single_year_age_distr[n] = float(1.0 / 101.0)
    tolerance = 2  # the resampled age should be within two years
    age_distr_vals = np.array(list(single_year_age_distr.values()), dtype=np.float)
    for n in range(int(1e3)):
        random_age = int(randrange(100))

        resampled_age = sp.resample_age(age_distr_vals, random_age)
        assert abs(random_age - resampled_age) <= tolerance


def test_generate_household_sizes(location='seattle_metro', state_location='Washington', country_location='usa'):
    sc.heading('Generate household sizes')

    Nhomes_to_sample_smooth = 1000
    household_size_distr = sp.get_household_size_distr(datadir, location, state_location, country_location)
    hh_sizes = sp.generate_household_sizes(Nhomes_to_sample_smooth, household_size_distr)
    assert len(hh_sizes) == 7


def test_generate_household_sizes_from_fixed_pop_size(location='seattle_metro', state_location='Washington',
                                                      country_location='usa'):
    household_size_distr = sp.get_household_size_distr(datadir, location, state_location, country_location)

    Nhomes = 1000
    hh_sizes = sp.generate_household_sizes_from_fixed_pop_size(Nhomes, household_size_distr)
    assert len(hh_sizes) == 7


def test_generate_all_households(location='seattle_metro', state_location='Washington',
                                 country_location='usa'):
    N = 1000
    household_size_distr = sp.get_household_size_distr(datadir, location, state_location, country_location)

    hh_sizes = sp.generate_household_sizes_from_fixed_pop_size(N,household_size_distr)
    hha_brackets = sp.get_head_age_brackets(datadir, country_location=country_location)
    hha_by_size_counts = sp.get_head_age_by_size_distr(datadir, country_location=country_location)

    age_brackets_filepath = sp.get_census_age_brackets_path(datadir, state_location, country_location)
    age_brackets = sp.get_age_brackets_from_df(age_brackets_filepath)
    age_by_brackets_dic = sp.get_age_by_brackets_dic(age_brackets)

    contact_matrix_dic = sp.get_contact_matrix_dic(datadir, sheet_name='United States of America')

    single_year_age_distr = {}
    for n in range(101):
        single_year_age_distr[n] = float(1.0 / 101.0)

    homes_dic, homes = sp.generate_all_households(N, hh_sizes, hha_by_size_counts, hha_brackets, age_brackets,
                                                  age_by_brackets_dic, contact_matrix_dic, single_year_age_distr)
    assert homes_dic, homes is not None


def test_get_totalpopsizes_from_household_sizes(location='seattle_metro', state_location='Washington',
                                                      country_location='usa'):
    household_size_distr = sp.get_household_size_distr(datadir, location, state_location, country_location)

    Nhomes_to_sample_smooth = 1000
    hh_sizes = sp.generate_household_sizes(Nhomes_to_sample_smooth, household_size_distr)
    sum_hh_sizes = sp.get_totalpopsize_from_household_sizes(hh_sizes)
    assert sum_hh_sizes is not None


def test_assign_uids_by_homes(state_location='Washington', country_location='usa'):
    homes = sp.get_head_age_by_size_distr(datadir, state_location, country_location, file_path=None,
                                          household_size_1_included=False, use_default=True)

    homes_by_uids, age_by_uid_dic = sp.assign_uids_by_homes(homes, id_len=16)

    assert homes_by_uids, age_by_uid_dic is not None


def test_get_school_enrollment_rates_path():
    path = sp.get_school_enrollment_rates_path(datadir=datadir, location='seattle_metro', state_location='Washington',
                                               country_location='usa')
    assert path is not None


def test_get_uids_in_school(location='seattle_metro', state_location='Washington',
                            country_location='usa'):
    NPeople = 10000
    uids_in_school, uids_in_school_by_age, ages_in_school_count = sp.get_uids_in_school(datadir, NPeople, location,
                                                                                        state_location,
                                                                                        country_location,
                                                                                        use_default=True)
    assert uids_in_school is not None


def test_send_students_to_school(location='seattle_metro', state_location='Washington',
                                 country_location='usa'):
    NPeople = 10000

    uids_in_school, uids_in_school_by_age, ages_in_school_count = sp.get_uids_in_school(datadir, NPeople, location,
                                                                                        state_location,
                                                                                        country_location,
                                                                                        use_default=True)

    school_size_distr_by_bracket = sp.get_school_size_distr_by_brackets(datadir, location, state_location,
                                                                        country_location)
    school_size_brackets = sp.get_school_size_brackets(datadir, location, state_location, country_location)
    school_sizes = sp.generate_school_sizes(school_size_distr_by_bracket, school_size_brackets, uids_in_school)

    age_brackets_filepath = sp.get_census_age_brackets_path(datadir, state_location, country_location)
    age_brackets = sp.get_age_brackets_from_df(age_brackets_filepath)
    age_by_brackets_dic = sp.get_age_by_brackets_dic(age_brackets)

    contact_matrix_dic = sp.get_contact_matrix_dic(datadir, sheet_name='United States of America')

    syn_schools, syn_school_uids = sp.send_students_to_school(school_sizes, uids_in_school, uids_in_school_by_age,
                                                              ages_in_school_count, age_brackets, age_by_brackets_dic,
                                                              contact_matrix_dic, verbose=False)
    assert syn_schools, syn_school_uids is not None


def test_get_uids_potential_workers(location='seattle_metro', state_location='Washington',
                                    country_location='usa'):
    Nhomes = 10000
    uids_in_school = sp.get_uids_in_school(datadir, Nhomes, location,
                                           state_location,
                                           country_location,
                                           use_default=True)
    employment_rates = sp.get_employment_rates(datadir, location=location, state_location=state_location,
                                               country_location=country_location, use_default=True)
    age_by_uid_dic = sp.read_in_age_by_uid(datadir, location, state_location, country_location, Nhomes)
    potential_worker_uids, potential_worker_uids_by_age, potential_worker_ages_left_count = sp.get_uids_potential_workers(
        uids_in_school, employment_rates, age_by_uid_dic)
    assert potential_worker_ages_left_count is not None


def test_generate_workplace_sizes(location='seattle_metro', state_location='Washington',
                                  country_location='usa'):
    Npeople = 10000
    uids_in_school, uids_in_school_by_age, uids_in_school_count = sp.get_uids_in_school(datadir, Npeople, location,
                                                                                        state_location,
                                                                                        country_location,
                                                                                        use_default=True)

    employment_rates = sp.get_employment_rates(datadir, location=location, state_location=state_location,
                                               country_location=country_location, use_default=True)

    age_by_uid_dic = sp.read_in_age_by_uid(datadir, location, state_location, country_location, Npeople)

    potential_worker_uids, potential_worker_uids_by_age, potential_worker_ages_left_count = sp.get_uids_potential_workers(
        uids_in_school, employment_rates, age_by_uid_dic)

    workers_by_age_to_assign_count = sp.get_workers_by_age_to_assign(employment_rates, potential_worker_ages_left_count,
                                                                     age_by_uid_dic)

    workplace_size_brackets = sp.get_workplace_size_brackets(datadir, location, state_location, country_location,
                                                             use_default=True)

    workplace_size_distr_by_brackets = sp.get_workplace_size_distr_by_brackets(datadir,
                                                                               state_location=state_location,
                                                                               country_location=country_location,
                                                                               use_default=True)
    workplace_sizes = sp.generate_workplace_sizes(workplace_size_distr_by_brackets, workplace_size_brackets,
                                                  workers_by_age_to_assign_count)
    print(workplace_sizes)


def test_generate_school_sizes(location='seattle_metro', state_location='Washington',
                               country_location='usa'):
    Nhomes = 10000
    uids_in_school = sp.get_uids_in_school(datadir, Nhomes, location,
                                           state_location,
                                           country_location,
                                           use_default=True)

    school_size_distr_by_bracket = sp.get_school_size_distr_by_brackets(datadir, location, state_location,
                                                                        country_location)
    school_size_brackets = sp.get_school_size_brackets(datadir, location, state_location, country_location)
    school_sizes = sp.generate_school_sizes(school_size_distr_by_bracket, school_size_brackets, uids_in_school)
    assert school_sizes is not None


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