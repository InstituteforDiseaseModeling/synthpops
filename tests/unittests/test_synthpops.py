""" Testing synthpops"""

import numpy as np
import sciris as sc
import pytest
import synthpops as sp
import synthpops.data_distributions as spdd
import synthpops.schools as spsch

# pytest.skip("Tests require refactoring - a few are calling the wrong functions to create data objects that go into other functions. This is why we are seeing indexing issues. ", allow_module_level=True)


datadir = sp.settings.datadir


@pytest.mark.skip(reason='Deprecated functions')
def test_all(location='seattle_metro', state_location='Washington', country_location='usa', sheet_name='United States of America'):
    ''' Run all tests '''

    sc.heading('Running all tests')

    sp.validate()  # Validate that data files can be found
    # dropbox_path = sp.datadir
    dropbox_path = sp.settings.datadir

    age_bracket_distr = spdd.read_age_bracket_distr(dropbox_path, location, state_location, country_location)
    gender_fraction_by_age = sp.read_gender_fraction_by_age_bracket(dropbox_path, location, state_location, country_location)
    age_brackets_file, age_brackets_filepath = sp.get_census_age_brackets_path(dropbox_path, state_location, country_location)
    print(age_brackets_filepath)
    age_brackets = sp.get_age_brackets_from_df(age_brackets_filepath)
    age_by_brackets_dic = sp.get_age_by_brackets_dic(age_brackets)

    # ## Test selecting an age and sex for an individual ###
    a, s = sp.get_age_sex(gender_fraction_by_age, age_bracket_distr, age_brackets)
    print(a, s)

    # ## Test age mixing matrix ###
    # num_agebrackets = 18

    # flu-like weights. calibrated to empirical diary survey data.
    weights_dic = {'H': 4.11, 'S': 11.41, 'W': 8.07, 'C': 2.79}

    age_mixing_matrix_dic = sp.get_contact_matrix_dic(dropbox_path, sheet_name)

    # ## Test sampling contacts based on age ###
    age, sex = sp.get_age_sex(gender_fraction_by_age, age_bracket_distr, age_brackets)  # sample an age (and sex) from the seattle metro distribution

    n_contacts = 30
    contact_ages = sp.sample_n_contact_ages(n_contacts, age, age_brackets, age_by_brackets_dic, age_mixing_matrix_dic, weights_dic)
    print(contact_ages)

    # shut down schools
    no_schools_weights = sc.dcp(weights_dic)
    no_schools_weights['S'] = 0.1  # research shows that even with school closure, kids still have some contact with their friends from school.

    f_reduced_contacts_students = 0.5
    f_reduced_contacts_nonstudents = 0.2

    if age < 20:
        n_reduced_contacts = int(n_contacts * (1 - f_reduced_contacts_students))
    else:
        n_reduced_contacts = int(n_contacts * (1 - f_reduced_contacts_nonstudents))

    contact_ages = sp.sample_n_contact_ages(n_reduced_contacts, age, age_brackets, age_by_brackets_dic, age_mixing_matrix_dic, no_schools_weights)
    print(contact_ages)

    return


@pytest.mark.skip(reason='Deprecated functions')
def test_n_single_ages(n_people=1e4, location='seattle_metro', state_location='Washington', country_location='usa'):

    sc.heading('Running single ages')
    sp.validate()
    datadir = sp.settings.datadir

    age_bracket_distr = spdd.read_age_bracket_distr(datadir, location, state_location, country_location)
    gender_fraction_by_age = sp.read_gender_fraction_by_age_bracket(datadir, location, state_location, country_location)
    age_brackets_file, age_brackets_filepath = sp.get_census_age_brackets_path(datadir, state_location, country_location)
    age_brackets = sp.get_age_brackets_from_df(age_brackets_filepath)

    # ## Test selecting an age and sex for an individual ###
    a, s = sp.get_age_sex(gender_fraction_by_age, age_bracket_distr, age_brackets)
    print(a, s)

    n_people = int(n_people)
    ages, sexes = [], []
    for p in range(n_people):
        a, s = sp.get_age_sex(gender_fraction_by_age, age_bracket_distr, age_brackets)
        ages.append(a)
        sexes.append(s)

    return


@pytest.mark.skip(reason='Deprecated functions')
def test_multiple_ages(n_people=1e4, location='seattle_metro', state_location='Washington', country_location='usa'):
    sc.heading('Running multiple ages')

    datadir = sp.settings.datadir

    age_bracket_distr = spdd.read_age_bracket_distr(datadir, location, state_location, country_location)
    gender_fraction_by_age = sp.read_gender_fraction_by_age_bracket(datadir, location, state_location, country_location)
    age_brackets_file, age_brackets_filepath = sp.get_census_age_brackets_path(datadir, state_location, country_location)
    age_brackets = sp.get_age_brackets_from_df(age_brackets_filepath)

    ages, sexes = sp.get_age_sex_n(gender_fraction_by_age, age_bracket_distr, age_brackets, n_people)
    print(len(ages), len(sexes))

    return


def test_resample_age():
    sc.heading('Resample age')

    single_year_age_distr = {}
    for n in range(101):
        single_year_age_distr[n] = float(1.0 / 101.0)
    tolerance = 2  # the resampled age should be within two years
    age_distr_vals = np.array(list(single_year_age_distr.values()), dtype=np.float)
    for n in range(int(1e3)):
        random_age = int(np.random.randint(100))

        resampled_age = sp.resample_age(age_distr_vals, random_age)
        assert abs(random_age - resampled_age) <= tolerance


@pytest.mark.skip(reason='Deprecated functions')
def test_generate_household_sizes(location='seattle_metro', state_location='Washington', country_location='usa'):
    sc.heading('Generate household sizes')

    Nhomes_to_sample_smooth = 1000
    household_size_distr = sp.get_household_size_distr(datadir, location, state_location, country_location)
    hh_sizes = sp.generate_household_sizes(Nhomes_to_sample_smooth, household_size_distr)
    assert len(hh_sizes) == 7


@pytest.mark.skip(reason='Deprecated functions')
def test_generate_household_sizes_from_fixed_pop_size(location='seattle_metro', state_location='Washington',
                                                      country_location='usa'):
    household_size_distr = sp.get_household_size_distr(datadir, location, state_location, country_location)

    Nhomes = 1000
    hh_sizes = sp.generate_household_sizes_from_fixed_pop_size(Nhomes, household_size_distr)
    assert len(hh_sizes) == 7


@pytest.mark.skip(reason='Deprecated functions')
def test_generate_all_households(location='seattle_metro', state_location='Washington',
                                 country_location='usa'):
    N = 1000
    household_size_distr = sp.get_household_size_distr(datadir, location, state_location, country_location)

    hh_sizes = sp.generate_household_sizes_from_fixed_pop_size(N, household_size_distr)
    hha_brackets = sp.get_head_age_brackets(datadir, state_location=state_location, country_location=country_location)
    hha_by_size_counts = sp.get_head_age_by_size_distr(datadir, state_location=state_location, country_location=country_location)

    age_brackets_file, age_brackets_filepath = sp.get_census_age_brackets_path(datadir, state_location, country_location)
    age_brackets = sp.get_age_brackets_from_df(age_brackets_filepath)
    age_by_brackets_dic = sp.get_age_by_brackets_dic(age_brackets)

    contact_matrix_dic = sp.get_contact_matrix_dic(datadir, sheet_name='United States of America')

    single_year_age_distr = {}
    for n in range(101):
        single_year_age_distr[n] = float(1.0 / 101.0)

    homes_dic, homes = sp.generate_all_households(N, hh_sizes, hha_by_size_counts, hha_brackets, age_brackets,
                                                  age_by_brackets_dic, contact_matrix_dic, single_year_age_distr)
    assert homes_dic, homes is not None


@pytest.mark.skip(reason='Deprecated functions')
def test_get_totalpopsizes_from_household_sizes(location='seattle_metro', state_location='Washington',
                                                country_location='usa'):
    household_size_distr = sp.get_household_size_distr(datadir, location, state_location, country_location)

    Nhomes_to_sample_smooth = 1000
    hh_sizes = sp.generate_household_sizes(Nhomes_to_sample_smooth, household_size_distr)
    sum_hh_sizes = sp.get_totalpopsize_from_household_sizes(hh_sizes)
    assert sum_hh_sizes is not None


@pytest.mark.skip(reason='Deprecated functions')
def test_generate_larger_households(location='seattle_metro', state_location='Washington',
                                    country_location='usa'):
    Nhomes_to_sample_smooth = 1000
    household_size_distr = sp.get_household_size_distr(datadir, location, state_location, country_location)
    hh_sizes = sp.generate_household_sizes(Nhomes_to_sample_smooth, household_size_distr)

    hha_brackets = sp.get_head_age_brackets(datadir, country_location=country_location)
    hha_by_size_counts = sp.get_head_age_by_size_distr(datadir, country_location=country_location)

    age_brackets_file, age_brackets_filepath = sp.get_census_age_brackets_path(datadir, state_location, country_location)
    age_brackets = sp.get_age_brackets_from_df(age_brackets_filepath)
    age_by_brackets_dic = sp.get_age_by_brackets_dic(age_brackets)

    contact_matrix_dic = sp.get_contact_matrix_dic(datadir, sheet_name='United States of America')

    single_year_age_distr = {}
    for n in range(101):
        single_year_age_distr[n] = float(1.0 / 101.0)

    # generate households of size 3
    size = 3
    # first variable is the household size to be created, so here this means we want to create all households of size 3 and the hh_sizes variable tells us how many of size 3 will be created at index 3-1 (since hh_sizes is an array rather than a dictionary)
    larger_households = sp.generate_larger_households(size, hh_sizes, hha_by_size_counts, hha_brackets, age_brackets, age_by_brackets_dic,
                                                      contact_matrix_dic, single_year_age_distr)
    assert larger_households is not None
    print(larger_households)

# @pytest.mark.skip(reason='Deprecated')
# def test_assign_uids_by_homes(state_location='Washington', country_location='usa'):
#     homes = sp.get_head_age_by_size_distr(datadir, state_location, country_location, file_path=None,
#                                           household_size_1_included=False, use_default=True)

#     homes_by_uids, age_by_uid_dic = sp.assign_uids_by_homes(homes, id_len=16)

#     assert homes_by_uids, age_by_uid_dic is not None


# def test_get_school_enrollment_rates_path():
#     path = sp.get_school_enrollment_rates_path(datadir=datadir, location='seattle_metro', state_location='Washington',
#                                                country_location='usa')
#     assert path is not None


# @pytest.mark.skip(reason='Reading/writing is deprecated')
# def test_get_uids_in_school(location='seattle_metro', state_location='Washington',
#                             country_location='usa', folder_name='contact_networks'):

#     Npeople = 10000

#     homes = sprw.read_setting_groups(datadir, location, state_location, country_location, folder_name, 'households', Npeople, with_ages=True)

#     homes_by_uids, age_by_uid_dic = sp.assign_uids_by_homes(homes)

#     uids_in_school, uids_in_school_by_age, ages_in_school_count = sp.get_uids_in_school(datadir, Npeople, location,
#                 state_location,
#                 country_location,
#                 age_by_uid_dic,
#                 homes_by_uids,
#                 use_default=False)
#     assert uids_in_school is not None


# @pytest.mark.skip(reason='Reading/writing is deprecated')
# def test_send_students_to_school(n=10000, location='seattle_metro', state_location='Washington',
#                                  country_location='usa', folder_name='contact_networks'):

#     homes = sprw.read_setting_groups(datadir, location, state_location, country_location, folder_name, 'households', n, with_ages=True)

#     homes_by_uids, age_by_uid_dic = sp.assign_uids_by_homes(homes)

#     uids_in_school, uids_in_school_by_age, ages_in_school_count = sp.get_uids_in_school(datadir, n, location,
#                                                                                         state_location,
#                                                                                         country_location,
#                                                                                         age_by_uid_dic,
#                                                                                         homes_by_uids,
#                                                                                         use_default=False)

#     school_size_distr_by_bracket = sp.get_school_size_distr_by_brackets(datadir, location, state_location,
#                                                                         country_location)
#     school_size_brackets = sp.get_school_size_brackets(datadir, location, state_location, country_location)
#     school_sizes = sp.generate_school_sizes(school_size_distr_by_bracket, school_size_brackets, uids_in_school)

#     age_brackets_file, age_brackets_filepath = sp.get_census_age_brackets_path(datadir, state_location, country_location)
#     age_brackets = sp.get_age_brackets_from_df(age_brackets_filepath)
#     age_by_brackets_dic = sp.get_age_by_brackets_dic(age_brackets)

#     contact_matrix_dic = sp.get_contact_matrix_dic(datadir, sheet_name='United States of America')

#     syn_schools, syn_school_uids, syn_school_types = sp.send_students_to_school(school_sizes, uids_in_school, uids_in_school_by_age,
#                                                                                 ages_in_school_count, age_brackets, age_by_brackets_dic,
#                                                                                 contact_matrix_dic)
#     assert syn_schools, syn_school_uids is not None

#     return syn_schools, syn_school_uids


# @pytest.mark.skip(reason='Reading/writing is deprecated')
# def test_get_uids_potential_workers(location='seattle_metro', state_location='Washington',
#                                     country_location='usa'):
#     n = 10000
#     homes = sprw.read_setting_groups(datadir, location, state_location, country_location, folder_name, 'households', n, with_ages=True)

#     homes_by_uids, age_by_uid_dic = sp.assign_uids_by_homes(homes)

#     uids_in_school, uids_in_school_by_age, ages_in_school_count = sp.get_uids_in_school(datadir, n, location,
#                                                                                         state_location,
#                                                                                         country_location,
#                                                                                         age_by_uid_dic,
#                                                                                         homes_by_uids,
#                                                                                         use_default=False)

#     employment_rates = sp.get_employment_rates(datadir, location=location, state_location=state_location,
#                                                country_location=country_location, use_default=True)

#     school_size_distr_by_bracket = sp.get_school_size_distr_by_brackets(datadir, location, state_location,
#                                                                         country_location)

#     school_size_brackets = sp.get_school_size_brackets(datadir, location, state_location, country_location)
#     school_sizes = sp.generate_school_sizes(school_size_distr_by_bracket, school_size_brackets, uids_in_school)
#     age_brackets_filepath = sp.get_census_age_brackets_path(datadir, state_location, country_location)
#     age_brackets = sp.get_age_brackets_from_df(age_brackets_filepath)
#     age_by_brackets_dic = sp.get_age_by_brackets_dic(age_brackets)
#     contact_matrix_dic = sp.get_contact_matrix_dic(datadir, sheet_name='United States of America')

#     syn_schools, syn_school_uids, syn_school_types = sp.send_students_to_school(school_sizes, uids_in_school,
#                                                                                 uids_in_school_by_age,
#                                                                                 ages_in_school_count, age_brackets,
#                                                                                 age_by_brackets_dic,
#                                                                                 contact_matrix_dic)

#     potential_worker_uids, potential_worker_uids_by_age, potential_worker_ages_left_count = sp.get_uids_potential_workers(
#         syn_school_uids, employment_rates, age_by_uid_dic)
#     assert potential_worker_ages_left_count is not None

#     return potential_worker_uids, potential_worker_uids_by_age, employment_rates, age_by_uid_dic


# @pytest.mark.skip(reason='Reading/writing is deprecated')
# def test_generate_workplace_sizes(location='seattle_metro', state_location='Washington',
#                                   country_location='usa', folder_name='contact_networks'):
#     n = 10000
#     uids_in_school, uids_in_school_by_age, ages_in_school_count = sp.get_uids_in_school(datadir, n, location,
#                                                                                         state_location,
#                                                                                         country_location,
#                                                                                         folder_name=folder_name,
#                                                                                         use_default=True)

#     school_size_distr_by_bracket = sp.get_school_size_distr_by_brackets(datadir, location, state_location,
#                                                                         country_location)
#     school_size_brackets = sp.get_school_size_brackets(datadir, location, state_location, country_location)
#     school_sizes = sp.generate_school_sizes(school_size_distr_by_bracket, school_size_brackets, uids_in_school)

#     age_brackets_file, age_brackets_filepath = sp.get_census_age_brackets_path(datadir, state_location, country_location)
#     age_brackets = sp.get_age_brackets_from_df(age_brackets_filepath)
#     age_by_brackets_dic = sp.get_age_by_brackets_dic(age_brackets)

#     contact_matrix_dic = sp.get_contact_matrix_dic(datadir, sheet_name='United States of America')

#     # Need to instead get syn_schools now
#     syn_schools, syn_school_uids, syn_school_types = sp.send_students_to_school(school_sizes, uids_in_school, uids_in_school_by_age,
#                                                                                 ages_in_school_count, age_brackets, age_by_brackets_dic,
#                                                                                 contact_matrix_dic)

#     employment_rates = sp.get_employment_rates(datadir, location=location, state_location=state_location,
#                                                country_location=country_location, use_default=True)

#     age_by_uid_dic = sprw.read_in_age_by_uid(datadir, location, state_location, country_location, folder_name, n)

#     potential_worker_uids, potential_worker_uids_by_age, potential_worker_ages_left_count = sp.get_uids_potential_workers(
#         syn_school_uids, employment_rates, age_by_uid_dic)

#     workers_by_age_to_assign_count = sp.get_workers_by_age_to_assign(employment_rates, potential_worker_ages_left_count,
#                                                                      age_by_uid_dic)

#     workplace_size_brackets = sp.get_workplace_size_brackets(datadir, location, state_location, country_location,
#                                                              use_default=True)

#     workplace_size_distr_by_brackets = sp.get_workplace_size_distr_by_brackets(datadir,
#                                                                                state_location=state_location,
#                                                                                country_location=country_location,
#                                                                                use_default=True)
#     workplace_sizes = sp.generate_workplace_sizes(workplace_size_distr_by_brackets, workplace_size_brackets,
#                                                   workers_by_age_to_assign_count)

#     return workers_by_age_to_assign_count, workplace_size_brackets, workplace_size_distr_by_brackets, workplace_sizes


# @pytest.mark.skip
# def test_assign_rest_of_workers(state_location='Washington', country_location='usa'):
#     workers_by_age_to_assign_count, workplace_size_brackets, workplace_size_distr_by_brackets, \
#     workplace_sizes = test_generate_workplace_sizes()

#     potential_worker_uids, potential_worker_uids_by_age, employment_rates, age_by_uid_dic = test_get_uids_potential_workers()

#     contact_matrix_dic = sp.get_contact_matrix_dic(datadir, sheet_name='United States of America')

#     age_brackets_16 = sp.get_census_age_brackets(datadir, state_location, country_location)
#     age_by_brackets_dic_16 = sp.get_age_by_brackets_dic(age_brackets_16)

#     syn_workplaces, syn_workplace_uids, potential_worker_uids, potential_worker_uids_by_age, workers_by_age_to_assign_count = sp.assign_rest_of_workers(
#         workplace_sizes, potential_worker_uids,
#         potential_worker_uids_by_age,
#         workers_by_age_to_assign_count,
#         dict(age_by_uid_dic), age_brackets_16, age_by_brackets_dic_16,
#         contact_matrix_dic)

#     # TODO: Issue #116 assign_rest_of_workers returns empty syn_workplaces and syn_workplace_uids
#     # syn_workplaces should return a list of lists where each sublist is a workplace with the ages of workers, not empty
#     # for workplace in syn_workplaces:
#     #     assert workplace is not None
#     # assert syn_workplaces != []

#     # syn_worplace_uids should be a list of workers ids, not empty
#     # assert syn_workplace_uids != []

#     # potential_worker_uids should return a list of potential worker ids
#     for worker_id in potential_worker_uids:
#         assert worker_id is not None

#     # potential_worker_uids_by_age should return a list of potential worker ids mapped by age
#     for worker_by_age in potential_worker_uids_by_age:
#         assert int(worker_by_age)

#     # workers_by_age_to_assign_count should be a dictionary mapping age to the count of workers left to assign
#     for worker in workers_by_age_to_assign_count.items():
#         assert tuple(worker)


@pytest.mark.skip('Uses file loading not supported')
def test_generate_school_sizes(location='seattle_metro', state_location='Washington',
                               country_location='usa', folder_name='contact_networks'):
    Nhomes = 10000
    uids_in_school = spsch.get_uids_in_school(datadir, Nhomes, location,
                                           state_location,
                                           country_location,
                                           folder_name=folder_name,
                                           use_default=True)

    school_size_distr_by_bracket = spdd.get_school_size_distr_by_brackets(datadir, location, state_location,
                                                                        country_location)
    school_size_brackets = spdd.get_school_size_brackets(datadir, location, state_location, country_location)
    school_sizes = spsch.generate_school_sizes(school_size_distr_by_bracket, school_size_brackets, uids_in_school)
    assert school_sizes is not None


# @pytest.mark.skip
# def test_assign_teachers_to_work(location='seattle_metro', state_location='Washington',
#                                  country_location='usa', folder_name='contact_networks', n=10000):
#     # Assign students to school
#     gen_schools, gen_school_uids = test_send_students_to_school()

#     employment_rates = sp.get_employment_rates(datadir, location=location, state_location=state_location,
#                                                country_location=country_location, use_default=True)

#     age_by_uid_dic = sp.read_in_age_by_uid(datadir, location, state_location, country_location, folder_name, n)

#     uids_in_school = sp.get_uids_in_school(datadir, n, location, state_location, country_location, folder_name=folder_name, use_default=True)

#     potential_worker_uids, potential_worker_uids_by_age, \
#     potential_worker_ages_left_count = sp.get_uids_potential_workers(uids_in_school, employment_rates, age_by_uid_dic)

#     workers_by_age_to_assign_count = sp.get_workers_by_age_to_assign(employment_rates, potential_worker_ages_left_count,
#                                                                      age_by_uid_dic)

#     # Assign teachers and update school lists
#     syn_schools, syn_school_uids, potential_worker_uids, potential_worker_uids_by_age, \
#     workers_by_age_to_assign_count = sp.assign_teachers_to_work(gen_schools, gen_school_uids, employment_rates,
#                                                                 workers_by_age_to_assign_count,
#                                                                 potential_worker_uids, potential_worker_uids_by_age,
#                                                                 potential_worker_ages_left_count,
#                                                                 student_teacher_ratio=30, teacher_age_min=25,
#                                                                 teacher_age_max=75)

    # for n in range(len(syn_schools)):
    #     print(syn_schools[n])
    #     assert syn_schools[n] is not None
    #     assert syn_school_uids[n] is not None

    # assert syn_schools == gen_schools
    # assert syn_school_uids == gen_school_uids
    # assert potential_worker_uids == potential_worker_uids
    # assert potential_worker_uids_by_age == potential_worker_uids_by_age
    # assert workers_by_age_to_assign_count == workers_by_age_to_assign_count


# %% Run as a script
if __name__ == '__main__':
    sc.tic()

    datadir = sp.settings.datadir
    n = 1000
    location = 'seattle_metro'  # for census distributions
    state_location = 'Washington'  # for state wide age mixing patterns
    country_location = 'usa'
    # location = 'portland_metro'
    # state_location = 'Oregon'
    # location = 'Dakar'
    # state_location = 'Dakar'
    # country_location = 'Senegal'
    folder_name = 'contact_networks'

    # We currently only have files for USA for this data
    # test_all(location, state_location, country_location)
    # test_n_single_ages(1e4, location, state_location, country_location)
    # test_multiple_ages(1e4, location, state_location, country_location)
    # test_get_uids_in_school(location, state_location, country_location)
    # test_send_students_to_school(n=10000, location=location, state_location=state_location,
    #                              country_location=country_location)
    # We currently have files for both Senegal and USA for this data
    test_resample_age()
    # test_generate_household_sizes()
    # test_generate_household_sizes_from_fixed_pop_size()
    # test_generate_all_households()
    # test_get_totalpopsizes_from_household_sizes()
    # test_assign_uids_by_homes()
    # test_get_school_enrollment_rates_path()
    # test_get_uids_potential_workers()
    # test_generate_workplace_sizes()
    # test_generate_school_sizes()

    # For US data only
    # ages, sexes = sp.get_usa_age_sex_n(datadir, location, state_location, country_location, 1e2)
    # print(ages, sexes)

    # country_location = 'Algeria'
    # age_brackets_filepath = sp.get_census_age_brackets_path(sp.datadir,country_location)
    # age_brackets = sp.get_age_brackets_from_df(age_brackets_filepath)
    # print(age_brackets)
    sc.toc()

print('Done.')
