"""
Compare the demographics of the generated population to the expected demographic distributions.
"""
import sciris as sc
import synthpops as sp
import numpy as np
import matplotlib as mplt


# parameters to generate a test population
pars = dict(
    n                               = 1e3,
    rand_seed                       = 123,
    max_contacts                    = None,

    country_location                = 'usa',
    state_location                  = 'Washington',
    location                        = 'seattle_metro',
    use_default                     = True,

    with_industry_code              = 0,
    with_facilities                 = 1,
    with_non_teaching_staff         = 1,
    use_two_group_reduction         = 1,
    with_school_types               = 1,

    average_LTCF_degree             = 20,
    ltcf_staff_age_min              = 20,
    ltcf_staff_age_max              = 60,

    school_mixing_type              = {'pk-es': 'age_and_class_clustered', 'ms': 'age_and_class_clustered', 'hs': 'random', 'uv': 'random'},  # you should know what school types you're working with
    average_class_size              = 20,
    inter_grade_mixing              = 0.1,
    teacher_age_min                 = 25,
    teacher_age_max                 = 75,
    staff_age_min                   = 20,
    staff_age_max                   = 75,

    average_student_teacher_ratio   = 20,
    average_teacher_teacher_degree  = 3,
    average_student_all_staff_ratio = 15,
    average_additional_staff_degree = 20,
)


def test_age_distribution_used():
    """
    Test that the age distribution used in sp.Pop.generate() is the expected one for the location specified.
    """
    sp.logger.info("Test that the age distribution used in sp.Pop.generate() are the expected age distributions. \nThis should be binned to the default number of age brackets (cfg.nbrackets).")

    pop = sp.make_population(**pars)

    age_distr = sp.read_age_bracket_distr(sp.datadir, location=pars['location'], state_location=pars['state_location'], country_location=pars['country_location'])
    assert len(age_distr) == sp.config.nbrackets, f'Check failed, len(age_distr_1): {len(age_distr)} does not match sp.config.nbrackets: {sp.config.nbrackets}.'
    print(f'Check passed, len(age_distr_1): {len(age_distr)} == sp.config.nbrackets: {sp.config.nbrackets}.')

    return pop


def test_age_brackets_used_with_contact_matrix():
    """
    Test that the age brackets used in sp.Pop.generate() matches the contact matrices used.

    Note:
        This is a test to ensure that within sp.Pop.generate() uses the right age brackets. By default, without specifying nbrackets in sp.get_census_age_brackets(), the number of age brackets will not match the granularity of the contact matrix.

    """

    sp.logger.info("Test that the age brackets used in sp.Pop.generate() with the contact matrices have the same number of bins as the contact matrices.")

    pop_obj = sp.Pop(**pars)
    sheet_name = pop_obj.sheet_name
    pop = pop_obj.to_dict()  # this is basically what sp.make_population does...

    contact_matrix_dic = sp.get_contact_matrix_dic(sp.datadir, sheet_name=sheet_name)
    contact_matrix_nbrackets = contact_matrix_dic[list(contact_matrix_dic.keys())[0]].shape[0]
    cm_age_brackets = sp.get_census_age_brackets(sp.datadir, country_location=pop_obj.country_location, state_location=pop_obj.state_location, location=pop_obj.location, nbrackets=contact_matrix_nbrackets)
    assert contact_matrix_nbrackets == len(cm_age_brackets), f'Check failed, len(contact_matrix_nbrackets): {contact_matrix_nbrackets} does not match len(cm_age_brackets): {len(cm_age_brackets)}.'
    print(f'Check passed. The age brackets loaded match the number of age brackets for the contact matrices used for the location.')


def test_older_ages_have_household_contacts():
    """
    Test that older age groups (85+) have at least some household contacts with
    other older individuals if expected. Together, if sp.Pop.generate() uses the
    incorrect number of age brackets with the contact matrices, older age groups
    will not be generated as household contacts for each other (when we look at
    the generated contact matrix for households, the blocks between 85+ year
    olds would then be 0 for relatively large populations, even though the
    household contact matrix would have us expect otherwise.)
    """
    test_pars = sc.dcp(pars)
    test_pars['n'] = 20e3

    pop = sp.Pop(**test_pars)
    pop_dict = pop.to_dict()

    contact_matrix_dic = sp.get_contact_matrix_dic(sp.datadir, sheet_name=pop.sheet_name)

    contact_matrix_nbrackets = contact_matrix_dic[list(contact_matrix_dic.keys())[0]].shape[0]
    cm_age_brackets = sp.get_census_age_brackets(sp.datadir, country_location=pop.country_location, state_location=pop.state_location, location=pop.location, nbrackets=contact_matrix_nbrackets)
    cm_age_by_brackets_dic = sp.get_age_by_brackets_dic(cm_age_brackets)

    age_threshold = 85
    age_threshold_bracket = cm_age_by_brackets_dic[age_threshold]

    expected_older_contact = np.sum(contact_matrix_dic['H'][age_threshold_bracket:, age_threshold_bracket:])

    matrix = sp.calculate_contact_matrix(pop_dict, setting_code='H')

    gen_older_age_contacts = np.sum(matrix[age_threshold:, age_threshold:])
    if expected_older_contact != 0:
        assert gen_older_age_contacts != 0, f'Check failed, individuals over {age_threshold} years old have no contacts with each other in households even though the household contact matrix expects them to.'

    else:
        assert gen_older_age_contacts == 0, f'Check failed, individuals over {age_threshold} years old have {gen_older_age_contacts} household contacts with each other even though the household contact matrix expects them to have none.'
    print('Check passed.')


if __name__ == '__main__':

    pop = test_age_distribution_used()
    test_age_brackets_used_with_contact_matrix()
    test_older_ages_have_household_contacts()
