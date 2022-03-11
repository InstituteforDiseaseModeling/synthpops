"""
Compare the demographics of the generated population to the expected demographic distributions.
"""
import sciris as sc
import synthpops as sp
import numpy as np
import settings


# parameters to generate a test population
pars = sc.objdict(
    n                  = settings.pop_sizes.medium,
    rand_seed          = 123,

    country_location   = 'usa',
    state_location     = 'Washington',
    location           = 'Island_County',
    use_default        = 1,

    household_method   = 'fixed_ages',
    smooth_ages        = 1,

    with_facilities    = 1,
    with_school_types  = 1,
    average_class_size = 18,

    school_mixing_type = {'pk': 'age_and_class_clustered',
                          'es': 'age_and_class_clustered',
                          'ms': 'age_and_class_clustered',
                          'hs': 'random', 'uv': 'random'},  # you should know what school types you're working with

)


def test_age_distribution_used():
    """
    Test that the age distribution used in sp.Pop.generate() is the expected one for the location specified.
    """
    sp.logger.info("Test that the age distribution used in sp.Pop.generate() are the expected age distributions. \nThis should be binned to the default number of age brackets (cfg.nbrackets).")

    pop = sp.Pop(**pars)
    loc_pars = pop.loc_pars
    age_dist = sp.read_age_bracket_distr(**loc_pars)
    assert len(age_dist) == sp.settings.nbrackets, f'Check failed, len(age_dist): {len(age_dist)} does not match sp.config.nbrackets: {sp.config.nbrackets}.'
    print(f'Check passed, len(age_dist): {len(age_dist)} == sp.config.nbrackets: {sp.settings.nbrackets}.')

    return pop


def test_age_brackets_used_with_contact_matrix():
    """
    Test that the age brackets used in sp.Pop.generate() matches the contact matrices used.

    Note:
        This is a test to ensure that within sp.Pop.generate() uses the right age brackets. By default, without specifying nbrackets in sp.get_census_age_brackets(), the number of age brackets will not match the granularity of the contact matrix.

    """

    sp.logger.info("Test that the age brackets used in sp.Pop.generate() with the contact matrices have the same number of bins as the contact matrices.")

    pop = sp.Pop(**pars)
    sheet_name = pop.sheet_name

    loc_pars = pop.loc_pars

    contact_matrices = sp.get_contact_matrices(sp.settings.datadir, sheet_name=sheet_name)
    contact_matrix_nbrackets = contact_matrices[list(contact_matrices.keys())[0]].shape[0]
    cm_age_brackets = sp.get_census_age_brackets(**sc.mergedicts(loc_pars, {'nbrackets': contact_matrix_nbrackets}))
    assert contact_matrix_nbrackets == len(cm_age_brackets), f'Check failed, len(contact_matrix_nbrackets): {contact_matrix_nbrackets} does not match len(cm_age_brackets): {len(cm_age_brackets)}.'
    print(f'Check passed. The age brackets loaded match the number of age brackets for the contact matrices used for the location.')


def test_older_ages_have_household_contacts():
    """
    Test that older age groups (80+) have at least some household contacts with
    other older individuals if expected. Together, if sp.Pop.generate() uses the
    incorrect number of age brackets with the contact matrices, older age groups
    will not be generated as household contacts for each other (when we look at
    the generated contact matrix for households, the blocks between 85+ year
    olds would then be 0 for relatively large populations, even though the
    household contact matrix would have us expect otherwise.)
    """
    test_pars = sc.dcp(pars)

    test_pars.n = settings.pop_sizes.medium_large  # decent size to check older populations in households

    pop = sp.Pop(**test_pars)
    pop_dict = pop.to_dict()
    loc_pars = pop.loc_pars

    contact_matrices = sp.get_contact_matrices(sp.settings.datadir, sheet_name=pop.sheet_name)

    contact_matrix_nbrackets = contact_matrices[list(contact_matrices.keys())[0]].shape[0]
    cm_age_brackets = sp.get_census_age_brackets(**sc.mergedicts(loc_pars, {'nbrackets': contact_matrix_nbrackets}))
    cm_age_by_brackets = sp.get_age_by_brackets(cm_age_brackets)

    age_threshold = 80
    age_threshold_bracket = cm_age_by_brackets[age_threshold]

    expected_older_contact = np.sum(contact_matrices['H'][age_threshold_bracket:, age_threshold_bracket:])

    matrix = sp.calculate_contact_matrix(pop_dict, layer='H')

    gen_older_age_contacts = np.sum(matrix[age_threshold:, age_threshold:])
    if expected_older_contact != 0:
        assert gen_older_age_contacts != 0, f'Check failed, individuals over {age_threshold} years old have no contacts with each other in households even though the household contact matrix expects them to.'

    else:
        assert gen_older_age_contacts == 0, f'Check failed, individuals over {age_threshold} years old have {gen_older_age_contacts} household contacts with each other even though the household contact matrix expects them to have none.'
    print('Check passed.')


def test_plot_contact_matrix(do_show=False):
    """"""
    pop = sp.Pop(**pars)
    pop.plot_contacts(do_show=do_show)


if __name__ == '__main__':

    pop = test_age_distribution_used()
    test_age_brackets_used_with_contact_matrix()
    test_older_ages_have_household_contacts()
    test_plot_contact_matrix()
