import pathlib
import pytest
import sciris as sc
import synthpops as sp
import matplotlib as mplt
from collections import Counter
import setup_e2e as e2e
import scipy
from setup_e2e import create_sample_pop_e2e, get_fig_dir_by_module


def test_household_average_contact_by_age(do_show, do_save, create_sample_pop_e2e, get_fig_dir_by_module):
    sp.logger.info("Test average household contacts by age by plotting the average age mixing matrix.")
    plotting_kwargs = sc.objdict(do_show=do_show, do_save=do_save, figdir=get_fig_dir_by_module)
    fig = create_sample_pop_e2e.plot_contacts(**plotting_kwargs)
    assert isinstance(fig, mplt.figure.Figure), 'Check failed. Figure not generated.'
    mplt.pyplot.close(fig)

def test_age_distribution():
    # todo: require statistics methods
    pass


def test_household_distribution( create_sample_pop_e2e):
    actual_households_count = Counter([len(i['member_uids']) for i in create_sample_pop_e2e.households])
    actual_households_size = [actual_households_count[i] for i in sorted(actual_households_count)]
    expected_households_dist = sp.get_household_size_distr(**create_sample_pop_e2e.loc_pars)
    expected_households_size = [expected_households_dist[i] * create_sample_pop_e2e.n_households for i in sorted(expected_households_dist)]

    # use t-test to compare instead
    test = scipy.stats.ttest_rel
    sp.statistic_test(expected_households_size, actual_households_size, test)


def test_household_head_ages_by_household_size_e2e(do_show, do_save, create_sample_pop_e2e, get_fig_dir_by_module):
    sp.logger.info("Test the age distribution of household heads by the household size.")
    plotting_kwargs = sc.objdict(do_show=do_show, do_save=do_save, figdir=get_fig_dir_by_module)
    fig, ax = sp.plot_household_head_ages_by_size(create_sample_pop_e2e, **plotting_kwargs)
    assert isinstance(fig, mplt.figure.Figure), 'Check failed. Figure not generated.'
    print('Check passed. Figure made.')
    mplt.pyplot.close(fig)

if __name__ == "__main__":
    # you can pass --do-save --do-show --artifact-dir argument to view/save the figures
    # pytest.main(['-v', '--do-show', __file__])

    # for running individual tests, you can do this
    # testcase = 'test_household_head_ages_by_household_size_e2e'
    testcase = 'test_household_distribution'
    pytest.main(['-v', '-k', testcase, '--do-show'])
