import synthpops as sp
from synthpops import contact_networks as cn
import sciris as sc
import pytest
import setup_e2e as e2e
from setup_e2e import create_sample_pop_e2e, get_fig_dir_by_module
import scipy
from scipy import stats as st

def test_work_size_distribution(do_show, do_save, create_sample_pop_e2e, get_fig_dir_by_module):
    sp.logger.info("Test workplace size distribution vs the work_size_count.dat")
    plotting_kwargs = sc.objdict(do_show=do_show, do_save=do_save, figdir=get_fig_dir_by_module)
    workplace_brackets_index = sp.get_index_by_brackets_dic(
        sp.get_workplace_size_brackets(datadir=create_sample_pop_e2e.datadir,
                                       location=create_sample_pop_e2e.location,
                                       state_location=create_sample_pop_e2e.state_location,
                                       country_location=create_sample_pop_e2e.country_location))
    actual_workplace_sizes = create_sample_pop_e2e.count_workplace_sizes()
    # count the workplaces by size bracket
    actual_count = {k:0 for k in set(workplace_brackets_index.values())}
    for i in workplace_brackets_index:
        actual_count[workplace_brackets_index[i]] += actual_workplace_sizes.get(i, 0)
    expected_distr = sp.norm_dic(sp.get_workplace_size_distr_by_brackets(datadir=create_sample_pop_e2e.datadir,
                                            location=create_sample_pop_e2e.location,
                                            state_location=create_sample_pop_e2e.state_location,
                                            country_location=create_sample_pop_e2e.country_location))

    # calculate the workplace contacts count and plot
    contacts = cn.get_contact_counts_by_layer(create_sample_pop_e2e.popdict, layer="w")
    sp.plot_contact_counts(contacts, varname="total_worker", varvalue=len(contacts.get("wpid").get("all")), **plotting_kwargs)
    # calculate expected count by using actual number of workplaces
    expected_count = {k:expected_distr[k]*sum(actual_count.values())for k in expected_distr}
    # perform statistical check
    sp.statistic_test([expected_count[i] for i in sorted(expected_count)], [actual_count[i] for i in sorted(actual_count)])
    #plot workplace size
    create_sample_pop_e2e.plot_workplace_sizes(**plotting_kwargs)

def test_employment_age_distribution(do_show, do_save, create_sample_pop_e2e, get_fig_dir_by_module):
    sp.logger.info("Test employment age distribution vs the employment_rates_by_age.dat")

    plotting_kwargs = sc.objdict(do_show=do_show, do_save=do_save, figdir=get_fig_dir_by_module)
    actual_employment_age_count = create_sample_pop_e2e.count_employment_by_age()
    total_employee = sum(list(actual_employment_age_count.values()))
    expected_employment_age_dist = sp.norm_dic(
        sp.get_employment_rates(datadir=create_sample_pop_e2e.datadir,
                                location=create_sample_pop_e2e.location,
                                state_location=create_sample_pop_e2e.state_location,
                                country_location=create_sample_pop_e2e.country_location))

    expected_employment_age_count = {i: round(expected_employment_age_dist[i] * total_employee)
                                     for i in expected_employment_age_dist}

    # generate list of ages based on the actual count
    generated_actual = sum([[i] * actual_employment_age_count[i] for i in actual_employment_age_count], [])
    generated_expected = sum([[i] * expected_employment_age_count[i] for i in expected_employment_age_count], [])
    # run statistical tests for employment by age distribution
    # TODO: Need to refine the data for fair comparison
    sp.statistic_test(expected=generated_expected, actual=generated_actual, test=st.kstest)
    # plot enrollment by age
    create_sample_pop_e2e.plot_enrollment_rates_by_age(**plotting_kwargs)

if __name__ == "__main__":
    # you can pass --do-save --do-show --artifact-dir argument to view/save the figures
    # for running individual tests, you can do this
    testcase = 'test_employment_age_distribution'
    pytest.main(['-v', '-k', testcase, '--do-show'])