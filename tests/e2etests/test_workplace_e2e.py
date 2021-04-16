import synthpops as sp
from synthpops import contact_networks as cn
import sciris as sc
import pytest
import setup_e2e as e2e
from setup_e2e import create_sample_pop_e2e, get_fig_dir_by_module
import scipy
from scipy import stats as st
import numpy as np
import re


def test_work_size_distribution(do_show, do_save, create_sample_pop_e2e, get_fig_dir_by_module):
    sp.logger.info("Test workplace size distribution vs the work_size_count.dat")

    plotting_kwargs = sc.objdict(do_show=do_show,
                                 do_save=do_save,
                                 figdir=get_fig_dir_by_module)

    workplace_brackets_index = sp.get_index_by_brackets_dic(
        sp.get_workplace_size_brackets(**create_sample_pop_e2e.loc_pars))

    actual_workplace_sizes = create_sample_pop_e2e.count_workplace_sizes()
    # actual_workplace_sizes = pop.count_workplace_sizes()
    # count the workplaces by size bracket

    actual_count = {k: 0 for k in set(workplace_brackets_index.values())}
    for i in workplace_brackets_index:
        actual_count[workplace_brackets_index[i]] += actual_workplace_sizes.get(i, 0)

    expected_distr = sp.norm_dic(sp.get_workplace_size_distr_by_brackets(**create_sample_pop_e2e.loc_pars))
    # expected_distr = sp.norm_dic(sp.get_workplace_size_distr_by_brackets(**pop.loc_pars))

    # calculate expected count by using actual number of workplaces
    expected_count = {k: expected_distr[k]*sum(actual_count.values())for k in expected_distr}
    # perform statistical check
    sp.statistic_test([expected_count[i] for i in sorted(expected_count)], [actual_count[i] for i in sorted(actual_count)])

    # pop.plot_workplace_sizes(**plotting_kwargs)
    create_sample_pop_e2e.plot_workplace_sizes(**plotting_kwargs)


def test_workplace_contact_distribution(do_show, do_save, create_sample_pop_e2e, get_fig_dir_by_module):
    # calculate the workplace contacts count and plot
    sp.logger.info("Test workplace contact distribution: workers in workplace such that worksize <= max_contacts must have"
                   "contacts equal to worksize-1, for workers in workplace with size > max_contacts, the distribution "
                   "should be closer to poisson distribution with mean = max_contacts ")
    plotting_kwargs = sc.objdict(do_show=do_show,
                                 do_save=do_save,
                                 figdir=get_fig_dir_by_module)
    contacts, contacts_by_id = cn.get_contact_counts_by_layer(create_sample_pop_e2e.popdict, layer="w")
    plotting_kwargs.append("title_prefix", f"Total Workers = {len(contacts.get('wpid').get('all'))}")
    plotting_kwargs.append("figname", f"workers_contact_count")
    sp.plot_contact_counts(contacts, **plotting_kwargs)
    plotting_kwargs.remove("title_prefix")
    plotting_kwargs.remove("figname")

    #check workplace with worksize <= max_contacts
    max_contacts = create_sample_pop_e2e.max_contacts['W']
    upperbound = st.poisson.interval(alpha=0.95, mu=max_contacts)[1]
    group_size_contacts = {
        f'all_worksize_contacts size > {max_contacts//2}': [],  # capture size > max_contacts//2
        f'large_worksize_contacts size > {upperbound}': [],  # capture size > upperbound
        f'medium_large_worksize_contacts size between {max_contacts}, {upperbound}': [],  # capture size between max_contacts and upperbound
        f'small_medium_worksize_contacts size between {max_contacts//2}, {max_contacts}': [],  # capture size between max_contacts//2 and max_contacts
    }
    for k, v in contacts_by_id.items():
        if len(v) <= max_contacts//2:
            assert len([i for i in v if i != len(v)-1 ]) == 0, \
                "Failed, not all contacts in {len(k)} are equal to {len(v)} : {v}"
        else:
            if len(v) > upperbound:
                group_size_contacts[f'large_worksize_contacts size > {upperbound}'] += v
            elif len(v) >= max_contacts:
                group_size_contacts[f'medium_large_worksize_contacts size between {max_contacts}, {upperbound}'] += v
            else:
                group_size_contacts[f'small_medium_worksize_contacts size between {max_contacts//2}, {max_contacts}'] += v
            group_size_contacts[f'all_worksize_contacts size > {max_contacts//2}'] += v

    file_pattern=re.compile(r'([\s><=])')
    for i in group_size_contacts:
        plotting_kwargs["title_prefix"] = i
        plotting_kwargs["figname"]= file_pattern.sub("_", i)
        check_truncated_poisson(testdata=group_size_contacts[i],
                                mu=max_contacts,
                                lowerbound=max_contacts//2,
                                skipcheck=True if "small" in i else True,
                                **plotting_kwargs)




def test_employment_age_distribution(do_show, do_save, create_sample_pop_e2e, get_fig_dir_by_module):
    sp.logger.info("Test employment age distribution vs the employment_rates_by_age.dat")

    plotting_kwargs = sc.objdict(do_show=do_show, do_save=do_save, figdir=get_fig_dir_by_module)
    actual_employment_age_count = create_sample_pop_e2e.count_employment_by_age()
    total_employee = sum(actual_employment_age_count.values())
    expected_employment_age_dist = sp.norm_dic(
        sp.get_employment_rates(**create_sample_pop_e2e.loc_pars))

    expected_employment_age_count = {i: round(expected_employment_age_dist[i] * total_employee)
                                     for i in expected_employment_age_dist}

    # generate list of ages based on the actual count
    generated_actual = sum([[i] * actual_employment_age_count[i] for i in actual_employment_age_count], [])
    generated_expected = sum([[i] * expected_employment_age_count[i] for i in expected_employment_age_count], [])
    # run statistical tests for employment by age distribution
    # TODO: Need to refine the data for fair comparison
    sp.statistic_test(expected=generated_expected, actual=generated_actual, test=st.kstest)
    # plot enrollment by age
    create_sample_pop_e2e.plot_employment_rates_by_age(**plotting_kwargs)


def check_truncated_poisson(testdata, mu, lowerbound=None, upperbound=None, skipcheck=False, **kwargs):
    """
    test if data fits in truncated poisson distribution between upperbound and lowerbound using kstest
    Args:
        testdata (array) : data to be tested
        mu (float) : expected mean for the poisson distribution
        lowerbound (float) : lowerbound for truncation
        upperbound (float) : upperbound for truncation

    Returns:
        None
    """
    sample_size = len(testdata)
    # need to exclude any value below or equal to lowerbound and any value above or equal to upperbound, so we first find the quantile location for
    # lowerbound and upperbound then only generate poisson cdf values in between these 2 locations
    minquantile = st.poisson.cdf(lowerbound, mu=mu) if lowerbound else 0
    maxquantile = st.poisson.cdf(upperbound, mu=mu) if upperbound else 1
    # create uniformly distributed number between minquantile and maxquantile (in the cdf quantile space)
    q = np.random.uniform(low=minquantile, high=maxquantile, size=sample_size)
    # use percent point function to get inverse of cdf
    expected_data = st.poisson.ppf(q=q, mu=mu)
    if not skipcheck:
        sp.statistic_test(expected_data, testdata, test=st.kstest)

    #plot comparison
    actual, bins = np.histogram(testdata, bins=10)
    expected = np.histogram(expected_data, bins=bins)[0]
    kwargs["generated"] = actual
    #merge 11 bins to 10 for bar plot align at center
    merged_bins = [round((bins[np.where(bins == i)[0][0]] + i)/2) for i in bins if np.where(bins == i)[0][0] < len(bins)-1]
    kwargs["xvalue"] = merged_bins
    sp.plot_array(expected, **kwargs)


if __name__ == "__main__":
    # you can pass --do-save --do-show --artifact-dir argument to view/save the figures
    # for running individual tests, you can do this
    # testcase = 'test_employment_age_distribution'
    testcase = 'test_work_size_distribution'
    pytest.main(['-v', '-k', testcase, '--do-show'])
