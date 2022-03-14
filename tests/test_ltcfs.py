"""
Test ltcf changes.
"""

import numpy as np
import sciris as sc
import synthpops as sp
import synthpops.plotting as sppl
import matplotlib as mplt
import matplotlib.pyplot as plt
import pytest
import settings


pars = sc.objdict(
            n                       = settings.pop_sizes.small_medium,
            rand_seed               = 123,

            with_facilities         = 1,
            with_non_teaching_staff = 1,

            school_mixing_type      = 'age_and_class_clustered',
)


def test_ltcf_resident_to_staff_ratios(do_show=False):
    """
    Compare the ltcf resident to staff ratios generated to the data available for the location.
    """
    sp.logger.info(f'Testing the ratios of ltcf residents to staff.')

    # to actually match decently, you need to model a higher population size, but for regular
    # test purposes, large sizes will be quite a lot to test on every merge
    pop = sp.Pop(**pars)
    popdict = pop.to_dict()
    loc_pars = pop.loc_pars

    expected_ltcf_ratio_distr = sp.get_long_term_care_facility_resident_to_staff_ratios_distr(**loc_pars)
    resident_to_staff_ratio_brackets = sp.get_long_term_care_facility_resident_to_staff_ratios_brackets(**loc_pars)

    bins = [resident_to_staff_ratio_brackets[i][0] - 1 for i in range(len(resident_to_staff_ratio_brackets))] + [resident_to_staff_ratio_brackets[max(resident_to_staff_ratio_brackets.keys())][0]]

    ltcfs = {}
    for i, person in popdict.items():
        if person['ltcfid'] is not None:
            ltcfs.setdefault(person['ltcfid'], {'residents': [], 'staff': []})
            if person['ltcf_res'] is not None:
                ltcfs[person['ltcfid']]['residents'].append(i)
            elif person['ltcf_staff']:
                ltcfs[person['ltcfid']]['staff'].append(i)

    gen_ratios = []
    for l in ltcfs:

        ratio = len(ltcfs[l]['residents']) / len(ltcfs[l]['staff'])
        gen_ratios.append(ratio)

    hist, bins = np.histogram(gen_ratios, bins=bins)
    gen_ltcf_ratio_distr = {i: hist[i] / sum(hist) for i in range(len(hist))}

    xlabels = [f'{resident_to_staff_ratio_brackets[b][0]:.0f}' for b in resident_to_staff_ratio_brackets]

    fig, ax = sppl.plot_array(list(expected_ltcf_ratio_distr.values()),
                              generated=list(gen_ltcf_ratio_distr.values()),
                              names=xlabels, do_show=False, binned=True)

    ax.set_xlabel('LTCF Resident to Staff Ratio')
    ax.set_title('Comparison of LTCF Resident to Staff Ratios')
    ax.set_ylim(0., 1.)

    if do_show:
        plt.show()


def test_ltcf_resident_ages(do_show=False):
    """
    Compare the ltcf resident ages generated with those expected for the location.
    """
    sp.logger.info(f"Testing that ltcf resident ages align with the expected resident ages for the location.")
    test_pars = sc.dcp(pars)

    # to actually match decently, you need to model a higher population size, but for regular
    # test purposes, large sizes will be quite a lot to test on every merge
    test_pars['n'] = settings.pop_sizes.large
    pop = sp.Pop(**test_pars)
    pop_dict = pop.to_dict()

    ltcf_resident_rates_by_age = sp.get_long_term_care_facility_use_rates(sp.settings.datadir,
                                                                          country_location=pop.country_location,
                                                                          state_location=pop.state_location,
                                                                          )
    ltcf_resident_ages = dict.fromkeys(ltcf_resident_rates_by_age.keys(), 0)
    age_count = pop.count_pop_ages()

    for i, person in pop_dict.items():
        if person['ltcf_res']:
            ltcf_resident_ages[person['age']] += 1

    gen_ltcf_rates = {a: ltcf_resident_ages[a] / age_count[a] for a in ltcf_resident_ages}
    fig, ax = sppl.plot_array([ltcf_resident_rates_by_age[a] for a in age_count], 
        generated=[gen_ltcf_rates[a] for a in age_count],
        do_show=False, binned=True)
    ax.set_xlabel('LTCF Resident Ages')
    ax.set_title('LTCF Resident Use Rates by Age')
    ax.set_ylim(0., 1.)
    ax.set_xlim(0., 100)

    if do_show:
        plt.show()


def test_ltcf_two_group_reduction_off():
    """
    Test that populations can be created with ltcfs that don't split people
    into two groups (residents and staff) to create edges based on role.
    """
    sp.logger.info("Testing population generation with ltcf connections created randomly instead of based on role. This means there is no guarantee that every resident is connected with a staff member. ")
    test_pars = sc.dcp(pars)
    test_pars.use_two_group_reduction = False
    pop = sp.Pop(**test_pars)

    assert len(pop.popdict) == test_pars.n, 'Check failed. Did not generate the right number of people.'
    print('Check passed.')


if __name__ == '__main__':

    sc.tic()
    test_ltcf_resident_to_staff_ratios(do_show=True)
    test_ltcf_resident_ages(do_show=True)

    sc.toc()