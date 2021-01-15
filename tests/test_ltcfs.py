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


pars = dict(n                               = 20e3,
            rand_seed                       = 0,
            max_contacts                    = None,
            # generate                        = True,
            location                        = 'seattle_metro',
            state_location                  = 'Washington',
            country_location                = 'usa',

            with_industry_code              = 0,
            with_facilities                 = 1,
            with_non_teaching_staff         = 1,
            use_two_group_reduction         = 1,
            with_school_types               = 0,

            average_LTCF_degree             = 20,
            ltcf_staff_age_min              = 20,
            ltcf_staff_age_max              = 60,

            school_mixing_type              = 'age_and_class_clustered',
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


def test_ltcf_resident_to_staff_ratios(do_show=False):
    """
    Compare the ltcf resident to staff ratios generated to the data available for the location.
    """
    sp.logger.info(f'Testing the ratios of ltcf residents to staff.')

    # to actually match decently, you need to model a higher population size, but for regular
    # test purposes, large sizes will be quite a lot to test on every merge
    pop = sp.make_population(**pars)

    expected_ltcf_ratio_distr = sp.get_long_term_care_facility_resident_to_staff_ratios_distr(sp.datadir, location=pars['location'], state_location=pars['state_location'], country_location=pars['country_location'])
    resident_to_staff_ratio_brackets = sp.get_long_term_care_facility_resident_to_staff_ratios_brackets(sp.datadir, location=pars['location'], state_location=pars['state_location'], country_location=pars['country_location'])

    bins = [resident_to_staff_ratio_brackets[i][0] - 1 for i in range(len(resident_to_staff_ratio_brackets))] + [resident_to_staff_ratio_brackets[max(resident_to_staff_ratio_brackets.keys())][0]]

    ltcfs = {}
    for i, person in pop.items():
        if person['snfid'] is not None:
            ltcfs.setdefault(person['snfid'], {'residents': [], 'staff': []})
            if person['snf_res'] is not None:
                ltcfs[person['snfid']]['residents'].append(i)
            elif person['snf_staff']:
                ltcfs[person['snfid']]['staff'].append(i)

    gen_ratios = []
    for l in ltcfs:

        ratio = len(ltcfs[l]['residents']) / len(ltcfs[l]['staff'])
        gen_ratios.append(ratio)

    hist, bins = np.histogram(gen_ratios, bins=bins)
    gen_ltcf_ratio_distr = {i: hist[i] / sum(hist) for i in range(len(hist))}

    xlabels = [f'{resident_to_staff_ratio_brackets[b][0]:.0f}' for b in resident_to_staff_ratio_brackets]

    fig, ax = sppl.plot_array(expected_ltcf_ratio_distr.values(),
                              generated=gen_ltcf_ratio_distr.values(),
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
    test_pars['n'] = 20e3
    pop = sp.Pop(**test_pars)
    pop_dict = pop.to_dict()

    ltcf_resident_rates_by_age = sp.get_long_term_care_facility_use_rates(sp.datadir,
                                                                          country_location=pop.country_location,
                                                                          state_location=pop.state_location,
                                                                          )
    expected_ltcf_rates = ltcf_resident_rates_by_age.values()
    ltcf_resident_ages = dict.fromkeys(ltcf_resident_rates_by_age.keys(), 0)
    age_count = dict.fromkeys(ltcf_resident_rates_by_age.keys(), 0)

    for i, person in pop_dict.items():
        age_count[person['age']] += 1
        if person['snf_res']:
            ltcf_resident_ages[person['age']] += 1

    gen_ltcf_rates = {a: ltcf_resident_ages[a] / age_count[a] for a in ltcf_resident_ages}

    width = 8
    height = 6

    fig, ax = sppl.plot_array(expected_ltcf_rates, generated=gen_ltcf_rates.values(), do_show=False, binned=True)
    ax.set_xlabel('LTCF Resident Ages')
    ax.set_title('LTCF Resident Use Rates by Age')
    ax.set_ylim(0., 1.)
    ax.set_xlim(0., 100)

    if do_show:
        plt.show()


if __name__ == '__main__':

    sc.tic()
    test_ltcf_resident_to_staff_ratios(do_show=True)
    test_ltcf_resident_ages(do_show=True)

    sc.toc()