"""
Plot the generated age-specific contact matrix after interventions remove edges.
"""

import covasim as cv
import numpy as np
import sciris as sc
import synthpops as sp

import matplotlib as mplt
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmocean
import cmasher as cmr
import seaborn as sns

import os
from collections import Counter
import pytest

dir_path = os.path.dirname(os.path.realpath(__file__))


# Pretty fonts
try:
    fontstyle = 'Roboto_Condensed'
    mplt.rcParams['font.family'] = fontstyle.replace('_', ' ')
except:
    mplt.rcParams['font.family'] = 'Roboto'
mplt.rcParams['font.size'] = 16


def calculate_contact_matrix(sim, density_or_frequency='density', setting_code='H'):

    setting_code = setting_code.lower()
    ages = sim.people.age
    ages = np.round(ages, 1)
    ages = ages.astype(int)

    max_age = max(ages)
    age_count = Counter(ages)
    age_range = np.arange(max_age+1)

    matrix = np.zeros((max_age+1, max_age+1))

    # loop over everyone
    for p in range(len(sim.people)):
        a = ages[p]
        contacts = sim.people.contacts[setting_code]['p2'][sim.people.contacts[setting_code]['p1'] == p]
        contact_ages = ages[contacts]

        if density_or_frequency == 'frequency':
            for ca in contact_ages:
                matrix[a][ca] += 1. / len(contact_ages)
        elif density_or_frequency == 'density':
            for ca in contact_ages:
                matrix[a][ca] += 1

    return matrix


def plot_contact_matrix_after_intervention(n, n_days, interventions, intervention_name, location='seattle_metro', state_location='Washington', country_location='usa', aggregate_flag=True, logcolors_flag=True, density_or_frequency='density', setting_code='H', cmap='cmr.freeze_r', fontsize=16, rotation=50):
    """
    Args:
        intervention (cv.intervention): a single intervention
    """
    pars = sc.objdict(
        pop_size=n,
        n_days=n_days,
        pop_type='synthpops'
        )

    # sim = sc.objdict()
    sim = cv.Sim(pars=pars, interventions=interventions)
    sim.run()

    age_brackets = sp.get_census_age_brackets(sp.datadir, state_location=state_location, country_location=country_location)
    age_by_brackets_dic = sp.get_age_by_brackets_dic(age_brackets)

    ages = sim.people.age
    ages = np.round(ages, 1)
    ages = ages.astype(int)
    max_age = max(ages)
    age_count = Counter(ages)
    age_count = dict(age_count)
    for i in range(max_age+1):
        if i not in age_count:
            age_count[i] = 0

    aggregate_age_count = sp.get_aggregate_ages(age_count, age_by_brackets_dic)

    matrix = calculate_contact_matrix(sim, density_or_frequency, setting_code)

    fig = sp.plot_contact_matrix(matrix, age_count, aggregate_age_count, age_brackets, age_by_brackets_dic, setting_code, density_or_frequency, logcolors_flag, aggregate_flag, cmap, fontsize, rotation)

    return fig


if __name__ == '__main__':

    n = int(5e3)

    location = 'seattle_metro'
    state_location = 'Washington'
    country_location = 'usa'
    n_days = 60

    setting_code = 'H'

    aggregate_flag = True
    # aggregate_flag = False
    logcolors_flag = True
    # logcolors_flag = False

    density_or_frequency = 'density'
    # density_or_frequency = 'frequency'

    cmap = 'cmr.freeze_r'
    fontsize = 16
    rotation = 50

    # do_save = True
    do_save = False
    do_show = True
    # do_show = False

    ### Define some example interventions ###

    # 1. Dynamic pars
    i00 = cv.test_prob(start_day=5, symp_prob=0.3)
    i01 = cv.dynamic_pars({'beta': {'days': [40, 50], 'vals': [0.005, 0.015]}, 'diag_factor': {'days': 30, 'vals': 0.0}})

    # 2. Sequence
    i02 = cv.sequence(days=[20, 40, 60], interventions=[
                        cv.test_num(daily_tests=[20]*n_days),
                        cv.test_prob(symp_prob=0.0),
                        cv.test_prob(symp_prob=0.2),
                    ])

    # 3. Change beta
    i03 = cv.change_beta([30, 50], [0.0, 1], layers='h')
    i04 = cv.change_beta([30, 40, 60], [0.0, 1.0, 0.5])

    # 4. Clip edges -- should match the change_beta scenarios
    i05 = cv.clip_edges(days=[30, 50], changes={'h':0.0})
    i06 = cv.clip_edges(days=[30, 40], changes=0.0)
    i07 = cv.clip_edges(days=[60, None], changes=0.5)

    # 5. Test number
    i08 = cv.test_num(daily_tests=[100, 100, 100, 0, 0, 0]*(n_days//6))

    # 6. Test probability
    i09 = cv.test_prob(symp_prob=0.1)

    # 7. Contact tracing
    i10 = cv.test_prob(start_day=20, symp_prob=0.01, asymp_prob=0.0, symp_quar_prob=1.0, asymp_quar_prob=1.0, test_delay=0)
    i11 = cv.contact_tracing(start_day=20, trace_probs=dict(h=0.9, s=0.7, w=0.7, c=0.3), trace_time=dict(h=0, s=1, w=1, c=3))

    i12 = cv.clip_edges(days=[18], changes=[0.], layers=['s'])  # Close schools
    i13 = cv.clip_edges(days=[20, 32], changes=[0.7, 0.7], layers=['w', 'c'])  # Reduce work and community
    i14 = cv.clip_edges(days=[32], changes=[0.3], layers=['w'])  # Reduce work and community more
    i15 = cv.clip_edges(days=[45, None], changes=[0.9, 0.9], layers=['w', 'c'])  # Reopen work and community more

    i16 = cv.test_prob(start_day=38, symp_prob=0.01, asymp_prob=0.0, symp_quar_prob=1.0, asymp_quar_prob=1.0, test_delay=2) # Start testing for TTQ
    i17 = cv.contact_tracing(start_day=40, trace_probs=dict(h=0.9, s=0.7, w=0.7, c=0.3), trace_time=dict(h=0, s=1, w=1, c=3)) # Start tracing for TTQ

    interventions = [i12, i14]

    intervention_name = 'intervention'

    fig = plot_contact_matrix_after_intervention(n, n_days, interventions, intervention_name, 
                                                 location=location, state_location=state_location, country_location=country_location, 
                                                 aggregate_flag=aggregate_flag, logcolors_flag=logcolors_flag, 
                                                 density_or_frequency=density_or_frequency, setting_code=setting_code, 
                                                 cmap='cmr.freeze_r', fontsize=16, rotation=50)

    if do_save:
        fig_path = sp.datadir.replace('data', 'figures')
        fig_name = os.path.join(fig_path, 'contact_matrices_152_countries', country_location, state_location, 
                                location + '_npop_' + str(n) + '_' + density_or_frequency + '_' + intervention_name + '.pdf')
        fig.savefig(fig_name, format='pdf')
        fig.savefig(fig_name.replace('pdf', 'png'), format='png')
        fig.savefig(fig_name.replace('pdf', 'svg'), format='svg')

    if do_show:
        plt.show()
