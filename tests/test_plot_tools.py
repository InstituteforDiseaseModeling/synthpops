import synthpops as sp
import sciris as sc
import numpy as np
import math
import copy
import matplotlib as mplt
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.ticker import LogLocator, LogFormatter
import matplotlib.font_manager as font_manager
from mpl_toolkits.axes_grid1 import make_axes_locatable

import functools
import os
from collections import Counter
import pytest

do_save = False

if not sp.config.full_data_available:
    pytest.skip("Data not available, tests not possible", allow_module_level=True)

try:
    username = os.path.split(os.path.expanduser('~'))[-1]
    fontdirdict = {
        'dmistry': '/home/dmistry/Dropbox (IDM)/GoogleFonts',
        'cliffk': '/home/cliffk/idm/covid-19/GoogleFonts',
    }
    if username not in fontdirdict:
        fontdirdict[username] = os.path.expanduser(os.path.expanduser('~'), 'Dropbox', 'GoogleFonts')

    font_path = fontdirdict[username]

    fontpath = fontdirdict[username]
    font_style = 'Roboto_Condensed'
    fontstyle_path = os.path.join(fontpath, font_style, font_style.replace('_', '') + '-Light.ttf')
    prop = font_manager.FontProperties(fname=fontstyle_path)
    mplt.rcParams['font.family'] = prop.get_name()
except:
    mplt.rcParams['font.family'] = 'Roboto'


def test_plot_generated_contact_matrix(setting_code='H', n=5000, aggregate_flag=True, logcolors_flag=True,
                                       density_or_frequency='density'):
    datadir = sp.datadir

    state_location = 'Washington'
    location = 'seattle_metro'
    country_location = 'usa'

    popdict = {}

    options_args = {'use_microstructure': True}
    network_distr_args = {'Npop': int(n)}
    contacts = sp.make_contacts(popdict, state_location=state_location, location=location, options_args=options_args,
                                network_distr_args=network_distr_args)

    age_brackets = sp.get_census_age_brackets(datadir, state_location=state_location, country_location=country_location)
    age_by_brackets_dic = sp.get_age_by_brackets_dic(age_brackets)

    ages = []
    for uid in contacts:
        ages.append(contacts[uid]['age'])

    age_count = Counter(ages)
    aggregate_age_count = sp.get_aggregate_ages(age_count, age_by_brackets_dic)

    freq_matrix_dic = sp.calculate_contact_matrix(contacts, density_or_frequency)

    fig = sp.plot_contact_frequency(freq_matrix_dic, age_count, aggregate_age_count, age_brackets, age_by_brackets_dic,
                                    setting_code, density_or_frequency, logcolors_flag, aggregate_flag)

    return fig


def test_plot_generated_trimmed_contact_matrix(setting_code='H', n=5000, aggregate_flag=True, logcolors_flag=True,
                                               density_or_frequency='density'):
    datadir = sp.datadir

    state_location = 'Washington'
    location = 'seattle_metro'
    country_location = 'usa'

    popdict = {}

    options_args = {'use_microstructure': True}
    network_distr_args = {'Npop': int(n)}
    contacts = sp.make_contacts(popdict, state_location=state_location, location=location, options_args=options_args,
                                network_distr_args=network_distr_args)
    contacts = sp.trim_contacts(contacts, trimmed_size_dic=None, use_clusters=False)

    age_brackets = sp.get_census_age_brackets(datadir, state_location=state_location, country_location=country_location)
    age_by_brackets_dic = sp.get_age_by_brackets_dic(age_brackets)

    ages = []
    for uid in contacts:
        ages.append(contacts[uid]['age'])

    age_count = Counter(ages)
    aggregate_age_count = sp.get_aggregate_ages(age_count, age_by_brackets_dic)

    freq_matrix_dic = sp.calculate_contact_matrix(contacts, density_or_frequency)

    fig = sp.plot_contact_frequency(freq_matrix_dic, age_count, aggregate_age_count, age_brackets, age_by_brackets_dic,
                                    setting_code, density_or_frequency, logcolors_flag, aggregate_flag)

    return fig


if __name__ == '__main__':

    datadir = sp.datadir

    n = int(100000)

    state_location = 'Washington'
    location = 'seattle_metro'
    country_location = 'usa'

    # setting_code = 'H'
    setting_code = 'S'
    # setting_code = 'W'

    aggregate_flag = True
    # aggregate_flag = False
    logcolors_flag = True
    # logcolors_flag = False

    density_or_frequency = 'density'
    # density_or_frequency = 'frequency'

    # do_save = True
    do_save = False

    do_trimmed = True
    # do_trimmed = False

    if setting_code in ['S', 'W']:
        if do_trimmed:
            fig = test_plot_generated_trimmed_contact_matrix(setting_code, n, aggregate_flag, logcolors_flag,
                                                             density_or_frequency)
        else:
            fig = test_plot_generated_contact_matrix(setting_code, n, aggregate_flag, logcolors_flag,
                                                     density_or_frequency)
    elif setting_code == 'H':
        fig = test_plot_generated_contact_matrix(setting_code, n, aggregate_flag, logcolors_flag, density_or_frequency)

    if do_save:
        fig_path = datadir.replace('data', 'figures')
        os.makedirs(fig_path, exist_ok=True)

        if setting_code in ['S', 'W']:
            if do_trimmed:
                if aggregate_flag:
                    fig_path = os.path.join(fig_path, 'contact_matrices_152_countries', country_location,
                                            state_location, location + '_npop_' + str(
                            n) + '_' + density_or_frequency + '_close_contact_matrix_setting_' + setting_code + '_aggregate_age_brackets.pdf')
                else:
                    fig_path = os.path.join(fig_path, 'contact_matrices_152_countries', country_location,
                                            state_location, location + '_npop_' + str(
                            n) + '_' + density_or_frequency + '_close_contact_matrix_setting_' + setting_code + '.pdf')
            else:
                if aggregate_flag:
                    fig_path = os.path.join(fig_path, 'contact_matrices_152_countries', country_location,
                                            state_location, location + '_npop_' + str(
                            n) + '_' + density_or_frequency + '_contact_matrix_setting_' + setting_code + '_aggregate_age_brackets.pdf')
                else:
                    fig_path = os.path.join(fig_path, 'contact_matrices_152_countries', country_location,
                                            state_location, location + '_npop_' + str(
                            n) + '_' + density_or_frequency + '_contact_matrix_setting_' + setting_code + '.pdf')

        elif setting_code == 'H':
            if aggregate_flag:
                fig_path = os.path.join(fig_path, 'contact_matrices_152_countries', country_location, state_location,
                                        location + '_npop_' + str(
                                            n) + '_' + density_or_frequency + '_contact_matrix_setting_' + setting_code + '_aggregate_age_brackets.pdf')
            else:
                fig_path = os.path.join(fig_path, 'contact_matrices_152_countries', country_location, state_location,
                                        location + '_npop_' + str(
                                            n) + '_' + density_or_frequency + '_contact_matrix_setting_' + setting_code + '.pdf')

        fig.savefig(fig_path, format='pdf')
    plt.show()
