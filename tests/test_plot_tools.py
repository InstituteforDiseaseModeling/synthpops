"""
Plot the generated age-specific contact matrix.
"""

import synthpops as sp
# import sciris as sc

import matplotlib as mplt
import matplotlib.pyplot as plt
import cmocean
import cmasher as cmr
import seaborn as sns

import os
from collections import Counter
import pytest


if not sp.config.full_data_available:
    pytest.skip("Data not available, tests not possible", allow_module_level=True)

# Pretty fonts
username = os.path.expanduser('~')

fontdirdic = {
    'dmistry': os.path.join(username, 'Dropbox (IDM)', 'GoogleFonts'),
    'cliffk': os.path.join(username, 'idm', 'covid-19', 'GoogleFonts'),
}
if username not in fontdirdic:
    fontdirdic[username] = os.path.join(username, 'Dropbox', 'COVID-19', 'GoogleFonts')

try:
    fontpath = fontdirdic[username]
    fontstyle = 'Roboto_Condensed'
    fontstyle_path = os.path.join(fontpath, fontstyle, fontstyle.replace('_', '') + '-Light.ttf')
    mplt.rcParams['font.family'] = fontstyle.replace('_', ' ')
except:
    mplt.rcParams['font.family'] = 'Roboto'
mplt.rcParams['font.size'] = 16

# try:
#     username = os.path.split(os.path.expanduser('~'))[-1]
#     fontdirdict = {
#         'dmistry': '/home/dmistry/Dropbox (IDM)/GoogleFonts',
#         'cliffk': '/home/cliffk/idm/covid-19/GoogleFonts',
#     }
#     if username not in fontdirdict:
#         fontdirdict[username] = os.path.expanduser(os.path.expanduser('~'), 'Dropbox', 'GoogleFonts')

#     font_path = fontdirdict[username]

#     fontpath = fontdirdict[username]
#     font_style = 'Roboto_Condensed'
#     fontstyle_path = os.path.join(fontpath, font_style, font_style.replace('_', '') + '-Light.ttf')
#     prop = font_manager.FontProperties(fname=fontstyle_path)
#     mplt.rcParams['font.family'] = prop.get_name()
# except:
#     mplt.rcParams['font.family'] = 'Roboto'


def test_plot_generated_contact_matrix(setting_code='H', n=5000, aggregate_flag=True, logcolors_flag=True,
                                       density_or_frequency='density', with_facilities=False, cmap='cmr.freeze_r', fontsize=16, rotation=50):
    datadir = sp.datadir

    state_location = 'Washington'
    location = 'seattle_metro'
    country_location = 'usa'

    # popdict = {}
    options_args = {'use_microstructure': True}
    network_distr_args = {'Npop': int(n)}

    population = sp.make_population(n, generate=True, with_facilities=with_facilities)

    # contacts = sp.make_contacts(popdict, state_location=state_location, location=location, options_args=options_args,
    #                             network_distr_args=network_distr_args)

    age_brackets = sp.get_census_age_brackets(datadir, state_location=state_location, country_location=country_location)
    age_by_brackets_dic = sp.get_age_by_brackets_dic(age_brackets)

    ages = []
    for uid in population:
        ages.append(population[uid]['age'])

    age_count = Counter(ages)
    aggregate_age_count = sp.get_aggregate_ages(age_count, age_by_brackets_dic)

    matrix = sp.calculate_contact_matrix(population, density_or_frequency, setting_code)

    fig = sp.plot_contact_frequency(matrix, age_count, aggregate_age_count, age_brackets, age_by_brackets_dic,
                                    setting_code, density_or_frequency, logcolors_flag, aggregate_flag, cmap, fontsize, rotation)

    return fig


def test_plot_generated_trimmed_contact_matrix(setting_code='H', n=5000, aggregate_flag=True, logcolors_flag=True,
                                               density_or_frequency='density', with_facilities=False, cmap='cmr.freeze_r', fontsize=16, rotation=50):
    datadir = sp.datadir

    state_location = 'Washington'
    location = 'seattle_metro'
    country_location = 'usa'

    # popdict = {}

    options_args = {'use_microstructure': True}
    network_distr_args = {'Npop': int(n)}
    # contacts = sp.make_contacts(popdict, state_location=state_location, location=location, options_args=options_args,
    #                             network_distr_args=network_distr_args)
    # contacts = sp.trim_contacts(contacts, trimmed_size_dic=None, use_clusters=False)

    population = sp.make_population(n, generate=True, with_facilities=with_facilities)

    age_brackets = sp.get_census_age_brackets(datadir, state_location=state_location, country_location=country_location)
    age_by_brackets_dic = sp.get_age_by_brackets_dic(age_brackets)

    ages = []
    for uid in population:
        ages.append(population[uid]['age'])

    age_count = Counter(ages)
    aggregate_age_count = sp.get_aggregate_ages(age_count, age_by_brackets_dic)

    matrix = sp.calculate_contact_matrix(population, density_or_frequency, setting_code)

    fig = sp.plot_contact_frequency(matrix, age_count, aggregate_age_count, age_brackets, age_by_brackets_dic,
                                    setting_code, density_or_frequency, logcolors_flag, aggregate_flag, cmap, fontsize, rotation)

    return fig


if __name__ == '__main__':

    datadir = sp.datadir

    n = int(200000)

    state_location = 'Washington'
    location = 'seattle_metro'
    country_location = 'usa'

    # setting_code = 'H'
    # setting_code = 'S'
    # setting_code = 'W'
    setting_code = 'LTCF'

    aggregate_flag = True
    # aggregate_flag = False
    logcolors_flag = True
    # logcolors_flag = False

    density_or_frequency = 'density'
    # density_or_frequency = 'frequency'

    with_facilities = True
    # with_facilities = False

    # some plotting styles
    cmap = 'cmr.freeze_r'
    fontsize = 26
    rotation = 90

    do_save = True
    # do_save = False

    do_trimmed = True
    # do_trimmed = False

    if setting_code in ['S', 'W']:
        if do_trimmed:
            fig = test_plot_generated_trimmed_contact_matrix(setting_code, n, aggregate_flag, logcolors_flag, density_or_frequency, with_facilities, cmap, fontsize, rotation)
        else:
            fig = test_plot_generated_contact_matrix(setting_code, n, aggregate_flag, logcolors_flag, density_or_frequency, with_facilities, cmap, fontsize, rotation)

    elif setting_code in ['H', 'LTCF']:
        fig = test_plot_generated_contact_matrix(setting_code, n, aggregate_flag, logcolors_flag, density_or_frequency, with_facilities, cmap, fontsize, rotation)

    if do_save:
        fig_path = datadir.replace('data', 'figures')
        os.makedirs(fig_path, exist_ok=True)

        if setting_code in ['S', 'W']:
            if do_trimmed:
                if aggregate_flag:
                    fig_path = os.path.join(fig_path, 'contact_matrices_152_countries', country_location,
                                            state_location, location + '_npop_' + str(n) + '_' + density_or_frequency + '_close_contact_matrix_setting_' + setting_code + '_aggregate_age_brackets.pdf')
                else:
                    fig_path = os.path.join(fig_path, 'contact_matrices_152_countries', country_location,
                                            state_location, location + '_npop_' + str(n) + '_' + density_or_frequency + '_close_contact_matrix_setting_' + setting_code + '.pdf')
            else:
                if aggregate_flag:
                    fig_path = os.path.join(fig_path, 'contact_matrices_152_countries', country_location,
                                            state_location, location + '_npop_' + str(n) + '_' + density_or_frequency + '_contact_matrix_setting_' + setting_code + '_aggregate_age_brackets.pdf')
                else:
                    fig_path = os.path.join(fig_path, 'contact_matrices_152_countries', country_location,
                                            state_location, location + '_npop_' + str(n) + '_' + density_or_frequency + '_contact_matrix_setting_' + setting_code + '.pdf')

        elif setting_code in ['H', 'LTCF']:
            if aggregate_flag:
                fig_path = os.path.join(fig_path, 'contact_matrices_152_countries', country_location, state_location,
                                        location + '_npop_' + str(n) + '_' + density_or_frequency + '_contact_matrix_setting_' + setting_code + '_aggregate_age_brackets.pdf')
            else:
                fig_path = os.path.join(fig_path, 'contact_matrices_152_countries', country_location, state_location,
                                        location + '_npop_' + str(n) + '_' + density_or_frequency + '_contact_matrix_setting_' + setting_code + '.pdf')

        fig.savefig(fig_path, format='pdf')
        fig.savefig(fig_path.replace('pdf', 'png'), format='png')
        fig.savefig(fig_path.replace('pdf', 'svg'), format='svg')
