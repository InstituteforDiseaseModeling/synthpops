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

import cmocean
import functools
import os
from collections import Counter
import pytest


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


def calculate_contact_matrix(contacts, density_or_frequency='density', setting_code='H'):
    uids = contacts.keys()
    uids = [uid for uid in uids]

    num_ages = 101

    # F_dic = {}
    M = np.zeros((num_ages, num_ages))
    # for k in ['M', 'H', 'S', 'W', 'C']:
        # F_dic[k] = np.zeros((num_ages, num_ages))

    for n, uid in enumerate(uids):
        # layers = contacts[uid]['contacts']
        age = contacts[uid]['age']
        # for k in layers:
        contact_ages = [contacts[c]['age'] for c in contacts[uid]['contacts'][setting_code]]
        contact_ages = np.array([int(a) for a in contact_ages])

        if len(contact_ages) > 0:
            if density_or_frequency == 'density':
                for ca in contact_ages:
                    M[age, ca] += 1.0/len(contact_ages)
                # F_dic[k][age, contact_ages] += 1 / len(contact_ages)
            elif density_or_frequency == 'frequency':
                for ca in contact_ages:
                    M[age, ca] += 1.0
                # F_dic[k][age, contact_ages] += 1

    return M


def plot_contact_matrix(matrix, age_count, aggregate_age_count, age_brackets, age_by_brackets_dic, setting_code='H', density_or_frequency='density', logcolors_flag=False, aggregate_flag=True):

    # cmap = mplt.cm.get_cmap(cmocean.cm.deep_r)
    cmap = mplt.cm.get_cmap(cmocean.cm.matter_r)

    fig = plt.figure(figsize=(7, 7), tight_layout=True)
    ax = fig.add_subplot(111)

    titles = {'H': 'Household', 'S': 'School', 'W': 'Work'}

    if aggregate_flag:
        num_agebrackets = len(age_brackets)
        aggregate_M = sp.get_aggregate_matrix(matrix, age_by_brackets_dic)
        asymmetric_M = sp.get_asymmetric_matrix(aggregate_M, aggregate_age_count)
    else:
        num_agebrackets = len(age_brackets)
        asymmetric_M = sp.get_asymmetric_matrix(matrix, age_count)

    if logcolors_flag:

        vbounds = {}
        if density_or_frequency == 'density':
            if aggregate_flag:
                vbounds['H'] = {'vmin': 1e-2, 'vmax': 1e-0}
                vbounds['S'] = {'vmin': 1e-3, 'vmax': 1e1}
                vbounds['W'] = {'vmin': 1e-3, 'vmax': 1e1}
            else:
                vbounds['H'] = {'vmin': 1e-3, 'vmax': 1e-1}
                vbounds['S'] = {'vmin': 1e-3, 'vmax': 1e-1}
                vbounds['W'] = {'vmin': 1e-3, 'vmax': 1e-1}

        elif density_or_frequency == 'frequency':
            if aggregate_flag:
                vbounds['H'] = {'vmin': 1e-2, 'vmax': 1e0}
                vbounds['S'] = {'vmin': 1e-3, 'vmax': 1e1}
                vbounds['W'] = {'vmin': 1e-2, 'vmax': 1e0}
            else:
                vbounds['H'] = {'vmin': 1e-2, 'vmax': 1e0}
                vbounds['S'] = {'vmin': 1e-2, 'vmax': 1e0}
                vbounds['W'] = {'vmin': 1e-2, 'vmax': 1e0}
        im = ax.imshow(asymmetric_M.T, origin='lower', interpolation='nearest', cmap=cmap, norm=LogNorm(vmin=vbounds[setting_code]['vmin'], vmax=vbounds[setting_code]['vmax']))
    else:
        im = ax.imshow(asymmetric_M.T, origin='lower', interpolation='nearest', cmap=cmap)
    implot = im

    divider = make_axes_locatable(ax)
    cax = divider.new_horizontal(size="3.5%", pad=0.1)

    fig.add_axes(cax)
    cbar = fig.colorbar(implot, cax=cax)
    cbar.ax.tick_params(axis='y', labelsize=20)
    if density_or_frequency == 'frequency':
        cbar.ax.set_ylabel('Frequency of Contacts', fontsize=20)
    else:
        cbar.ax.set_ylabel('Density of Contacts', fontsize=20)
    ax.tick_params(labelsize=20)
    ax.set_xlabel('Age', fontsize=22)
    ax.set_ylabel('Age of Contacts', fontsize=22)
    ax.set_title(titles[setting_code] + ' Contact Patterns', fontsize=28)

    if aggregate_flag:
        tick_labels = [str(age_brackets[b][0]) + '-' + str(age_brackets[b][-1]) for b in age_brackets]
        ax.set_xticks(np.arange(len(tick_labels)))
        ax.set_xticklabels(tick_labels, fontsize=18)
        ax.set_xticklabels(tick_labels, fontsize=18, rotation=50)
        ax.set_yticks(np.arange(len(tick_labels)))
        ax.set_yticklabels(tick_labels, fontsize=18)

    return fig


def plot_generated_contact_matrix(datadir, n, location='seattle_metro', state_location='Washington', country_location='usa', setting_code='H', aggregate_flag=True, logcolors_flag=True, density_or_frequency='density'):

    popdict = {}

    options_args = {'use_microstructure': True}
    network_distr_args = {'Npop': int(n)}
    contacts = sp.make_contacts(popdict, country_location=country_location, state_location=state_location, location=location, options_args=options_args, network_distr_args=network_distr_args)

    age_brackets = sp.get_census_age_brackets(datadir, state_location=state_location, country_location=country_location)
    age_by_brackets_dic = sp.get_age_by_brackets_dic(age_brackets)

    ages = []
    for uid in contacts:
        ages.append(contacts[uid]['age'])

    num_agebrackets = len(age_brackets)

    age_count = Counter(ages)
    aggregate_age_count = sp.get_aggregate_ages(age_count, age_by_brackets_dic)

    symmetric_matrix = calculate_contact_matrix(contacts, density_or_frequency, setting_code)

    fig = plot_contact_matrix(symmetric_matrix, age_count, aggregate_age_count, age_brackets, age_by_brackets_dic, setting_code=setting_code, density_or_frequency=density_or_frequency, logcolors_flag=logcolors_flag, aggregate_flag=aggregate_flag)
    return fig


def plot_generated_trimmed_contact_matrix(datadir, n, location='seattle_metro', state_location='Washington', country_location='usa', setting_code='H', aggregate_flag=True, logcolors_flag=True, density_or_frequency='density', trimmed_size_dic=None):

    popdict = {}

    options_args = {'use_microstructure': True}
    network_distr_args = {'Npop': int(n)}
    contacts = sp.make_contacts(popdict, country_location=country_location, state_location=state_location, location=location, options_args=options_args, network_distr_args=network_distr_args)
    contacts = sp.trim_contacts(contacts, trimmed_size_dic=trimmed_size_dic, use_clusters=False)

    age_brackets = sp.get_census_age_brackets(datadir, state_location=state_location, country_location=country_location)
    age_by_brackets_dic = sp.get_age_by_brackets_dic(age_brackets)

    ages = []
    for uid in contacts:
        ages.append(contacts[uid]['age'])

    num_agebrackets = len(age_brackets)

    age_count = Counter(ages)
    aggregate_age_count = sp.get_aggregate_ages(age_count, age_by_brackets_dic)

    symmetric_matrix = calculate_contact_matrix(contacts, density_or_frequency, setting_code)

    fig = plot_contact_matrix(symmetric_matrix, age_count, aggregate_age_count, age_brackets, age_by_brackets_dic, setting_code=setting_code, density_or_frequency=density_or_frequency, logcolors_flag=logcolors_flag, aggregate_flag=aggregate_flag)
    return fig


def plot_data_contact_matrix(datadir, location='seattle_metro', state_location='Washington', country_location='usa', sheet_name='United States of America', setting_code='H', logcolors_flag=True):

    asymmetric_M = sp.get_contact_matrix(datadir, setting_code, sheet_name=sheet_name)

    age_brackets = sp.get_census_age_brackets(datadir, state_location=state_location, country_location=country_location)
    age_by_brackets_dic = sp.get_age_by_brackets_dic(age_brackets)

    cmap = mplt.cm.get_cmap(cmocean.cm.matter_r)

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111)

    titles = {'H': 'Household', 'S': 'School', 'W': 'Work'}

    if logcolors_flag:

        vbounds = {}
        if density_or_frequency == 'density':
            # if aggregate_flag:
            vbounds['H'] = {'vmin': 1e-2, 'vmax': 1e1}
            vbounds['S'] = {'vmin': 1e-3, 'vmax': 1e-0}
            vbounds['W'] = {'vmin': 1e-3, 'vmax': 1e-0}
            # else:
                # vbounds['H'] = {'vmin': 1e-3, 'vmax': 1e-1}
                # vbounds['S'] = {'vmin': 1e-3, 'vmax': 1e-1}
                # vbounds['W'] = {'vmin': 1e-3, 'vmax': 1e-1}

        elif density_or_frequency == 'frequency':
            # if aggregate_flag:
            vbounds['H'] = {'vmin': 1e-2, 'vmax': 1e0}
            vbounds['S'] = {'vmin': 1e-2, 'vmax': 1e1}
            vbounds['W'] = {'vmin': 1e-2, 'vmax': 1e0}
            # else:
                # vbounds['H'] = {'vmin': 1e-2, 'vmax': 1e0}
                # vbounds['S'] = {'vmin': 1e-2, 'vmax': 1e0}
                # vbounds['W'] = {'vmin': 1e-2, 'vmax': 1e0}

        im = ax.imshow(asymmetric_M.T, origin='lower', interpolation='nearest', cmap=cmap, norm=LogNorm(vmin=vbounds[setting_code]['vmin'], vmax=vbounds[setting_code]['vmax']))
    else:
        im = ax.imshow(asymmetric_M.T, origin='lower', interpolation='nearest', cmap=cmap)
    implot = im

    divider = make_axes_locatable(ax)
    cax = divider.new_horizontal(size="4%", pad=0.15)

    fig.add_axes(cax)
    cbar = fig.colorbar(implot, cax=cax)
    cbar.ax.tick_params(axis='y', labelsize=20)
    if density_or_frequency == 'frequency':
        cbar.ax.set_ylabel('Frequency of Contacts', fontsize=20)
    else:
        cbar.ax.set_ylabel('Density of Contacts', fontsize=20)
    ax.tick_params(labelsize=20)
    ax.set_xlabel('Age', fontsize=24)
    ax.set_ylabel('Age of Contacts', fontsize=24)
    ax.set_title(titles[setting_code] + ' Contact Patterns', fontsize=28)

    if aggregate_flag:
        tick_labels = [str(age_brackets[b][0]) + '-' + str(age_brackets[b][-1]) for b in age_brackets]
        ax.set_xticks(np.arange(len(tick_labels)))
        ax.set_xticklabels(tick_labels, fontsize=1)
        ax.set_xticklabels(tick_labels, fontsize=18, rotation=50)
        ax.set_yticks(np.arange(len(tick_labels)))
        ax.set_yticklabels(tick_labels, fontsize=18)

    return fig


if __name__ == '__main__':
    
    datadir = sp.datadir

    n = 100e3

    location = 'seattle_metro'
    state_location = 'Washington'
    country_location = 'usa'

    sheet_name = 'United States of America'

    setting_code = 'H'
    setting_code = 'S'
    setting_code = 'W'

    aggregate_flag = True
    # aggregate_flag = False

    logcolors_flag = True
    # logcolors_flag = False

    # density_or_frequency = 'density'
    density_or_frequency = 'frequency'

    do_save = True
    # do_save = False

    do_trimmed = True
    # do_trimmed = False

    trimmed_size_dic = {'S': 20, 'W': 10}

    if setting_code in ['S', 'W']:
        if do_trimmed:
            fig = plot_generated_trimmed_contact_matrix(datadir, n, location, state_location, country_location, setting_code, aggregate_flag, logcolors_flag, density_or_frequency, trimmed_size_dic)
        else:
            fig = plot_generated_contact_matrix(datadir, n, location, state_location, country_location, setting_code, aggregate_flag, logcolors_flag, density_or_frequency)

    elif setting_code == 'H':
        fig = plot_generated_contact_matrix(datadir, n, location, state_location, country_location, setting_code, aggregate_flag, logcolors_flag, density_or_frequency)

    if do_save:
        fig_path = datadir.replace('data', 'figures')
        os.makedirs(fig_path, exist_ok=True)

        if setting_code in ['S', 'W']:
            if do_trimmed:
                if aggregate_flag:
                    fig_path = os.path.join(fig_path, 'contact_matrices_152_countries', country_location, state_location, location + '_npop_' + str(n) + '_' + density_or_frequency + '_close_contact_matrix_setting_' + setting_code + '_aggregate_age_brackets.pdf')
                else:
                    fig_path = os.path.join(fig_path, 'contact_matrices_152_countries', country_location, state_location, location + '_npop_' + str(n) + '_' + density_or_frequency + '_close_contact_matrix_setting_' + setting_code + '.pdf')
            else:
                if aggregate_flag:
                    fig_path = os.path.join(fig_path, 'contact_matrices_152_countries', country_location, state_location, location + '_npop_' + str(n) + '_' + density_or_frequency + '_contact_matrix_setting_' + setting_code + '_aggregate_age_brackets.pdf')
                else:
                    fig_path = os.path.join(fig_path, 'contact_matrices_152_countries', country_location, state_location, location + '_npop_' + str(n) + '_' + density_or_frequency + '_contact_matrix_setting_' + setting_code + '.pdf')

        elif setting_code == 'H':
            if aggregate_flag:
                fig_path = os.path.join(fig_path, 'contact_matrices_152_countries', country_location, state_location, location + '_npop_' + str(n) + '_' + density_or_frequency + '_contact_matrix_setting_' + setting_code + '_aggregate_age_brackets.pdf')
            else:
                fig_path = os.path.join(fig_path, 'contact_matrices_152_countries', country_location, state_location, location + '_npop_' + str(n) + '_' + density_or_frequency + '_contact_matrix_setting_' + setting_code + '.pdf')
        fig.savefig(fig_path, format='pdf')
        fig.savefig(fig_path.replace('pdf', 'png'), format='png', dpi=300)
    plt.show()

# fig = plot_data_contact_matrix(datadir, location, state_location, country_location, sheet_name, setting_code, logcolors_flag)
# fig_path = datadir.replace('data', 'figures')
# plt.show()

# fig.savefig(os.path.join(fig_path, 'contact_matrices_152_countries', country_location, 'contact_matrix_' + country_location + '_' + setting_code + '.pdf'), format='pdf')
