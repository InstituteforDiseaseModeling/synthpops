"""
This module plots the age-specific contact matrix in different settings.
"""

import os
from copy import deepcopy

import sciris as sc
import numpy as np
import networkx as nx

import matplotlib as mplt
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmocean
import cmasher as cmr
# from matplotlib.ticker import LogLocator, LogFormatter
# import matplotlib.font_manager as font_manager

from . import base as spb
from . import data_distributions as spdata
from . import sampling as spsamp
from . import contacts as spct
from . import contact_networks as spcn
from .config import datadir


# Pretty fonts
try:
    fontstyle = 'Roboto_Condensed'
    mplt.rcParams['font.family'] = fontstyle.replace('_', ' ')
except:
    mplt.rcParams['font.family'] = 'Roboto'
mplt.rcParams['font.size'] = 16


def calculate_contact_matrix(population, density_or_frequency='density', setting_code='H'):
    """
    Calculate the symmetric age-specific contact matrix from the connections
    for all people in the population. density_or_frequency sets the type of
    contact matrix calculated.

    When density_or_frequency is set to 'frequency' each person is assumed to
    have a fixed amount of contact with others they are connected to in a
    setting so each person will split their contact amount equally among their
    connections. This means that if a person has links to 4 other individuals
    then 1/4 will be added to the matrix element matrix[age_i][age_j] for each
    link, where age_i is the age of the individual and age_j is the age of the
    contact. This follows the mass action principle such that increased density
    or number of people a person is in contact with leads to decreased per-link
    or connection contact rate.

    When density_or_frequency is set to 'density' the amount of contact each
    person has with others scales with the number of people they are connected
    to. This means that a person with connections to 4 individuals has a higher
    total contact rate than a person with connection to 3 individuals. For this
    definition if a person has links to 4 other individuals then 1 will be
    added to the matrix element matrix[age_i][age_j] for each contact. This
    follows the 'pseudo'mass action principle such that the per-link or
    connection contact rate is constant.

    Args:
        population (dict)          : A dictionary of a population with attributes.
        density_or_frequency (str) : option for the type of contact matrix calculated.
        setting_code (str)         : name of the physial contact setting: H for households, S for schools, W for workplaces, C for community or other, and 'lTCF' for long term care facilities

    Returns:
        Symmetric age specific contact matrix.

    """
    uids = population.keys()
    uids = [uid for uid in uids]

    num_ages = 101

    M = np.zeros((num_ages, num_ages))

    for n, uid in enumerate(uids):
        age = population[uid]['age']
        contact_ages = [population[c]['age'] for c in population[uid]['contacts'][setting_code]]
        contact_ages = np.array([int(a) for a in contact_ages])

        if len(contact_ages) > 0:
            if density_or_frequency == 'frequency':
                for ca in contact_ages:
                    M[age, ca] += 1.0 / len(contact_ages)
            elif density_or_frequency == 'density':
                for ca in contact_ages:
                    M[age, ca] += 1.0
    return M


def plot_contact_matrix(matrix, age_count, aggregate_age_count, age_brackets, age_by_brackets_dic, setting_code='H', density_or_frequency='density', logcolors_flag=False, aggregate_flag=True, cmap='cmr.freeze_r', fontsize=16, rotation=50, title_prefix=None):
    """
    Plots the age specific contact matrix where the matrix element matrix_ij is the contact rate or frequency
    for the average individual in age group i with all of their contacts in age group j. Can either be density
    or frequency definition, as well as a single year age contact matrix or a contact matrix for aggregated
    age brackets.

    Args:
        matrix (np.array)                : symmetric contact matrix, element ij is the contact for an average individual in age group i with all of their contacts in age group j
        age_count (dict)                 : dictionary with the count of individuals in the population for each age
        aggregate_age_count (dict)       : dictionary with the count of individuals in the population in each age bracket
        age_brackets (dict)              : dictionary mapping age bracket keys to age bracket range
        age_by_brackets_dic (dict)       : dictionary mapping age to the age bracket range it falls in
        setting_code (str)               : name of the physial contact setting: H for households, S for schools, W for workplaces, C for community or other
        density_or_frequency (str)       : If 'density', then each contact counts for 1/(group size -1) of a person's contact in a group, elif 'frequency' then count each contact. This means that more people in a group leads to higher rates of contact/exposure.
        logcolors_flag (bool)            : If True, plot heatmap in logscale
        aggregate_flag (book)            : If True, plot the contact matrix for aggregate age brackets, else single year age contact matrix.
        cmap(str or matplotlib colormap) : colormap
        fontsize (int)                   : base font size
        rotation (int)                   : rotation for x axis labels
        title_prefix(str)                : optional title prefix for the figure

    Returns:
        A fig object.

    Note:
        For the long term care facilities you may want the age count and the aggregate age count to only consider those who live or work in long term care facilities because otherwise this will be the whole population wide average mixing in that setting

    """
    cmap = mplt.cm.get_cmap(cmap)

    fig = plt.figure(figsize=(10, 10), tight_layout=True)
    ax = []
    cax = []
    cbar = []
    leg = []
    implot = []

    titles = {'H': 'Household', 'S': 'School', 'W': 'Work', 'LTCF': 'Long Term Care Facilities'}

    if aggregate_flag:
        aggregate_M = spb.get_aggregate_matrix(matrix, age_by_brackets_dic)
        asymmetric_M = spb.get_asymmetric_matrix(aggregate_M, aggregate_age_count)
    else:
        asymmetric_M = spb.get_asymmetric_matrix(matrix, age_count)

    for i in range(1):
        ax.append(fig.add_subplot(1, 1, i + 1))

    if logcolors_flag:

        vbounds = {}
        if density_or_frequency == 'frequency':
            if aggregate_flag:
                vbounds['H'] = {'vmin': 1e-2, 'vmax': 1e-0}
                vbounds['S'] = {'vmin': 1e-3, 'vmax': 1e-0}
                vbounds['W'] = {'vmin': 1e-3, 'vmax': 1e-0}
                vbounds['LTCF'] = {'vmin': 1e-3, 'vmax': 1e-1}
            else:
                vbounds['H'] = {'vmin': 1e-3, 'vmax': 1e-1}
                vbounds['S'] = {'vmin': 1e-3, 'vmax': 1e-1}
                vbounds['W'] = {'vmin': 1e-3, 'vmax': 1e-1}
                vbounds['LTCF'] = {'vmin': 1e-3, 'vmax': 1e-0}

        elif density_or_frequency == 'density':
            if aggregate_flag:
                vbounds['H'] = {'vmin': 1e-2, 'vmax': 1e0}
                vbounds['S'] = {'vmin': 1e-2, 'vmax': 1e1}
                vbounds['W'] = {'vmin': 1e-2, 'vmax': 1e1}
                vbounds['LTCF'] = {'vmin': 1e-3, 'vmax': 1e-0}

            else:
                vbounds['H'] = {'vmin': 1e-2, 'vmax': 1e0}
                vbounds['S'] = {'vmin': 1e-2, 'vmax': 1e0}
                vbounds['W'] = {'vmin': 1e-2, 'vmax': 1e0}
                vbounds['LTCF'] = {'vmin': 1e-2, 'vmax': 1e-0}

        im = ax[0].imshow(asymmetric_M.T, origin='lower', interpolation='nearest', cmap=cmap, norm=LogNorm(vmin=vbounds[setting_code]['vmin'], vmax=vbounds[setting_code]['vmax']))

    else:

        im = ax[0].imshow(asymmetric_M.T, origin='lower', interpolation='nearest', cmap=cmap)

    implot.append(im)

    if fontsize > 20:
        rotation = 90

    for i in range(len(ax)):
        divider = make_axes_locatable(ax[i])
        cax.append(divider.new_horizontal(size="4%", pad=0.15))

        fig.add_axes(cax[i])
        cbar.append(fig.colorbar(implot[i], cax=cax[i]))
        cbar[i].ax.tick_params(axis='y', labelsize=fontsize + 4)
        if density_or_frequency == 'frequency':
            cbar[i].ax.set_ylabel('Frequency of Contacts', fontsize=fontsize + 2)
        else:
            cbar[i].ax.set_ylabel('Density of Contacts', fontsize=fontsize + 2)
        ax[i].tick_params(labelsize=fontsize + 2)
        ax[i].set_xlabel('Age', fontsize=fontsize + 6)
        ax[i].set_ylabel('Age of Contacts', fontsize=fontsize + 6)
        # ax[i].set_title(titles[setting_code] + ' Contact Patterns', fontsize=fontsize + 10)
        ax[i].set_title(
            (title_prefix if title_prefix is not None else '') + titles[setting_code] + ' Age Mixing', fontsize=fontsize + 10)

        if aggregate_flag:
            tick_labels = [str(age_brackets[b][0]) + '-' + str(age_brackets[b][-1]) for b in age_brackets]
            ax[i].set_xticks(np.arange(len(tick_labels)))
            ax[i].set_xticklabels(tick_labels, fontsize=fontsize)
            ax[i].set_xticklabels(tick_labels, fontsize=fontsize, rotation=rotation)
            ax[i].set_yticks(np.arange(len(tick_labels)))
            ax[i].set_yticklabels(tick_labels, fontsize=fontsize)
        else:
            ax[i].set_xticks(np.arange(0, len(age_count) + 1, 10))
            ax[i].set_yticks(np.arange(0, len(age_count) + 1, 10))

    return fig


