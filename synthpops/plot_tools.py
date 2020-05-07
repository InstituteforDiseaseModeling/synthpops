import sciris as sc
import numpy as np
import networkx as nx
from .base import *
from . import data_distributions as spdata
from . import sampling as spsamp
from . import contacts as spct
from . import contact_networks as spcn
from .config import datadir

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


def calculate_contact_matrix(contacts, density_or_frequency='density', setting_code='H'):
    """
    Calculate the symmetric contact matrix.

    Args:
        contacts (dict)               : dictionary of individuals with attributes, including their age and the ids of their contacts
        density_or_frequency (string) : If 'density', then each contact counts for 1/(group size -1) of a person's contact in a group, elif 'frequency' then count each contact. This means that more people in a group leads to higher rates of contact/exposure.
    
    Returns:
        Symmetric age specific contact matrix.
    """
    uids = contacts.keys()
    uids = [uid for uid in uids]

    num_ages = 101

    M = np.zeros((num_ages, num_ages))

    for n, uid in enumerate(uids):
        age = contacts[uid]['age']
        contact_ages = [contacts[c]['age'] for c in contacts[uid]['contacts'][setting_code]]
        contact_ages = np.array([int(a) for a in contact_ages])

        if len(contact_ages) > 0:
            if density_or_frequency == 'density':
                for ca in contact_ages:
                    M[age, ca] += 1.0/len(contact_ages)
            elif density_or_frequency == 'frequency':
                for ca in contact_ages:
                    M[age, ca] += 1.0
    return M


def plot_contact_frequency(freq_matrix_dic, age_count, aggregate_age_count, age_brackets, age_by_brackets_dic, setting_code='H', density_or_frequency='density', logcolors_flag=False, aggregate_flag=True):
    """
    Plots the age specific contact matrix where the matrix element matrix_ij is the contact rate or frequency
    for the average individual in age group i with all of their contacts in age group j. Can either be density
    or frequency definition, as well as a single year age contact matrix or a contact matrix for aggregated
    age brackets.

    Args:
        freq_matrix_dic (matrix)      : symmetric contact matrix, element ij is the contact for an average individual in age group i with all of their contacts in age group j
        age_count (dict)              : dictionary with the count of individuals in the population for each age
        aggregate_age_count (dict)    : dictionary with the count of individuals in the population in each age bracket
        age_brackets (dict)           : dictionary mapping age bracket keys to age bracket range
        age_by_brackets_dic (dict)    : dictionary mapping age to the age bracket range it falls in
        setting_code (string)         : name of the physial contact setting: H for households, S for schools, W for workplaces, C for community or other
        density_or_frequency (string) : If 'density', then each contact counts for 1/(group size -1) of a person's contact in a group, elif 'frequency' then count each contact. This means that more people in a group leads to higher rates of contact/exposure.
        logcolors_flag (bool)         : If True, plot heatmap in logscale
        aggregate_flag (book)         : If True, plot the contact matrix for aggregate age brackets, else single year age contact matrix.

    Returns:
        A fig object.
    """
    cmap = mplt.cm.get_cmap(cmocean.cm.deep_r)
    # cmap = mplt.cm.get_cmap(cmocean.cm.matter_r)

    fig = plt.figure(figsize=(9, 9))
    ax = []
    cax = []
    cbar = []
    leg = []
    implot = []

    titles = {'H': 'Household', 'S': 'School', 'W': 'Work'}

    if aggregate_flag:
        aggregate_M = get_aggregate_matrix(freq_matrix_dic, age_by_brackets_dic)
        asymmetric_M = get_asymmetric_matrix(aggregate_M, aggregate_age_count)
    else:
        asymmetric_M = get_asymmetric_matrix(freq_matrix_dic, age_count)

    for i in range(1):
        ax.append(fig.add_subplot(1, 1, i+1))

    if logcolors_flag:

        vbounds = {}
        if density_or_frequency == 'density':
            if aggregate_flag:
                vbounds['H'] = {'vmin': 1e-3, 'vmax': 1e-0}
                vbounds['S'] = {'vmin': 1e-3, 'vmax': 1e-0}
                vbounds['W'] = {'vmin': 1e-3, 'vmax': 1e-0}
            else:
                vbounds['H'] = {'vmin': 1e-3, 'vmax': 1e-1}
                vbounds['S'] = {'vmin': 1e-3, 'vmax': 1e-1}
                vbounds['W'] = {'vmin': 1e-3, 'vmax': 1e-1}

        elif density_or_frequency == 'frequency':
            if aggregate_flag:
                vbounds['H'] = {'vmin': 1e-2, 'vmax': 1e0}
                vbounds['S'] = {'vmin': 1e-2, 'vmax': 1e0}
                vbounds['W'] = {'vmin': 1e-2, 'vmax': 1e0}
            else:
                vbounds['H'] = {'vmin': 1e-2, 'vmax': 1e0}
                vbounds['S'] = {'vmin': 1e-2, 'vmax': 1e0}
                vbounds['W'] = {'vmin': 1e-2, 'vmax': 1e0}

        im = ax[0].imshow(asymmetric_M.T, origin='lower', interpolation='nearest', cmap=cmap, norm=LogNorm(vmin=vbounds[setting_code]['vmin'], vmax=vbounds[setting_code]['vmax']))

    else:

        im = ax[0].imshow(asymmetric_M.T, origin='lower', interpolation='nearest', cmap=cmap)

    implot.append(im)

    for i in range(1):
        divider = make_axes_locatable(ax[i])
        cax.append(divider.new_horizontal(size="4%", pad=0.15))

        fig.add_axes(cax[i])
        cbar.append(fig.colorbar(implot[i], cax=cax[i]))
        cbar[i].ax.tick_params(axis='y', labelsize=20)
        if density_or_frequency == 'frequency':
            cbar[i].ax.set_ylabel('Frequency of Contacts', fontsize=18)
        else:
            cbar[i].ax.set_ylabel('Density of Contacts', fontsize=18)
        ax[i].tick_params(labelsize=18)
        ax[i].set_xlabel('Age', fontsize=22)
        ax[i].set_ylabel('Age of Contacts', fontsize=22)
        ax[i].set_title(titles[setting_code] + ' Contact Patterns', fontsize=26)

        if aggregate_flag:
            tick_labels = [str(age_brackets[b][0]) + '-' + str(age_brackets[b][-1]) for b in age_brackets]
            ax[i].set_xticks(np.arange(len(tick_labels)))
            ax[i].set_xticklabels(tick_labels, fontsize=16)
            ax[i].set_xticklabels(tick_labels, fontsize=16, rotation=50)
            ax[i].set_yticks(np.arange(len(tick_labels)))
            ax[i].set_yticklabels(tick_labels, fontsize=16)

    return fig
