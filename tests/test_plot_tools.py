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
# 
import functools
import os
from collections import Counter
import pytest

do_save = False


def test_plot_generated_contact_matrix(setting_code='S',n=5000,aggregate_flag=True,logcolors_flag=True,density_or_frequency='density'):

    datadir = sp.datadir

    state_location = 'Washington'
    location = 'seattle_metro'
    country_location = 'usa'

    popdict = {}

    options_args = {'use_microstructure': True}
    network_distr_args = {'Npop': int(n)}
    contacts = sp.make_contacts(popdict,state_location = state_location,location = location, options_args = options_args, network_distr_args = network_distr_args)

    age_brackets = sp.get_census_age_brackets(datadir,state_location=state_location,country_location=country_location)
    age_by_brackets_dic = sp.get_age_by_brackets_dic(age_brackets)

    ages = []
    for uid in contacts:
        ages.append(contacts[uid]['age'])

    num_agebrackets = len(age_brackets)

    age_count = Counter(ages)
    aggregate_age_count = sp.get_aggregate_ages(age_count,age_by_brackets_dic)

    freq_matrix_dic = sp.calculate_contact_matrix(contacts,density_or_frequency)

    fig = sp.plot_contact_frequency(freq_matrix_dic,setting_code,age_count,aggregate_age_count,age_brackets,age_by_brackets_dic,density_or_frequency,logcolors_flag,aggregate_flag)

    return fig


def test_plot_generated_trimmed_contact_matrix(setting_code,n=5000,aggregate_flag=True,logcolors_flag=True,density_or_frequency='density'):

    datadir = sp.datadir

    state_location = 'Washington'
    location = 'seattle_metro'
    country_location = 'usa'

    popdict = {}

    options_args = {'use_microstructure': True}
    network_distr_args = {'Npop': int(n)}
    contacts = sp.make_contacts(popdict,state_location = state_location,location = location, options_args = options_args, network_distr_args = network_distr_args)
    contacts = sp.trim_contacts(contacts,trimmed_size_dic=None,use_clusters=False)

    age_brackets = sp.get_census_age_brackets(datadir,state_location=state_location,country_location=country_location)
    age_by_brackets_dic = sp.get_age_by_brackets_dic(age_brackets)

    ages = []
    for uid in contacts:
        ages.append(contacts[uid]['age'])

    num_agebrackets = len(age_brackets)

    age_count = Counter(ages)
    aggregate_age_count = sp.get_aggregate_ages(age_count,age_by_brackets_dic)

    freq_matrix_dic = sp.calculate_contact_matrix(contacts,density_or_frequency)

    fig = sp.plot_contact_frequency(freq_matrix_dic,setting_code,age_count,aggregate_age_count,age_brackets,age_by_brackets_dic,density_or_frequency,logcolors_flag,aggregate_flag)

    return fig



if __name__ == '__main__':

    datadir = sp.datadir

    
    n = int(20000)

    state_location = 'Washington'
    location = 'seattle_metro'
    country_location = 'usa'

    setting_code = 'H'
    # setting_code = 'S'
    # setting_code = 'W'
    
    aggregate_flag = True
    aggregate_flag = False
    logcolors_flag = True

    density_or_frequency = 'density'
    # density_or_frequency = 'frequency'

    # fig = test_plot_generated_contact_matrix(setting_code,n,aggregate_flag,logcolors_flag,density_or_frequency)
    fig = test_plot_generated_trimmed_contact_matrix(setting_code,n,aggregate_flag,logcolors_flag,density_or_frequency)

    if do_save:
        fig.savefig('n_' + str(n) + '_people_' + density_or_frequency + '_close_contact_matrix_setting_' + setting_code + '.pdf',format = 'pdf')
    plt.show()
