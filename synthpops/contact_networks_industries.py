"""
This module uses workplace data by industry from `North American Industry Classification System`_ (NAICS) codes  to model workplaces as specific industries and the contact patterns for workers within each workplace.

.. _North American Industry Classification System: https://www.census.gov/eos/www/naics/
"""

import sciris as sc
import numpy as np
import networkx as nx
import pandas as pd
from collections import Counter
import os
from .base import *
from . import data_distributions as spdata
from . import sampling as spsamp
from . import contacts as spct
from . import contact_networks as spcnx
from .config import datadir

from copy import deepcopy
import matplotlib as mplt
import matplotlib.pyplot as plt
import cmocean


def get_establishments_by_industries_df(datadir, locations, state_location, country_location, level):
    """
    Filter a pandas DataFrame on establishment sizes by industry for the locations of interest at the county level.

    Args:
        datadir (string)            : The file path to the data directory.
        locations (list of string)  : A list with the names of the locations at the county level.
        state_location (string)     : The name of the state the location is in.
        country_location (string)   : The name of the country the location is in.
        level (string)              : The scale of region at which data is available.

    Returns:
        A pandas DataFrame with necessary columns to calculate establishment sizes by industry for the specified locations of interest.
    """
    file_path = os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, 'workplaces', 'workplaces_by_' + level + '_2015.csv')
    df = pd.read_csv(file_path)
    df = df[df['County'].isin(locations)]
    cols = ['County', 'NAICS Code', 'NAICS Industry', 'Enterprise Size', 'Establishments', 'Firms']
    df = df[cols]
    return df


def get_industry_type_df(datadir, country_location):
    """
    Get the 2017 NAICS US Codes and Titles.

    Args:
        datadir (string)          : The file path to the data directory.
        country_location (string) : The name of the country.

    Returns:
        A pandas DataFrame with 2017 NAICS US Code and Title.
    """
    file_path = os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', 'usa', '2-6 digit_2017_Codes.xlsx')
    df = pd.read_excel(file_path, skiprows=0)
    return df


def get_simplified_industry_type_df(datadir, country_location):
    """
    Get the simplified 2017 NAICS US Codes.

    Args:
        datadir (string)          : The file path to the data directory.
        country_location (string) : The name of the country.

    Returns:
        A pandas DataFrame with 2 digit 2017 NAICS US Codes mapping to main industry types.
    """
    file_path = os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', 'usa', '2-6 digit_2017_Codes_simplified.dat')
    return pd.read_csv(file_path, delimiter=';')


def get_industry_code(industry_type_df, industry_title):
    """
    Get the 2017 NAICS Code based on industry title.

    Args:
        industry_type_df (dataframe)    : The pandas DataFrame.
        industry_title (string)         : The 2017 NAICS US Title.

    Returns:
        The 2017 NAICS US Code as an integer.
    """
    return industry_type_df[industry_type_df['2017 NAICS US Title'] == industry_title]['2017 NAICS US   Code'].values[0].astype(int)


def get_main_industry_code(industry_type_df, industry_title):
    """
    Get the 2-digit 2017 NAICS US Code based on industry title.

    Args:
        industry_type_df (DataFrame)    : The pandas DataFrame.
        industry_title (string)         : The 2017 NAICS US Title.

    Returns:
        The 2-digit 2017 NAICS US Code as an integer.
    """
    code = str(get_industry_code(industry_type_df, industry_title))
    code = code[0:2]
    code = int(code)
    return code


def get_industry_title(industry_type_df, industry_code):
    """
    Get the 2017 NAICS US Title based on full industry code.

    Args:
        industry_type_df (DataFrame)    : The pandas DataFrame.
        industry_code (int)             : The 2017 NAICS US Code.

    Returns:
        The 2017 NAICS US Title.
    """
    return industry_type_df[industry_type_df['2017 NAICS US   Code'] == industry_code]['2017 NAICS US Title'].values[0]


def get_main_industry_title(industry_type_df, industry_code):
    """
    Get the main 2017 NAICS US Title based on 2-digit industry code.

    Args:
        industry_type_df (DataFrame)    : The pandas DataFrame.
        industry_code (int)             : The 2-digit 2017 NAICS US Code.

    Returns:
        The 2017 NAICS US Title.
    """
    industry_code = str(industry_code)
    industry_code = industry_code[0:2]
    industry_code = int(industry_code)
    industry_title = get_industry_code(industry_type_df, industry_code)
    return industry_title


def get_simplified_industry_title(simplified_industry_type_df, industry_code):
    """
    Get the simplified 2017 NAICS US Title from the 2-digit 2017 NAICS Code.

    Args:
        simplified_industry_type_df (DataFrame) : The pandas DataFrame
        industry_code (int)                     : The 2-digit 2017 NAICS US Code.

    Returns:
        The 2017 NAICS US Title for the 2-digit code.
    """
    return simplified_industry_type_df[simplified_industry_type_df['2017 NAICS US Code'] == industry_code]['2017 NAICS US Title'].values[0]


def get_simplified_industry_code(simplified_industry_type_df, industry_title):
    """
    Get the simplified 2017 NAICS US Code from the full title.

    Args:
        simplified_industry_type_df (DataFrame) : The pandas DataFrame.
        industry_title (string)                 : The full 2017 NAICS US Title.

    Returns:
        The 2-digit 2017 NAICS US Code.
    """
    return simplified_industry_type_df[simplified_industry_type_df['2017 NAICS US Title'] == industry_type]['2017 NAICS US Code'].values[0]


def get_establishment_size_brackets_df(datadir, locations, state_location='Washington', country_location='usa', level='county'):
    """
    Get size brackets DataFrame from Bureau of Labor Statistics (BLS) 2017 Data.

    Args:
        datadir (string)            : The file path to the data directory.
        locations (list)            : A list with the names of the locations at the county level.
        state_location (string)     : The name of the state the location is in.
        country_location (string)   : The name of the country the location is in.
        level (string)              : The scale of region at which data is available.

    Returns:
        A Dataframe of size brackets for establishments in the United States.
    """
    df = get_establishments_by_industries_df(datadir, locations, state_location, country_location, level)
    d = df[df['NAICS Industry'] == 'Total']

    size_labels = set(df['Enterprise Size'].values)
    index_to_size_labels_dic = {}
    index_to_size_brackets_dic = {}
    index_to_size_bracket_start_dic = {}
    index_to_size_bracket_end_dic = {}
    index_to_size_bracket_range_dic = {}
    size_label_to_bracket_dic = {}

    for ll in size_labels:
        l = ll.split(': ')
        index = int(l[0])

        if l[1] == 'Total':
            continue

        b = l[1].split(' ')[0].replace(',', '').split('-')
        if len(b) > 1:
            sb, eb = int(b[0]), int(b[1])
        else:
            sb, eb = int(b[0]), int(b[0])

        if sb == 0:
            sb = 1  # can't have workplaces of size 0...

        index_to_size_labels_dic[index] = ll
        index_to_size_brackets_dic[index] = [sb, eb]
        index_to_size_bracket_start_dic[index] = sb
        index_to_size_bracket_end_dic[index] = eb
        index_to_size_bracket_range_dic[index] = np.arange(sb, eb+1)
        size_label_to_bracket_dic[ll] = np.arange(sb, eb+1)

    new_df = pd.DataFrame(data={
                                'label': index_to_size_labels_dic,
                                # 'bracket': index_to_size_brackets_dic,
                                'bracket_start': index_to_size_bracket_start_dic,
                                'bracket_end': index_to_size_bracket_end_dic,
                                # 'bracket': index_to_size_bracket_range_dic
                                })
    new_df = new_df.sort_index()
    return new_df, size_label_to_bracket_dic


# def get_establishment_sizes_distr(datadir, locations, state_location='Washington', country_location='usa', level='county'):
# #     """
# #     Get size d
# #     """
#     size_label_df, size_label_to_bracket_dic = get_establishment_size_brackets_df(datadir, locations, state_location, country_location, level)
#     labels = sorted(size_label_to_bracket_dic.keys())
#     df = get_establishments_by_industries_df(datadir, locations, state_location, country_location, level)
#     d = df[df['NAICS Industry'] == 'Total']

#     size_distr_by_label_dic = {}
#     for label in labels:
#         size_distr_by_label_dic[label] = d[d['Enterprise Size'] == label]['Establishments'].values[0]
#     return norm_dic(size_distr_by_label_dic)


# def generate_workplace_sizes_and_industries(establishments_df, size_label_df, size_label_to_bracket_dic, size_distr_by_label_dic):
#     labels = sorted(size_label_to_bracket_dic.keys())

#     n = 900
#     distr_array = [size_distr_by_label_dic[label] for label in labels]
#     l = np.random.choice(labels, p=distr_array)
#     s = np.random.choice(size_label_to_bracket_dic[l])




# def generate_workplace_sizes_and_industries(datadir, locations, state_location='Washington', country_location='usa', level='county'):
#     establishments_df = get_establishments_by_industries_df(datadir, locations, state_location, country_location, level)
#     size_label_df, size_label_to_bracket_dic = get_establishment_size_brackets_df(datadir, locations, state_location, country_location, level)
#     labels = sorted(size_label_to_bracket_dic.keys())
#     size_distr_by_label_dic = get_establishment_sizes_distr(datadir, locations, state_location, country_location, level)

#     n = 900


#     distr_array = [size_distr_by_label_dic[label] for label in labels]
#     print(distr_array)
#     l = np.random.choice(labels, p=distr_array)
#     s = np.random.choice(size_label_to_bracket_dic[l])
#     print(l, size_distr_by_label_dic[l], size_label_to_bracket_dic[l][0], size_label_to_bracket_dic[l][-1], s)

# def get_industry_distr_by


def generate_synthetic_population_with_workplace_industries(n, datadir,location='seattle_metro',state_location='Washington',country_location='usa',sheet_name='United States of America',level='county',verbose=False,plot=False):
    """
    Modify the workplace network as generated by :py:meth:`~synthpops.generate_synthetic_population` to include  contact patterns according to each industry.

    Args:
        n (int)                                   : The number of people in the population.
        datadir (string)                          : The file path to the data directory.
        location (string)                         : The name of the location.
        state_location (string)                   : The name of the state the location is in.
        country_location (string)                 : The name of the country the location is in.
        sheet_name (string)                       : The name of the sheet in the Excel file with contact patterns.
        level (string)                            : The scale of region at which data is available.
        verbose (bool)                            : If True, print statements as contacts are being generated.
        plot (bool)                               : If True, plot and show a comparison of the generated workplace sizes vs. the expected sizes based on NAICS data.

    Returns:
        None

    """
    age_brackets = spdata.get_census_age_brackets(datadir,state_location,country_location)
    age_by_brackets_dic = get_age_by_brackets_dic(age_brackets)

    num_agebrackets = len(age_brackets)
    contact_matrix_dic = spdata.get_contact_matrix_dic(datadir,sheet_name)

    household_size_distr = spdata.get_household_size_distr(datadir,location,state_location,country_location)

    if n < 5000:
        raise NotImplementedError("Population is too small to currently be generated properly. Try a size larger than 5000.")

    # this could be unnecessary if we get the single year age distribution in a different way.
    n_to_sample_smoothly = int(1e6)
    # hh_sizes = sp.generate_household_sizes(n_to_sample_smoothly,household_size_distr)
    # hh_sizes = sp.generate_household_sizes(n_to_sample_smoothly,household_size_distr)
    # sp.trim_households(1000,household_size_distr)
    spct.make_popdict(n=5000)
