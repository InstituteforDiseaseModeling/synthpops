"""
Read in data distributions.
"""

import os
import json
import numpy as np
import pandas as pd
import sciris as sc
from collections import Counter
from . import base as spb
from . import config as cfg


def get_relative_path(datadir):
    base_dir = datadir
    if len(cfg.rel_path) > 1:
        base_dir = os.path.join(datadir, *cfg.rel_path)
    return base_dir


def get_nbrackets():
    return cfg.nbrackets


def get_age_brackets_from_df(ab_file_path):
    """
    Create a dict of age bracket ranges from ab_file_path.

    Args:
        ab_file_path (string): file path to get the ends of different age brackets from

    Returns:
        A dictionary with a np.ndarray of the age range that maps to each age bracket key.

    **Examples**::

        get_age_brackets_from_df(ab_file_path) returns a dictionary age_brackets, where age_brackets[0] is the age range for the first age bracket, age_brackets[1] is the age range for the second age bracket, etc.

    """
    age_brackets = {}
    check_exists = False if ab_file_path is None else os.path.exists(ab_file_path)

    # check if ab_file_path exists, if not raise error
    if check_exists is False:
        raise ValueError(f"The file path {ab_file_path} does not exist. Please check that this file exists.")

    ab_df = pd.read_csv(ab_file_path, header=None)
    for index, row in enumerate(ab_df.iterrows()):
        age_min = row[1].values[0]
        age_max = row[1].values[1]
        age_brackets[index] = np.arange(age_min, age_max+1)
    return age_brackets


def get_age_bracket_distr_path(datadir, location=None, state_location=None, country_location=None, nbrackets=None):
    """
    Get file_path for age distribution by age brackets.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in

    Returns:
        A file path to the age distribution by age bracket data.

    """
    datadir = get_relative_path(datadir)
    levels = [location, state_location, country_location]
    if all(level is None for level in levels):
        raise NotImplementedError("Missing inputs. Please check that you have supplied the correct location and state_location strings.")
    elif country_location is None:
        raise NotImplementedError("Missing country_location string. Please check that you have supplied this string.")
    elif state_location is None:
        return os.path.join(datadir,  country_location, 'age_distributions', f'{country_location}_age_bracket_distr_{cfg.nbrackets}.dat')
    elif location is None:
        return os.path.join(datadir,  country_location, state_location, 'age_distributions', f'{state_location}_age_bracket_distr_{cfg.nbrackets}.dat')
    else:
        return os.path.join(datadir,  country_location, state_location, location, 'age_distributions', f'{location}_age_bracket_distr_{cfg.nbrackets}.dat')

    # paths = cfg.FilePaths(location, state_location, country_location)
    # if nbrackets is None:
    #     nbrackets = cfg.nbrackets
    # base = f'age_bracket_distr_{nbrackets}'
    # prefix = '{location}_' + base
    # alt_prefix = None

    # file = paths.get_demographic_file(location=location, filedata_type='age_distributions', prefix=prefix, suffix='.dat', filter_list=None, alt_prefix=alt_prefix)
    # return file


def read_age_bracket_distr(datadir, location=None, state_location=None, country_location=None, nbrackets=None, file_path=None, use_default=False):
    """
    A dict of age distribution by age brackets. If use_default, then we'll first try to look for location specific data and if that's not available we'll use default data from Seattle, WA. This
    may not be appropriate for the population under study so it's best to provide as much data as you can for the specific population.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in
        file_path (string)        : file path to user specified gender by age bracket distribution data
        use_default (bool)        : if True, try to first use the other parameters to find data specific to the location under study, otherwise returns default data drawing from Seattle, Washington.

    Returns:
        A dictionary of the age distribution by age bracket. Keys map to a range of ages in that age bracket.

    """
    if file_path is None:
        file_path = get_age_bracket_distr_path(datadir, location, state_location, country_location, nbrackets)

    try:
        df = pd.read_csv(file_path)
    except:
        if use_default:
            file_path = get_age_bracket_distr_path(datadir, location=cfg.default_location, state_location=cfg.default_state, country_location=cfg.default_country, nbrackets=cfg.nbrackets)
            df = pd.read_csv(file_path)
        else:
            raise NotImplementedError(f"Data unavailable for the location specified. Please check input strings or set use_default to True to use default values from {cfg.default_location}, {cfg.default_state}.")
    return dict(zip(np.arange(len(df)), df.percent))


def get_household_size_distr_path(datadir, location=None, state_location=None, country_location=None):
    """
    Get file_path for household size distribution.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in

    Returns:
        A file path to the household size distribution data.

    """
    datadir = get_relative_path(datadir)
    levels = [location, state_location, country_location]
    if all(level is None for level in levels):
        raise NotImplementedError("Missing inputs. Please check that you have supplied the correct location and state_location strings.")
    elif country_location is None:
        raise NotImplementedError("Mssing country_location string. Please check that you have supplied this string.")
    elif state_location is None:
        return os.path.join(datadir,  country_location, 'household_size_distributions', f'{country_location}_household_size_distr.dat')
    elif location is None:
        return os.path.join(datadir,  country_location, state_location, 'household_size_distributions', f'{state_location}_household_size_distr.dat')
    else:
        return os.path.join(datadir,  country_location, state_location, location, 'household_size_distributions', f'{location}_household_size_distr.dat')

    # paths = cfg.FilePaths(location, state_location, country_location)
    # base = 'household_size_distr'
    # prefix = f'{location}_' + base
    # alt_prefix = None

    # file = paths.get_demographic_file(location=location, filedata_type='household_size_distributions', prefix=prefix, suffix='.dat', alt_prefix=alt_prefix)
    # return file


def get_household_size_distr(datadir, location=None, state_location=None, country_location=None, file_path=None, use_default=False):
    """
    A dictionary of the distributions of household sizes. If you don't give the file_path, then supply the location and state_location strings.
    If use_default, then we'll first try to look for location specific data and if that's not available we'll use default data from Seattle, WA. This
    may not be appropriate for the population under study so it's best to provide as much data as you can for the specific population.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in
        file_path (string)        : file path to user specified gender by age bracket distribution data
        use_default (bool)        : if True, try to first use the other parameters to find data specific to the location under study, otherwise returns default data drawing from Seattle, Washington.

    Returns:
        A dictionary of the household size distribution data. Keys map to the household size as an integer, values are the percent of households of that size.

    """
    if file_path is None:
        file_path = get_household_size_distr_path(datadir, location, state_location, country_location)
    try:
        # file_path = get_household_size_distr_path(datadir, location, state_location, country_location)
        df = pd.read_csv(file_path)
    except:
        if use_default:
            file_path = get_household_size_distr_path(datadir, location=cfg.default_location, state_location=cfg.default_state, country_location=cfg.default_country)
            # print("file_path: " + str(file_path))
            df = pd.read_csv(file_path)
        else:
            raise NotImplementedError(f"Data unavailable for the location specified. Please check input strings or set use_default to True to use default values from {cfg.default_location}, {cfg.default_state}.")
    return dict(zip(df.household_size, df.percent))


def get_head_age_brackets_path(datadir, location=None, state_location=None, country_location=None):
    """
    Get file_path for head of household age brackets. If data doesn't exist at the state level, only give the country_location.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state
        country_location (string) : name of the country the state_location is in

    Returns:
        A file path to the age brackets for head of household distribution data.

    """
    levels = [location, state_location, country_location]
    if all(level is None for level in levels):
        raise NotImplementedError("Missing input strings. Try again.")
    elif country_location is None:
        raise NotImplementedError("Missing country_location string. Please check that you have supplied this string.")
    elif state_location is None:
        return os.path.join(datadir, country_location, 'household_living_arrangements', f'{country_location}_head_age_brackets.dat')
    elif location is None:
        return os.path.join(datadir, country_location, state_location, 'household_living_arrangements', f'{state_location}_head_age_brackets.dat')
    else:
        return os.path.join(datadir, country_location, state_location, location, 'household_living_arrangements', f'{location}_head_age_brackets.dat')

    # paths = cfg.FilePaths(location, state_location, country_location)
    # base = 'head_age_brackets'
    # prefix = f'{location}_' + base
    # alt_prefix = None
    # filedata_type = 'household_living_arrangements'

    # if cfg.alt_location is not None:
    #     alt_prefix = prefix
    # """
    #     alt_location = cfg.alt_location.location
    #     alt_state_location = cfg.alt_location.state_location
    #     alt_country_location = cfg.alt_location.country_location
    #     alt_prefix ="{location}_" + base
    #     if alt_country_location == 'usa':
    #         alt_prefix = base
    # """

    # file = paths.get_demographic_file(location=location, filedata_type=filedata_type, prefix=prefix,
    #                                   suffix='.dat', alt_prefix=alt_prefix)
    # return file


def get_head_age_brackets(datadir, location=None, state_location=None, country_location=None, file_path=None, use_default=False):
    """
    Get a dictionary of head age brackets either from the file_path directly, or using the other parameters to figure out what the file_path should be.
    If use_default, then we'll first try to look for location specific data and if that's not available we'll use default data from Seattle, WA. This
    may not be appropriate for the population under study so it's best to provide as much data as you can for the specific population.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state
        country_location (string) : name of the country the state_location is in
        file_path (string)        : file path to user specified gender by age bracket distribution data
        use_default (bool)        : if True, try to first use the other parameters to find data specific to the location under study, otherwise returns default data drawing from Seattle, Washington.

    Returns:
        A dictionary of the age brackets for head of household distribution data. Keys map to the age bracket as an integer, values are the percent of households which head of household in that age bracket.

    """
    if file_path is None:
        file_path = get_head_age_brackets_path(datadir, location=location, state_location=state_location, country_location=country_location)
    try:
        age_brackets = get_age_brackets_from_df(file_path)
    except:
        if use_default:
            file_path = get_head_age_brackets_path(datadir, location=location, state_location=cfg.default_state,
                                                   country_location=cfg.default_country)
            age_brackets = get_age_brackets_from_df(file_path)
        else:
            raise NotImplementedError(f"Data unavailable for the location specified. Please check input strings or set use_default to True to use default values from {cfg.default_location}, {cfg.default_state}.")
    return age_brackets


def get_household_head_age_by_size_path(datadir, location=None, state_location=None, country_location=None):
    """
    Get file_path for head of household age by size counts or distribution. If the data doesn't exist at the state level, only give the country_location.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state
        country_location (string) : name of the country the state_location is in

    Returns:
        A file path to the head of household age by household size count or distribution data.
    """
    levels = [location, state_location, country_location]
    if all(level is None for level in levels):
        raise NotImplementedError("Missing input strings. Try again.")
    elif country_location is None:
        raise NotImplementedError("Missing country_location string. Please check that you have supplied this string.")
    elif state_location is None:
        return os.path.join(datadir, country_location, 'household_living_arrangements', f'{country_location}_household_head_age_and_size_count.dat')
    elif location is None:
        return os.path.join(datadir, country_location, state_location, 'household_living_arrangements', f'{state_location}_household_head_age_and_size_count.dat')
    else:
        return os.path.join(datadir, country_location, state_location, location, 'household_living_arrangements', f'{location}_household_head_age_and_size_count.dat')

    # paths = cfg.FilePaths(None, state_location, country_location)
    # base = 'household_head_age_and_size_count'
    # prefix = '{location}_' + base
    # alt_prefix = None

    # filedata_type = 'household_living_arrangements'

    # if cfg.alt_location is not None:
    #     alt_prefix = prefix
    # """
    #     alt_location = cfg.alt_location.location
    #     alt_state_location = cfg.alt_location.state_location
    #     alt_country_location = cfg.alt_location.country_location
    #     alt_prefix ="{location}_" + base

    #     if alt_country_location == 'usa':      #-----remove
    #         alt_prefix = base
    # """

    # file = paths.get_demographic_file(location=None, filedata_type=filedata_type, prefix=prefix,
    #                                   suffix='.dat', alt_prefix=alt_prefix)
    # return file


def get_household_head_age_by_size_df(datadir, location=None, state_location=None, country_location=None, file_path=None, use_default=False):
    """
    Return a pandas df of head of household age by the size of the household. If the file_path is given return from there first.
    If use_default, then we'll first try to look for location specific data and if that's not available we'll use default data from Seattle, WA. This
    may not be appropriate for the population under study so it's best to provide as much data as you can for the specific population.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state
        country_location (string) : name of the country the state_location is in
        file_path (string)        : file path to user specified gender by age bracket distribution data
        use_default (bool)        : if True, try to first use the other parameters to find data specific to the location under study, otherwise returns default data drawing from Seattle, Washington.

    Returns:
        A file path to the head of household age by household size count or distribution data.
    """
    if file_path is None:
        file_path = get_household_head_age_by_size_path(datadir, location=location, state_location=state_location, country_location=country_location)
    try:
        df = pd.read_csv(file_path)
    except:
        if use_default:
            file_path = get_household_head_age_by_size_path(datadir, location=cfg.default_location, state_location=cfg.default_state, country_location=cfg.default_country)
            df = pd.read_csv(file_path)
        else:
            raise NotImplementedError(f"Data unavailable for the location specified. Please check input strings or set use_default to True to use default values from {cfg.default_location}, {cfg.default_state}.")
    return df


def get_head_age_by_size_distr(datadir, location=None, state_location=None, country_location=None, file_path=None, household_size_1_included=False, use_default=False):
    """
    Create an array of head of household age bracket counts (col) given by size (row). If use_default, then we'll first try to look for location
    specific data and if that's not available we'll use default data from Seattle, WA. This may not be appropriate for the population under study
    so it's best to provide as much data as you can for the specific population.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state
        country_location (string) : name of the country the state_location is in
        file_path (string)        : file path to user specified gender by age bracket distribution data
        household_size_1_included : if True, age distribution for who lives alone is included in the head of household age by household size dataframe, so it will be used. Else, assume a uniform distribution for this among all ages of adults.
        use_default (bool)        : if True, try to first use the other parameters to find data specific to the location under study, otherwise returns default data drawing from Seattle, Washington.

    Returns:
        An array where each row s represents the age distribution of the head of households for households of size s-1.

    """
    hha_df = get_household_head_age_by_size_df(datadir, location=location, state_location=state_location, country_location=country_location, file_path=file_path, use_default=use_default)

    hha_by_size = np.zeros((2 + len(hha_df), len(hha_df.columns)-1))

    if household_size_1_included:
        for s in range(1, len(hha_df)+1):
            d = hha_df[hha_df['family_size'] == s].values[0][1:]
            hha_by_size[s-1] = d
    else:
        hha_by_size[0, :] += 1
        for s in range(2, len(hha_df)+2):
            d = hha_df[hha_df['family_size'] == s].values[0][1:]
            hha_by_size[s-1] = d
    return hha_by_size


def get_census_age_brackets_path(datadir, location=None, state_location=None, country_location=None, nbrackets=None):
    """
    Get file_path for census age brackets: depends on the state or country of the source data on contact patterns.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state
        country_location (string) : name of the country the state_location is in

    Returns:
        A file path to the age brackets to be used with census age data in combination with the contact matrix data.
    """
    datadir = get_relative_path(datadir)
    if nbrackets is None:
        nbrackets = cfg.nbrackets
    levels = [location, state_location, country_location]

    if all(level is None for level in levels):
        raise NotImplementedError("Missing input strings. Try again.")
    elif country_location is None:
        raise NotImplementedError("Missing country_location string. Please check that you have supplied this string.")
    elif state_location is None:
        return os.path.join(datadir, country_location, 'age_distributions', f'{country_location}_census_age_brackets_{nbrackets}.dat')
    elif location is None:
        return os.path.join(datadir, country_location, state_location, 'age_distributions', f'{state_location}_census_age_brackets_{nbrackets}.dat')
    else:
        return os.path.join(datadir, country_location, state_location, location, 'age_distributions', f'{location}_census_age_brackets_{nbrackets}.dat')

    # if nbrackets is None:
    #     nbrackets = cfg.nbrackets
    # base = f"census_age_brackets_{nbrackets}"
    # prefix = "{location}_" + base
    # alt_prefix = None
    # print("whatcensus", prefix)

    # if cfg.alt_location is not None:
    #     alt_prefix = prefix
    # paths = cfg.FilePaths(None, state_location, country_location)
    # file = paths.get_data_file(location=location, filedata_type='age_distributions', prefix=prefix, suffix='.dat', alt_prefix=alt_prefix)

    # file_path = file
    # # return file, file_path
    # return file


def get_census_age_brackets(datadir, location=None, state_location=None, country_location=None, file_path=None, use_default=False, nbrackets=None):
    """
    Get census age brackets: depends on the country or source of contact pattern data. If use_default, then we'll
    first try to look for location specific data and if that's not available we'll use default data from Seattle, WA.
    This may not be appropriate for the population under study so it's best to provide as much data as you can for the
    specific population.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state
        country_location (string) : name of the country the state_location is in
        file_path (string)        : file path to user specified gender by age bracket distribution data
        use_default (bool)        : if True, try to first use the other parameters to find data specific to the location under study, otherwise returns default data drawing from Seattle, Washington.

    Returns:
        A dictionary of the range of ages that map to each age bracket.

    """
    if nbrackets is None:
        nbrackets = cfg.nbrackets

    if file_path is None:
        file_path = get_census_age_brackets_path(datadir, location, state_location, country_location, nbrackets=nbrackets)

    try:
        age_brackets = get_age_brackets_from_df(file_path)
    except:
        if use_default:
            file_path = get_census_age_brackets_path(datadir, location=cfg.default_location, state_location=cfg.default_state, country_location=cfg.default_country, nbrackets=nbrackets)
            age_brackets = get_age_brackets_from_df(file_path)
        else:
            raise NotImplementedError(f"Data unavailable for the location specified. Please check input strings or set use_default to True to use default values from {cfg.default_location}, {cfg.default_state}.")
    return age_brackets


def get_contact_matrix(datadir, setting_code, sheet_name=None, file_path=None, delimiter=' ', header=None):
    """
    Get setting specific contact matrix given sheet name to use. If file_path is given, then delimiter and header should also be specified.

    Args:
        datadir (string)          : file path to the data directory
        setting_code (string)     : name of the physial contact setting: H for households, S for schools, W for workplaces, C for community or other
        sheet_name (string)       : name of the sheet in the excel file with contact patterns
        file_path (string)        : file path to user specified gender by age bracket distribution data
        delimiter (string)        : delimter for the contact matrix file
        header (int)              : row number for the header of the file

    Returns:
        Matrix of contact patterns where each row i is the average contact patterns for an individual in age bracket i and the columns represent the age brackets of their contacts. The matrix element i,j is then the contact rate, number, or frequency for the average individual in age bracket i with all of their contacts in age bracket j in that physical contact setting.
    """
    if file_path is None:
        setting_names = {'H': 'home', 'S': 'school', 'W': 'work', 'C': 'other_locations'}
        base_dir = get_relative_path(datadir)

        if setting_code in setting_names:
            file_path = os.path.join(base_dir, 'MUestimates_' + setting_names[setting_code] + '_1.xlsx')
            try: # Shortcut: use pre-processed data
                obj_path = file_path.replace('_1.xlsx', '.obj').replace('_2.xlsx', '.obj')
                data = sc.loadobj(obj_path)
                arr = data[sheet_name]
                return arr
            except Exception as E:
                errormsg = f'Warning: could not load pickled data ({str(E)}), defaulting to Excel...'
                print(errormsg)
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet_name, header=0)
                except:
                    file_path = file_path.replace('_1.xlsx', '_2.xlsx')
                    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
                return np.array(df)
        else:
            raise NotImplementedError("Invalid setting code. Try again.")
    else:
        try:
            df = pd.read_csv(file_path, delimiter=delimiter, header=header)
            return np.array(df)
        except:
            raise NotImplementedError("Contact matrix did not open. Check inputs.")


def get_contact_matrix_dic(datadir, sheet_name=None, file_path_dic=None, delimiter=' ', header=None, use_default=False):
    # need review for additional countries
    """
    Create a dict of setting specific age mixing matrices. If use_default, then we'll first try to look for location specific
    data and if that's not available we'll use default data from Seattle, WA. This may not be appropriate for the population
    under study so it's best to provide as much data as you can for the specific population.

    Args:
        datadir (string)          : file path to the data directory
        setting_code (string)     : name of the physial contact setting: H for households, S for schools, W for workplaces, C for community or other
        sheet_name (string)       : name of the sheet in the excel file with contact patterns
        file_path (string)        : file path to user specified gender by age bracket distribution data
        delimiter (string)        : delimter for the contact matrix file
        header (int)              : row number for the header of the file

    Returns:
        A dictionary of the different contact matrices for each population, given by the sheet name. Keys map to the different possible physical contact settings for which data are available.

    """
    matrix_dic = {}
    if file_path_dic is None:
        file_path_dic = dict.fromkeys(['H', 'S', 'W', 'C'], None)
    try:
        for setting_code in ['H', 'S', 'W', 'C']:
            matrix_dic[setting_code] = get_contact_matrix(datadir, setting_code, sheet_name, file_path_dic[setting_code], delimiter, header)
    except:
        if use_default:
            for setting_code in ['H', 'S', 'W', 'C']:
                matrix_dic[setting_code] = get_contact_matrix(datadir, setting_code, sheet_name='United States of America')
        else:
            raise NotImplementedError("Data unavailable for the location specified. Please check input strings or set use_default to True to use default values from the United States of America.")
    return matrix_dic


def get_school_enrollment_rates_path(datadir, location=None, state_location=None, country_location=None):
    """
    Get a file_path for enrollment rates by age.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in

    Returns:
        A file path to the school enrollment rates.
    """
    datadir = get_relative_path(datadir)
    levels = [location, state_location, country_location]
    if all(level is None for level in levels):
        raise NotImplementedError("Missing input strings. Try again.")
    elif country_location is None:
        raise NotImplementedError("Missing country_location string. Please check that you have supplied this string.")
    elif state_location is None:
        return os.path.join(datadir,  country_location, 'enrollment', f'{country_location}_enrollment_by_age.dat')
    elif location is None:
        return os.path.join(datadir,  country_location, state_location, 'enrollment', f'{state_location}_enrollment_by_age.dat')
    else:
        return os.path.join(datadir,  country_location, state_location, location, 'enrollment', f'{location}_enrollment_by_age.dat')

    # paths = cfg.FilePaths(None, state_location, country_location)
    # base = 'school_enrollment_by_age'
    # prefix = f"{location}_" + base
    # alt_prefix = None
    # """
    # # remove after data refactor
    # if country_location == 'usa':      # usa --remove after refactor
    #     prefix = prefix if location is None else prefix.format(location=location)
    # """
    # #remove after data e-factor
    # if cfg.alt_location is not None:
    #     alt_prefix = prefix
    # """
    #     alt_location = cfg.alt_location.location
    #     alt_state_location = cfg.alt_location.state_location
    #     alt_country_location = cfg.alt_location.country_location
    #     alt_prefix ="{location}_" + base

    #     if alt_country_location == 'usa':      # usa --remove after re-factor
    #         alt_prefix = alt_prefix if alt_location is None else alt_prefix.format(location=alt_location)
    # """
    # file = paths.get_demographic_file(location=None, filedata_type='enrollment', prefix=prefix, suffix='.dat', alt_prefix=alt_prefix)
    # return file


def get_school_enrollment_rates(datadir, location=None, state_location=None, country_location=None, file_path=None, use_default=False):
    """
    Get dictionary of enrollment rates by age. If use_default, then we'll first try to look for location specific
    data and if that's not available we'll use default data from Seattle, WA. This may not be appropriate for the population
    under study so it's best to provide as much data as you can for the specific population.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in
        file_path (string)        : file path to user specified gender by age bracket distribution data
        use_default (bool)        : if True, try to first use the other parameters to find data specific to the location under study, otherwise returns default data drawing from Seattle, Washington.

    Returns:
        A dictionary of school enrollment rates by age.
    """
    if file_path is None:
        file_path = get_school_enrollment_rates_path(datadir, location, state_location, country_location)

    try:
        df = pd.read_csv(file_path)
    except:
        if use_default:
            file_path = get_school_enrollment_rates_path(datadir, location=cfg.default_location, state_location=cfg.default_state, country_location=cfg.default_country)
            df = pd.read_csv(file_path)
        else:
            raise NotImplementedError(f"Data unavailable for the location specified. Please check input strings or set use_default to True to use default values from {cfg.default_location}, {cfg.default_state}.")
    return dict(zip(df.Age, df.Percent))


# Generalized function for any location that has enrollment sizes

def get_school_size_brackets_path(datadir, location=None, state_location=None, country_location=None):
    """
    Get file_path for school size brackets specific to the location under study.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in

    Returns:
        A file path to school size brackets.
    """
    datadir = get_relative_path(datadir)
    levels = [location, state_location, country_location]
    if all(level is None for level in levels):
        raise NotImplementedError("Missing input strings. Try again.")
    elif country_location is None:
        raise NotImplementedError("Missing country_location string. Please check that you have supplied this string.")
    elif state_location is None:
        return os.path.join(datadir, country_location, 'schools', f'{country_location}_school_size_brackets.dat')
    elif location is None:
        return os.path.join(datadir, country_location, state_location, 'schools', f'{state_location}_school_size_brackets.dat')
    else:
        return os.path.join(datadir, country_location, state_location, location, 'schools', f'{location}_school_size_brackets.dat')

    # paths = cfg.FilePaths(location, state_location, country_location)
    # #prefix = "{location}_school_size_brackets" if location is None else f"{location}_school_size_brackets"
    # base = "school_size_brackets"
    # prefix = f'{location}_' + base
    # alt_prefix = None

    # #remove after data e-factor
    # if cfg.alt_location is not None:
    #     alt_prefix = prefix

    # file = paths.get_demographic_file(location, filedata_type='schools', prefix=prefix,suffix='.dat', alt_prefix=alt_prefix)

    # return file


def get_school_size_brackets(datadir, location=None, state_location=None, country_location=None, file_path=None, use_default=False):
    """
    Get school size brackets: depends on the source/location of the data. If use_default, then we'll
    first try to look for location specific data and if that's not available we'll use default data from Seattle, WA.
    This may not be appropriate for the population under study so it's best to provide as much data as you can for the
    specific population.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in
        file_path (string)        : file path to user specified gender by age bracket distribution data
        use_default (bool)        : if True, try to first use the other parameters to find data specific to the location under study, otherwise returns default data drawing from Seattle, Washington.

    Returns:
        A dictionary of school size brackets.
    """
    if file_path is None:
        file_path = get_school_size_brackets_path(datadir, location, state_location, country_location)
    try:
        school_size_brackets = get_age_brackets_from_df(file_path)
        # print('whatschoolsizebrackets', school_size_brackets)
    except:
        if use_default:
            file_path = get_school_size_brackets_path(datadir, location=cfg.default_location, state_location=cfg.default_state, country_location=cfg.default_country)
            school_size_brackets = get_age_brackets_from_df(file_path)
        else:
            raise NotImplementedError(f"Data unavailable for the location specified. Please check input strings or set use_default to True to use default values from {cfg.default_location}, {cfg.default_state}.")
    return school_size_brackets


def get_school_size_distr_by_brackets_path(datadir, location=None, state_location=None, country_location=None):
    """
    Get file_path for the distribution of school size by brackets.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in

    Returns:
        A file path to the distribution of school sizes by bracket.
    """
    datadir = get_relative_path(datadir)
    levels = [location, state_location, country_location]
    if all(level is None for level in levels):
        raise NotImplementedError("Missing input strings. Try again.")
    elif country_location is None:
        raise NotImplementedError("Missing country_location string. Please check that you have supplied this string.")
    elif state_location is None:
        return os.path.join(datadir, country_location, 'schools', f'{country_location}_school_size_distr.dat')
    elif location is None:
        return os.path.join(datadir, country_location, state_location, 'schools', f'{state_location}_school_size_distr.dat')
    else:
        return os.path.join(datadir, country_location, state_location, location, 'schools', f'{location}_school_size_distr.dat')

    # paths = cfg.FilePaths(location, state_location, country_location)
    # base = 'school_size_distr'
    # prefix = f"{location}_" + base
    # alt_prefix = None

    # #remove after data e-factor
    # if cfg.alt_location is not None:
    #     alt_prefix = prefix

    # file = paths.get_demographic_file(location, filedata_type='schools', prefix=prefix, suffix='.dat', alt_prefix=alt_prefix)

    # return file


def get_school_size_distr_by_brackets(datadir, location=None, state_location=None, country_location=None, counts_available=False, file_path=None, use_default=False):
    """
    Get distribution of school sizes by bracket. Either you have enrollments by individual school or you have school size distribution that is binned. Either way, you want to get a school size distribution.
    If use_default, then we'll first try to look for location specific data and if that's not available we'll use default data from Seattle, WA. This may not
    be appropriate for the population under study so it's best to provide as much data as you can for the specific population.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in
        counts_available (bool)   : if True, a list of school sizes is available and a count of the sizes can be constructed
        file_path (string)        : file path to user specified distribution data
        use_default (bool)        : if True, try to first use the other parameters to find data specific to the location under study, otherwise returns default data drawing from Seattle, Washington.

    Returns:
        A dictionary of the distribution of school sizes by bracket.
    """
    if file_path is None:
        file_path = get_school_size_distr_by_brackets_path(datadir, location, state_location, country_location)
    try:
        df = pd.read_csv(file_path)
    except:
        if use_default:
            file_path = get_school_size_distr_by_brackets_path(datadir, location=cfg.default_location, state_location=cfg.default_state, country_location=cfg.default_country)
            df = pd.read_csv(file_path)
        else:
            raise NotImplementedError(f"Data unavailable for the location specified. Please check input strings or set use_default to True to use default values from {cfg.default_location}, {cfg.default_state}.")
    size_distr = dict(zip(df.size_bracket, df.percent))
    size_distr = spb.norm_dic(size_distr)

    return size_distr


# ### Default school type data ### #
def get_default_school_type_age_ranges():
    """
    Define and return default school types and the age range for each.

    Return:
        A dictionary of default school types and the age range for each.

    """
    school_type_age_ranges = {}
    school_type_age_ranges['pk'] = np.arange(3, 6)
    school_type_age_ranges['es'] = np.arange(6, 11)
    school_type_age_ranges['ms'] = np.arange(11, 14)
    school_type_age_ranges['hs'] = np.arange(14, 18)
    school_type_age_ranges['uv'] = np.arange(18, 100)

    return school_type_age_ranges


def get_default_school_types_by_age():
    """
    Define and return default probabilities of school type for each age.

    Return:
        A dictionary of default probabilities for the school type likely for each age.

    """
    school_type_age_ranges = get_default_school_type_age_ranges()

    school_types_by_age = {}
    for a in range(100):
        school_types_by_age[a] = dict.fromkeys(list(school_type_age_ranges.keys()), 0.)

    for k in school_type_age_ranges.keys():
        for a in school_type_age_ranges[k]:
            school_types_by_age[a][k] = 1.

    return school_types_by_age


def get_default_school_types_by_age_single():
    """
    Define and return default school type by age by assigning the school type with the highest probability.

    Return:
        A dictionary of default school type by age.

    """
    school_types_by_age = get_default_school_types_by_age()
    school_types_by_age_single = {}
    for a in range(100):
        values_to_keys_dic = {school_types_by_age[a][k]: k for k in school_types_by_age[a]}
        max_v = max(values_to_keys_dic.keys())
        max_k = values_to_keys_dic[max_v]
        school_types_by_age_single[a] = max_k

    return school_types_by_age_single


def get_default_school_size_distr_brackets():
    """
    Define and return default school size distribution brackets.

    Return:
        A dictionary of school size brackets.

    """
    return get_school_size_brackets(cfg.datadir, country_location='usa', use_default=True)


def get_default_school_size_distr_by_type():
    """
    Define and return default school size distribution for each school type. The school size distributions are binned to size groups or brackets.

    Return:
        A dictionary of school size distributions binned by size groups or brackets for each type of default school.

    """
    school_size_distr_by_type = {}

    school_types = ['pk', 'es', 'ms', 'hs', 'uv']

    for k in school_types:
        school_size_distr_by_type[k] = get_school_size_distr_by_brackets(cfg.datadir, country_location='usa', use_default=True)

    return school_size_distr_by_type





def get_school_size_distr_by_type_path(datadir, location=None, state_location=None, country_location=None):
    """
    Get file_path for the school size distribution by school type.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in

    Returns:
        str: A file path to the school size distribution data by different school types for the region specified.
    """
    datadir = get_relative_path(datadir)
    levels = [location, state_location, country_location]
    if all(level is None for level in levels):
        raise NotImplementedError("Missing input strings. Try again.")
    elif country_location is None:
        raise NotImplementedError("Missing country_location string. Please check that you have supplied this string.")
    elif state_location is None:
        return os.path.join(datadir, country_location, 'employment', f'{country_location}_school_size_distribution_by_type.dat')
    elif location is None:
        return os.path.join(datadir, country_location, state_location, 'employment', f'{state_location}_school_size_distribution_by_type.dat')
    else:
        return os.path.join(datadir, country_location, state_location, location, 'employment', f'{location}_school_size_distribution_by_type.dat')


def get_school_size_distr_by_type(datadir, location=None, state_location=None, country_location=None, file_path=None, use_default=None):
    """
    Get the school size distribution by school types. If use_default, then we'll try to look for location specific data first,
    and if that's not available we'll use default data from the set default locations (see sp.config.py). This may not be appropriate
    for the population under study so it's best to provide as much data as you can for the specific population.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in, which should be the 'usa'
        file_path (string)        : file path to user specified gender by age bracket distribution data
        use_default (bool)        : if True, try to first use the other parameters to find data specific to the location under study, otherwise returns default data drawing from Seattle, Washington.

    Returns:
        A dictionary of school size distributions binned by size groups or brackets for each type of default school.
    """

    if file_path is None:
        file_path = get_school_size_distr_by_type_path(datadir, location, state_location, country_location)

    try:
        f = open(file_path, 'r')
        data = json.load(f)

        # convert keys to ints for the size distribution by type
        for i in data:
            str_data_i = data[i].copy()
            if isinstance(str_data_i, dict):
                data[i] = {int(k): v for k, v in str_data_i.items()}
    except:
        if use_default:
            data = get_default_school_size_distr_by_type()  # convert to a static data file and then you can move data clean up to the end of the function
            # file_path = get_school_size_distr_by_brackets_path(datadir, location=cfg.default_location, state_location=cfg.default_state, country_location=cfg.default_country)
            # f = open(file_path, 'r')
            # data = json.load(f)
        else:
            raise ValueError(f"Data unavailable for the location specified. Please check input strings or set use_default to True to use default values from {cfg.location}, {cfg.default_state}, {cfg.default_country}.")

    # # convert keys to ints for the size distribution by type
    # for i in data:
    #     str_data_i = data[i].copy()
    #     if isinstance(str_data_i, dict):
    #         data[i] = {int(k): v for k, v in str_data_i.items()}

    return data


def get_employment_rates_path(datadir, location=None, state_location=None, country_location=None):
    """
    Get file_path for employment rates by age.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in

    Returns:
        A file path to employment rates by age.
    """
    # paths = cfg.FilePaths(location, state_location, country_location)
    # base = 'employment_rates_by_age'
    # prefix = f"{location}_" + base
    # alt_prefix = None

    # #remove after data e-factor
    # if cfg.alt_location is not None:
    #     apt_prefix = prefix

    # file = paths.get_demographic_file(location, filedata_type='employment', prefix=prefix, suffix='.dat', alt_prefix=alt_prefix)
    # print(file)
    # return file

    datadir = get_relative_path(datadir)
    levels = [location, state_location, country_location]
    if all(level is None for level in levels):
        raise NotImplementedError("Missing input strings. Try again.")
    elif country_location is None:
        raise NotImplementedError("Missing country_location string. Please check that you have supplied this string.")
    elif state_location is None:
        return os.path.join(datadir, country_location, 'employment', f'{country_location}_employment_rates_by_age.dat')
    elif location is None:
        return os.path.join(datadir, country_location, state_location, 'employment', f'{state_location}_employment_rates_by_age.dat')
    else:
        return os.path.join(datadir, country_location, state_location, location, 'employment', f'{location}_employment_rates_by_age.dat')


def get_employment_rates(datadir, location, state_location, country_location, file_path=None, use_default=False):
    """
    Get employment rates by age. If use_default, then we'll first try to look for location specific data and if that's not
    available we'll use default data from Seattle, WA. This may not be appropriate for the population under study so it's best
    to provide as much data as you can for the specific population.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in, which should be the 'usa'
        file_path (string)        : file path to user specified gender by age bracket distribution data
        use_default (bool)        : if True, try to first use the other parameters to find data specific to the location under study, otherwise returns default data drawing from Seattle, Washington.

    Returns:
        A dictionary of employment rates by age.
    """
    if file_path is None:
        file_path = get_employment_rates_path(datadir, location, state_location, country_location)

    try:
        df = pd.read_csv(file_path)
    except:
        if use_default:
            file_path = get_employment_rates_path(datadir, location=cfg.default_location, state_location=cfg.default_state, country_location=cfg.default_country)
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Data unavailable for the location specified. Please check input strings or set use_default to True to use default values from {cfg.default_location}, {cfg.default_state}.")
    return dict(zip(df.Age, df.Percent))


def get_workplace_size_brackets_path(datadir, location=None, state_location=None, country_location=None):
    """
    Get file_path for workplace size brackets.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in

    Returns:
        A file path to workplace size brackets.
    """

    # paths = cfg.FilePaths(location, state_location, country_location)
    # base = 'work_size_brackets'
    # prefix = f"{location}_" + base
    # alt_prefix = None

    # if cfg.alt_location is not None:
    #     apt_prefix = prefix

    # file = paths.get_demographic_file(location, filedata_type='workplaces', prefix=prefix, suffix='.dat', alt_prefix=alt_prefix)

    # return file

    datadir = get_relative_path(datadir)
    levels = [location, state_location, country_location]
    if all(level is None for level in levels):
        raise NotImplementedError("Missing input strings. Try again.")
    elif country_location is None:
        raise NotImplementedError("Missing country_location string. Please check that you have supplied this string.")
    elif state_location is None:
        return os.path.join(datadir, country_location, 'workplaces', f'{country_location}_work_size_brackets.dat')
    elif location is None:
        return os.path.join(datadir, country_location, state_location, 'workplaces', f'{state_location}_work_size_brackets.dat')
    else:
        return os.path.join(datadir, country_location, state_location, location, 'workplaces', f'{location}_work_size_brackets.dat')


def get_workplace_size_brackets(datadir, location=None, state_location=None, country_location=None, file_path=None, use_default=False):
    """
    Get workplace size brackets. If use_default, then we'll first try to look for location specific data and if that's not
    available we'll use default data from Seattle, WA. This may not be appropriate for the population under study so it's best
    to provide as much data as you can for the specific population.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in, which should be the 'usa'
        file_path (string)        : file path to user specified gender by age bracket distribution data
        use_default (bool)        : if True, try to first use the other parameters to find data specific to the location under study, otherwise returns default data drawing from Seattle, Washington.

    Returns:
        A dictionary of workplace size brackets.
    """

    if file_path is None:
        file_path = get_workplace_size_brackets_path(datadir, location, state_location, country_location)
    try:
        workplace_size_brackets = get_age_brackets_from_df(file_path)
    except:
        if use_default:
            file_path = get_workplace_size_brackets_path(datadir, state_location=cfg.default_state, country_location=cfg.default_country)
            workplace_size_brackets = get_age_brackets_from_df(file_path)
        else:
            raise NotImplementedError(f"Data unavailable for the location specified. Please check input strings or set use_default to True to use default values from {cfg.default_location}, {cfg.default_state}.")
    return workplace_size_brackets


def get_workplace_size_distr_by_brackets_path(datadir, location=None, state_location=None, country_location=None):
    """
    Get file_path for the distribution of workplace size by brackets.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in

    Returns:
        A file path to the distribution of workplace sizes by bracket.
    """

    # paths = cfg.FilePaths(location, state_location, country_location)
    # base = 'work_size_count'
    # prefix = f"{location}_" + base
    # alt_prefix = None

    # if cfg.alt_location is not None:
    #     apt_prefix = prefix

    # file = paths.get_demographic_file(location, filedata_type='workplaces', prefix=prefix, suffix='.dat', alt_prefix=alt_prefix)

    # return file

    datadir = get_relative_path(datadir)
    levels = [location, state_location, country_location]
    if all(level is None for level in levels):
        raise NotImplementedError("Missing input strings. Try again.")
    elif country_location is None:
        raise NotImplementedError("Missing country_location string. Please check that you have supplied this string.")
    elif state_location is None:
        return os.path.join(datadir, country_location, 'workplaces', f'{country_location}_work_size_count.dat')
    elif location is None:
        return os.path.join(datadir, country_location, state_location, 'workplaces', f'{state_location}_work_size_count.dat')
    else:
        return os.path.join(datadir, country_location, state_location, location, 'workplaces', f'{location}_work_size_count.dat')


def get_workplace_size_distr_by_brackets(datadir, location=None, state_location=None, country_location=None, file_path=None, use_default=False):
    """
    Get the distribution of workplace size by brackets.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in
        file_path (string)        : file path to user specified gender by age bracket distribution data
        use_default (bool)        : if True, try to first use the other parameters to find data specific to the location under study, otherwise returns default data drawing from Seattle, Washington.

    Returns:
        A dictionary of the distribution of workplace sizes by bracket.
    """
    if file_path is None:
        file_path = get_workplace_size_distr_by_brackets_path(datadir, location, state_location, country_location)

    try:
        df = pd.read_csv(file_path)
    except:
        if use_default:
            file_path = get_workplace_size_distr_by_brackets_path(datadir, state_location=cfg.default_state, country_location=cfg.default_country)
            df = pd.read_csv(file_path)
        else:
            raise NotImplementedError(f"Data unavailable for the location specified. Please check input strings or set use_default to True to use default values from {cfg.default_location}, {cfg.default_state}.")
    return dict(zip(df.work_size_bracket, df.size_count))


def get_state_postal_code(state_location, country_location):
    """
    Get the state postal code.

    Args:
        state_location (string)   : name of the state
        country_location (string) : name of the country the state is in

    Return:
        str: A postal code for the state_location.
    """
    base_dir = get_relative_path(cfg.datadir)
    file_path = os.path.join(base_dir,  country_location, 'postal_codes.csv')

    df = pd.read_csv(file_path, delimiter=',')
    dic = dict(zip(df.state, df.postal_code))
    return dic[state_location]


def get_usa_long_term_care_facility_path(datadir, state_location=None, country_location=None, part=None):
    """
    Get file_path for state level data on Long Term Care Providers for the US from 2015-2016.

    Args:
        datadir (string)          : file path to the data directory
        state_location (string)   : name of the state
        country_location (string) : name of the country the state is in
        part (int)                : part 1 or 2 of the table

    Returns:
        str: A file path to data on Long Term Care Providers from 'Long-Term Care Providers and Services Users in the United States - State Estimates Supplement: National Study of Long-Term Care Providers, 2015-2016'.
        Part 1 or 2 are available.
    """
    if country_location is None:
        raise NotImplementedError("Missing country_location string.")
    if state_location is None:
        raise NotImplementedError("Missing state_location string.")
    if part != 1 and part != 2:
        raise NotImplementedError("Part must be 1 or 2. Please try again.")
    postal_code = get_state_postal_code(state_location, country_location)
    datadir = get_relative_path(datadir)
    return os.path.join(datadir, country_location, state_location, 'assisted_living', f'LongTermCare_Table_48_Part{part}_{postal_code}_2015_2016.csv')

    # paths = cfg.FilePaths(None, state_location, country_location)
    # base = f'LongTermCare_Table_48_Part{part}_{postal_code}_2015_2016'
    # prefix = base
    # alt_prefix = None

    # if cfg.alt_location is not None:
    #     apt_prefix = prefix

    # file = paths.get_data_file(None, filedata_type='assisted_living', prefix=prefix, suffix='.csv', alt_prefix=alt_prefix)
    # return file


def get_usa_long_term_care_facility_data(datadir, state_location=None, country_location=None, part=None, file_path=None, use_default=False):
    """
    Get state level data table from National survey on Long Term Care Providers for the US from 2015-2016.

    Args:
        datadir (string)          : file path to the data directory
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in
        part (int)                : part 1 or 2 of the table
        file_path (string)        : file path to user specified distribution data
        use_default (bool)        : if True, try to first use the other parameters to find data specific to the location under study, otherwise returns default data drawing from Seattle, Washington.

    Returns:
        str: A file path to data on the size distribution of residents per facility for Long Term Care Facilities.
    """

    if file_path is None:
        file_path = get_usa_long_term_care_facility_path(datadir, state_location, country_location, part)
    try:
        df = pd.read_csv(file_path, header=2)
    except:
        if use_default:
            file_path = get_usa_long_term_care_facility_path(datadir, state_location=cfg.default_state, country_location=cfg.default_country, part=part)
            df = pd.read_csv(file_path, header=2)
        else:
            raise NotImplementedError(f"Data unavailable for the location specified. Please check input strings or set use_default to True to use default values from {cfg.default_state}, {cfg.default_country}.")
    return df


def get_usa_long_term_care_facility_residents_path(datadir, location=None, state_location=None, country_location=None):
    """
    Get file_path for the size distribution of residents per facility for Long Term Care Facilities.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in

    Returns:
        A file path to data on the size distribution of residents per facility for Long Term Care Facilities.
    """
    # paths = cfg.FilePaths(location, state_location, country_location)
    # base = 'aggregated_residents_distr'
    # prefix = f"{location}_" + base
    # alt_prefix = None

    # if cfg.alt_location is not None:
    #     apt_prefix = prefix

    # file = paths.get_data_file(location, filedata_type='assisted_living', prefix=prefix, suffix='.csv', alt_prefix=alt_prefix)

    # return file

    datadir = get_relative_path(datadir)
    levels = [location, state_location, country_location]
    if all(level is None for level in levels):
        raise NotImplementedError("Missing input strings. Try again.")
    elif country_location is None:
        raise NotImplementedError("Missing country_location string. Please check that you have supplied this string.")
    elif state_location is None:
        return os.path.join(datadir, country_location, 'assisted_living', f'{country_location}_aggregated_residents_distr.csv')
    elif location is None:
        return os.path.join(datadir, country_location, state_location, 'assisted_living', f'{state_location}_aggregated_residents_distr.csv')
    else:
        return os.path.join(datadir, country_location, state_location, location, 'assisted_living', f'{location}_aggregated_residents_distr.csv')


def get_usa_long_term_care_facility_residents_distr(datadir, location=None, state_location=None, country_location=None, file_path=None, use_default=None):
    """
    Get size distribution of residents per facility for Long Term Care Facilities.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in
        file_path (string)        : file path to user specified distribution data
        use_default (bool)        : if True, try to first use the other parameters to find data specific to the location under study, otherwise returns default data drawing from Seattle, Washington.

    Returns:
        A dictionary of the distribution of residents per facility for Long Term Care Facilities.
    """
    if file_path is None:
        file_path = get_usa_long_term_care_facility_residents_path(datadir, location=location, state_location=state_location, country_location=country_location)
    try:
        df = pd.read_csv(file_path, header=0)
    except:
        if use_default:
            file_path = get_usa_long_term_care_facility_residents_path(datadir, location=cfg.default_location, state_location=cfg.default_state, country_location=country_location)
            df = pd.read_csv(file_path, header=0)
        else:
            raise ValueError(f"Data unavailable for the location specified. Please check input strings or set use_default to True to use default values from {default_location}, {cfg.default_state}.")
    return dict(zip(df.bin, df.percent))


def get_usa_long_term_care_facility_residents_distr_brackets_path(datadir, location=None, state_location=None, country_location=None):
    """
    Get file_path for the size bins for the distribution of residents per facility for Long Term Care Facilities.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in

    Returns:
        A file path to data on the size bins for the distribution of residents per facility for Long Term Care Facilities.
    """

    # paths = cfg.FilePaths(location, state_location, country_location)
    # base = 'aggregated_residents_bins'
    # prefix = f"{location}_" + base
    # alt_prefix = None

    # if cfg.alt_location is not None:
    #     apt_prefix = prefix

    # file = paths.get_data_file(location, filedata_type='assisted_living', prefix=prefix, suffix='.csv', alt_prefix=alt_prefix)
    # return file

    datadir = get_relative_path(datadir)
    levels = [location, state_location, country_location]
    if all(level is None for level in levels):
        raise NotImplementedError("Missing input strings. Try again.")
    elif country_location is None:
        raise NotImplementedError("Missing country_location string. Please check that you have supplied this string.")
    elif state_location is None:
        return os.path.join(datadir, country_location, 'assisted_living', f'{country_location}_aggregated_residents_bins.csv')
    elif location is None:
        return os.path.join(datadir, country_location, state_location, 'assisted_living', f'{state_location}_aggregated_residents_bins.csv')
    else:
        return os.path.join(datadir, country_location, state_location, location, 'assisted_living', f'{location}_aggregated_residents_bins.csv')


def get_usa_long_term_care_facility_residents_distr_brackets(datadir, location=None, state_location=None, country_location=None, file_path=None, use_default=None):
    """
    Get size bins for the distribution of residents per facility for Long Term Care Facilities.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in, which should be the 'usa'
        size_distr_file_path (string)        : file path to user specified gender by age bracket distribution data
        use_default (bool)        : if True, try to first use the other parameters to find data specific to the location under study, otherwise returns default data drawing from Seattle, Washington.

    Returns:
        A dictionary of size brackets or bins for residents per facility.
    """

    if file_path is None:
        file_path = get_usa_long_term_care_facility_residents_distr_brackets_path(datadir, location, state_location, country_location)
    try:
        size_brackets = get_age_brackets_from_df(file_path)
    except:
        if use_default:
            file_path = get_usa_long_term_care_facility_residents_distr_brackets_path(datadir, location=cfg.default_location, state_location=cfg.default_state, country_location=cfg.default_country,)
            size_brackets = get_age_brackets_from_df(file_path)
        else:
            raise NotImplementedError(f"Data unavailable for the location specified. Please check input strings or set use_default to True to use default values from {cfg.default_location}, {cfg.default_state}.")
    return size_brackets


def get_usa_long_term_care_facility_resident_to_staff_ratios_path(datadir, location=None, state_location=None, country_location=None):
    """
    Get file_path for the distribution of resident to staff ratios per facility for Long Term Care Facilities.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in

    Returns:
        A file path to data on the distribution of resident to staff ratios per facility for Long Term Care Facilities.
    """

    # paths = cfg.FilePaths(location, state_location, country_location)
    # base = 'aggregated_resident_to_staff_ratios_distr'
    # prefix = f"{location}_" + base
    # alt_prefix = None

    # if cfg.alt_location is not None:
    #     apt_prefix = prefix

    # file = paths.get_data_file(location, filedata_type='assisted_living', prefix=prefix, suffix='.csv', alt_prefix=alt_prefix)

    # return file
    datadir = get_relative_path(datadir)
    levels = [location, state_location, country_location]
    if all(level is None for level in levels):
        raise NotImplementedError("Missing input strings. Try again.")
    elif country_location is None:
        raise NotImplementedError("Missing country_location string. Please check that you have supplied this string.")
    elif state_location is None:
        return os.path.join(datadir, country_location, 'assisted_living', f'{country_location}_aggregated_resident_to_staff_ratios_distr.csv')
    elif location is None:
        return os.path.join(datadir, country_location, state_location, 'assisted_living', f'{state_location}_aggregated_resident_to_staff_ratios_distr.csv')
    else:
        return os.path.join(datadir, country_location, state_location, location, 'assisted_living', f'{location}_aggregated_resident_to_staff_ratios_distr.csv')


def get_usa_long_term_care_facility_resident_to_staff_ratios_distr(datadir, location=None, state_location=None, country_location=None, file_path=None, use_default=None):
    """
    Get size distribution of resident to staff ratios per facility for Long Term Care Facilities.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in
        file_path (string)        : file path to user specified distribution data
        use_default (bool)        : if True, try to first use the other parameters to find data specific to the location under study, otherwise returns default data drawing from Seattle, Washington.

    Returns:
        A dictionary of the distribution of residents per facility for Long Term Care Facilities.
    """
    if file_path is None:
        file_path = get_usa_long_term_care_facility_resident_to_staff_ratios_path(datadir, location=location, state_location=state_location, country_location=country_location)
    try:
        df = pd.read_csv(file_path, header=0)
    except:
        if use_default:
            file_path = get_usa_long_term_care_facility_resident_to_staff_ratios_path(datadir, location=cfg.default_location, state_location=cfg.default_state, country_location=cfg.default_country)
            df = pd.read_csv(file_path, header=0)
        else:
            raise ValueError(f"Data unavailable for the location specified. Please check input strings or set use_default to True to use default values from {cfg.default_location}, {cfg.default_state}.")
    return dict(zip(df.bin, df.percent))


def get_usa_long_term_care_facility_resident_to_staff_ratios_brackets_path(datadir, location=None, state_location=None, country_location=None):
    """
    Get file_path for the size bins for the distribution of residents to staff ratios per facility for Long Term Care Facilities.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in

    Returns:
        str: A file path to data on the size bins for the distribution of resident to staff ratios per facility for Long Term Care Facilities.
    """
    datadir = get_relative_path(datadir)
    levels = [location, state_location, country_location]
    if all(level is None for level in levels):
        raise NotImplementedError("Missing input strings. Try again.")
    elif country_location is None:
        raise NotImplementedError("Missing country_location string. Please check that you have supplied this string.")
    elif state_location is None:
        return os.path.join(datadir, country_location, 'assisted_living', f'{country_location}_aggregated_resident_to_staff_ratios_bins.csv')
    elif location is None:
        return os.path.join(datadir, country_location, state_location, 'assisted_living', f'{state_location}_aggregated_resident_to_staff_ratios_bins.csv')
    else:
        return os.path.join(datadir, country_location, state_location, location, 'assisted_living', f'{location}_aggregated_resident_to_staff_ratios_bins.csv')


def get_usa_long_term_care_facility_resident_to_staff_ratios_brackets(datadir, location=None, state_location=None, country_location=None, file_path=None, use_default=None):
    """
    Get size bins for the distribution of resident to staff ratios per facility for Long Term Care Facilities.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in, which should be the 'usa'
        size_distr_file_path (string)        : file path to user specified gender by age bracket distribution data
        use_default (bool)        : if True, try to first use the other parameters to find data specific to the location under study, otherwise returns default data drawing from Seattle, Washington.

    Returns:
        A dictionary of size brackets or bins for resident to staff ratios per facility.
    """

    if file_path is None:
        file_path = get_usa_long_term_care_facility_residents_distr_brackets_path(datadir, location, state_location, country_location)
    try:
        size_brackets = get_age_brackets_from_df(file_path)
    except:
        if use_default:
            file_path = get_usa_long_term_care_facility_residents_distr_brackets_path(datadir, location=cfg.default_location, state_location=cfg.default_state, country_location=cfg.default_country)
            size_brackets = get_age_brackets_from_df(file_path)
        else:
            raise NotImplementedError("Data unavailable for the location specified. Please check input strings or set use_default to True to use default values from Seattle, Washington.")
    return size_brackets
