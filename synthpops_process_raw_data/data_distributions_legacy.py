"""
Read in data distributions.
Legacy version, prior to JSON location file format.
This is here to support 'scripts/migrate_legacy_data.py', which converts data from the old data schema
to the new one.
Version from commit 5aa01c9b6141208173e13d6e86656329a18b97df.
"""

import os  # pragma: no cover
import json # pragma: no cover
import numpy as np # pragma: no cover
import pandas as pd # pragma: no cover
import sciris as sc # pragma: no cover
from collections import Counter # pragma: no cover
from synthpops import base as spb # pragma: no cover
from synthpops import config as cfg # pragma: no cover
from synthpops import defaults


def get_relative_path(datadir): # pragma: no cover
    """
    Get the path relative for the datadir.

    Args:
        datadir (str): path to a specified data directory

    Returns:
        str: A path relative to a specified data directory datadir
    """
    base_dir = datadir
    if len(defaults.settings.relative_path):
        base_dir = os.path.join(datadir, *defaults.settings.relative_path)
    return base_dir


def get_nbrackets():  # pragma: no cover
    """Return the default number of age brackets."""
    return defaults.settings.nbrackets


def get_age_brackets_from_df(ab_file_path): # pragma: no cover
    """
    Create a dict of age bracket ranges from ab_file_path.

    Args:
        ab_file_path (string): file path to get the ends of different age
        brackets from

    Returns:
        A dictionary with a np.ndarray of the age range that maps to each age
        bracket key.

    **Examples**::

        get_age_brackets_from_df(ab_file_path) returns a dictionary
        age_brackets, where age_brackets[0] is the age range for the first age
        bracket, age_brackets[1] is the age range for the second age bracket,
        etc.

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


def get_age_bracket_distr_path(datadir, location=None, state_location=None, country_location=None, nbrackets=None): # pragma: no cover
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
    # datadir = get_relative_path(datadir)
    levels = [location, state_location, country_location]
    if nbrackets is None:
        nbrackets = defaults.settings.nbrackets
    if all(level is None for level in levels):
        raise NotImplementedError("Missing inputs. Please check that you have supplied the correct location and state_location strings.")
    elif country_location is None:
        raise NotImplementedError("Missing country_location string. Please check that you have supplied this string.")
    elif state_location is None:
        return os.path.join(datadir,  country_location, 'age_distributions', f'{country_location}_age_bracket_distr_{nbrackets}.dat')
    elif location is None:
        return os.path.join(datadir,  country_location, state_location, 'age_distributions', f'{state_location}_age_bracket_distr_{nbrackets}.dat')
    else:
        return os.path.join(datadir,  country_location, state_location, location, 'age_distributions', f'{location}_age_bracket_distr_{nbrackets}.dat')


def read_age_bracket_distr(datadir, location=None, state_location=None, country_location=None, nbrackets=None, file_path=None, use_default=False): # pragma: no cover
    """
    A dict of age distribution by age brackets. If use_default, then we'll first
    try to look for location specific data and if that's not available we'll use
    default data from settings.location, settings.state_location,
    settings.country_location. This may not be appropriate for the
    population under study so it's best to provide as much data as you can for
    the specific population.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in
        file_path (string)        : file path to user specified age bracket distribution data
        use_default (bool)        : if True, try to first use the other parameters to find data specific to the location under study, otherwise returns default data drawing from the settings.location, settings.state_location, settings.country_location.

    Returns:
        A dictionary of the age distribution by age bracket. Keys map to a range
        of ages in that age bracket.

    """
    if file_path is None:
        file_path = get_age_bracket_distr_path(datadir, location, state_location, country_location, nbrackets)

    try:
        df = pd.read_csv(file_path)
    except:
        if use_default:
            file_path = get_age_bracket_distr_path(datadir, location=defaults.settings.location, state_location=defaults.settings.state_location, country_location=defaults.settings.country_location, nbrackets=defaults.settings.nbrackets)
            df = pd.read_csv(file_path)
        else:
            raise NotImplementedError(f"Data unavailable for the location specified. Please check input strings or set use_default to True to use default values from {defaults.settings.location}, {defaults.settings.state_location}.")
    return dict(zip(np.arange(len(df)), df.percent))


def get_household_size_distr_path(datadir, location=None, state_location=None, country_location=None): # pragma: no cover
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
    # datadir = get_relative_path(datadir)
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


def get_household_size_distr(datadir, location=None, state_location=None, country_location=None, file_path=None, use_default=False): # pragma: no cover
    """
    A dictionary of the distribution of household sizes. If you don't give the
    file_path, then supply the location, state_location, and country_location
    strings. If use_default, then we'll first try to look for location specific
    data and if that's not available we'll use default data from
    settings.location, settings.state_location,
    settings.country_location. This may not be appropriate for the
    population under study so it's best to provide as much data as you can for
    the specific population.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in
        file_path (string)        : file path to user specified household size distribution data
        use_default (bool)        : if True, try to first use the other parameters to find data specific to the location under study, otherwise returns default data drawing from settings.location, settings.state_location, settings.country_location.

    Returns:
        A dictionary of the household size distribution data. Keys map to the
        household size as an integer, values are the percent of households of
        that size.

    """
    if file_path is None:
        file_path = get_household_size_distr_path(datadir, location, state_location, country_location)
    try:
        df = pd.read_csv(file_path)
    except:
        if use_default:
            file_path = get_household_size_distr_path(datadir, location=defaults.settings.location, state_location=defaults.settings.state_location, country_location=defaults.settings.country_location)
            df = pd.read_csv(file_path)
        else:
            raise NotImplementedError(f"Data unavailable for the location specified. Please check input strings or set use_default to True to use default values from {defaults.settings.location}, {defaults.settings.state_location}.")
    return dict(zip(df.household_size, df.percent))


def get_head_age_brackets_path(datadir, location=None, state_location=None, country_location=None): # pragma: no cover
    """
    Get file_path for head of household age brackets.

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


def get_head_age_brackets(datadir, location=None, state_location=None, country_location=None, file_path=None, use_default=False): # pragma: no cover
    """
    Get a dictionary of head age brackets either from the file_path directly, or
    using the other parameters to figure out what the file_path should be. If
    use_default, then we'll first try to look for location specific data and if
    that's not available we'll use default data from settings.location,
    settings.state_location, settings.country_location. This may not
    be appropriate for the population under study so it's best to provide as
    much data as you can for the specific population.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state
        country_location (string) : name of the country the state_location is in
        file_path (string)        : file path to user specified head age brackets data
        use_default (bool)        : if True, try to first use the other parameters to find data specific to the location under study, otherwise returns default data drawing from the settings.location, settings.state_location, settings.country_location.

    Returns:
        A dictionary of the age brackets for head of household distribution
        data. Keys map to the age bracket as an integer, values are the percent
        of households which head of household in that age bracket.

    """
    if file_path is None:
        file_path = get_head_age_brackets_path(datadir, location=location, state_location=state_location, country_location=country_location)
    try:
        age_brackets = get_age_brackets_from_df(file_path)
    except:
        if use_default:
            file_path = get_head_age_brackets_path(datadir, location=location, state_location=defaults.settings.state_location,
                                                   country_location=defaults.settings.country_location)
            age_brackets = get_age_brackets_from_df(file_path)
        else:
            raise NotImplementedError(f"Data unavailable for the location specified. Please check input strings or set use_default to True to use default values from {defaults.settings.location}, {defaults.settings.state_location}, {defaults.settings.country_location}.")
    return age_brackets


def get_household_head_age_by_size_path(datadir, location=None, state_location=None, country_location=None): # pragma: no cover
    """
    Get file_path for head of household age by size counts or distribution. If
    the data doesn't exist at the state level, only give the country_location.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state
        country_location (string) : name of the country the state_location is in

    Returns:
        A file path to the head of household age by household size count or
        distribution data.
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


def get_household_head_age_by_size_df(datadir, location=None, state_location=None, country_location=None, file_path=None, use_default=False): # pragma: no cover
    """
    Return a pandas df of head of household age by the size of the household. If
    the file_path is given return from there first. If use_default, then we'll
    first try to look for location specific data and if that's not available
    we'll use default data from settings.location,
    settings.state_location, settings.country_location. This may not
    be appropriate for the population under study so it's best to provide as
    much data as you can for the specific population.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state
        country_location (string) : name of the country the state_location is in
        file_path (string)        : file path to user specified data for the age of the head of the household by household size
        use_default (bool)        : if True, try to first use the other parameters to find data specific to the location under study, otherwise returns default data drawing from the settings.location, settings.state_location, settings.country_location.

    Returns:
        A file path to the head of household age by household size count or
        distribution data.
    """
    if file_path is None:
        file_path = get_household_head_age_by_size_path(datadir, location=location, state_location=state_location, country_location=country_location)
    try:
        df = pd.read_csv(file_path)
    except:
        if use_default:
            file_path = get_household_head_age_by_size_path(datadir, location=defaults.settings.location, state_location=defaults.settings.state_location, country_location=defaults.settings.country_location)
            df = pd.read_csv(file_path)
        else:
            raise NotImplementedError(f"Data unavailable for the location specified. Please check input strings or set use_default to True to use default values from {defaults.settings.location}, {defaults.settings.state_location}, {defaults.settings.country_location}.")
    return df


def get_head_age_by_size_distr(datadir, location=None, state_location=None, country_location=None, file_path=None, use_default=False): # pragma: no cover
    """
    Create an array of head of household age bracket counts (column) given by
    size (row). If use_default, then we'll first try to look for location
    specific data and if that's not available we'll use default data from the
    settings.location, settings.state_location,
    settings.country_location. This may not be appropriate for the
    population under study so it's best to provide as much data as you can for
    the specific population.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state
        country_location (string) : name of the country the state_location is in
        file_path (string)        : file path to user specified age of the head of the household by household size distribution data
        use_default (bool)        : if True, try to first use the other parameters to find data specific to the location under study, otherwise returns default data drawing from settings.location, settings.state_location, settings.country_location.

    Returns:
        An array where each row s represents the age distribution of the head of
        households for households of size s-1.
    """
    hha_df = get_household_head_age_by_size_df(datadir, location=location, state_location=state_location, country_location=country_location, file_path=file_path, use_default=use_default)

    hha_by_size = np.zeros((2 + len(hha_df), len(hha_df.columns)-1))

    # This was not originally part of 'data_distributions.py' that became this module, but instead was added after.
    # We detect if household_size_1_included is true or false based on the data.
    # if hha_df['family_size'][0] == 1:
    #     household_size_1_included = True

    # if household_size_1_included:
    #     for s in range(1, len(hha_df)+1):
    #         d = hha_df[hha_df['family_size'] == s].values[0][1:]
    #         hha_by_size[s-1] = d
    # else:
    #     hha_by_size[0, :] += 1
    #     for s in range(2, len(hha_df)+2):
    #         d = hha_df[hha_df['family_size'] == s].values[0][1:]
    #         hha_by_size[s-1] = d
    return hha_by_size


def get_census_age_brackets_path(datadir, location=None, state_location=None, country_location=None, nbrackets=None): # pragma: no cover
    """
    Get file_path for census age brackets: will depend on the state or country
    of the source data on the age distribution and age specific contact
    patterns.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state
        country_location (string) : name of the country the state_location is in

    Returns:
        A file path to the age brackets to be used with census age data in
        combination with the contact matrix data.
    """
    # datadir = get_relative_path(datadir)
    if nbrackets is None:
        nbrackets = get_nbrackets()

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


def get_census_age_brackets(datadir, location=None, state_location=None, country_location=None, file_path=None, use_default=False, nbrackets=None): # pragma: no cover
    """
    Get census age brackets: depends on the country or source of the age
    distribution and the contact pattern data. If use_default, then we'll first
    try to look for location specific data and if that's not available we'll use
    default data from settings.location, settings.state_location,
    settings.country_location. This may not be appropriate for the
    population under study so it's best to provide as much data as you can for
    the specific population.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state
        country_location (string) : name of the country the state_location is in
        file_path (string)        : file path to user specified census age brackets
        use_default (bool)        : if True, try to first use the other parameters to find data specific to the location under study, otherwise returns default data drawing from settings.location, settings.state_location, settings.country_location.

    Returns:
        A dictionary of the range of ages that map to each age bracket.

    """
    # if nbrackets is None:
    #     nbrackets = defaults.settings.nbrackets

    if file_path is None:
        file_path = get_census_age_brackets_path(datadir, location, state_location, country_location, nbrackets=nbrackets)

    try:
        age_brackets = get_age_brackets_from_df(file_path)
    except:
        if use_default:
            file_path = get_census_age_brackets_path(datadir, location=defaults.settings.location, state_location=defaults.settings.state_location, country_location=defaults.settings.country_location, nbrackets=nbrackets)
            age_brackets = get_age_brackets_from_df(file_path)
        else:
            raise NotImplementedError(f"Data unavailable for the location specified. Please check input strings or set use_default to True to use default values from {defaults.settings.location}, {defaults.settings.state_location}, {defaults.settings.country_location}.")
    return age_brackets


def get_contact_matrix(datadir, setting_code, sheet_name=None, file_path=None, delimiter=' ', header=None): # pragma: no cover
    """
    Get setting specific age contact matrix given sheet name to use. If
    file_path is given, then delimiter and header should also be specified.

    Args:
        datadir (string)          : file path to the data directory
        setting_code (string)     : name of the physial contact setting: H for households, S for schools, W for workplaces, C for community or other
        sheet_name (string)       : name of the sheet in the excel file with contact patterns
        file_path (string)        : file path to user specified age contact matrix
        delimiter (string)        : delimter for the contact matrix file
        header (int)              : row number for the header of the file

    Returns:
        Matrix of contact patterns where each row i is the average contact
        patterns for an individual in age bracket i and the columns represent
        the age brackets of their contacts. The matrix element i,j is then the
        contact rate, number, or frequency for the average individual in age
        bracket i with all of their contacts in age bracket j in that physical
        contact setting.
    """
    if file_path is None:
        setting_names = {'H': 'home', 'S': 'school', 'W': 'work', 'C': 'other_locations'}
        # base_dir = get_relative_path(datadir)

        if setting_code in setting_names:
            file_path = os.path.join(datadir, 'MUestimates_' + setting_names[setting_code] + '_1.xlsx')
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


def get_contact_matrix_dic(datadir, sheet_name=None, file_path_dic=None, delimiter=' ', header=None, use_default=False): # pragma: no cover
    # need review for additional countries
    """
    Create a dict of setting specific age contact matrices. If use_default, then
    we'll first try to look for location specific data and if that's not
    available we'll use default data from settings.sheet_name. This may
    not be appropriate for the population under study so it's best to provide as
    much data as you can for the specific population.

    Args:
        datadir (string)          : file path to the data directory
        setting_code (string)     : name of the physial contact setting: H for households, S for schools, W for workplaces, C for community or other
        sheet_name (string)       : name of the sheet in the excel file with contact patterns
        file_path_dic (string)    : dictionary to file paths of user specified age contact matrix, where keys are "H", "S", "W", and "C".
        delimiter (string)        : delimter for the contact matrix file
        header (int)              : row number for the header of the file

    Returns:
        A dictionary of the different contact matrices for each population,
        given by the sheet name. Keys map to the different possible physical
        contact settings for which data are available.

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
                matrix_dic[setting_code] = get_contact_matrix(datadir, setting_code, sheet_name=defaults.settings.sheet_name)
        else:
            raise NotImplementedError(f"Data unavailable for the location specified. Please check input strings or set use_default to True to use default values from the {defaults.settings.sheet_name}.")
    return matrix_dic


def get_school_enrollment_rates_path(datadir, location=None, state_location=None, country_location=None): # pragma: no cover
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
    # datadir = get_relative_path(datadir)
    levels = [location, state_location, country_location]
    if all(level is None for level in levels):
        raise NotImplementedError("Missing input strings. Try again.")
    elif country_location is None:
        raise NotImplementedError("Missing country_location string. Please check that you have supplied this string.")
    elif state_location is None:
        return os.path.join(datadir,  country_location, 'enrollment', f'{country_location}_enrollment_rates_by_age.dat')
    elif location is None:
        return os.path.join(datadir,  country_location, state_location, 'enrollment', f'{state_location}_enrollment_rates_by_age.dat')
    else:
        return os.path.join(datadir,  country_location, state_location, location, 'enrollment', f'{location}_enrollment_rates_by_age.dat')


def get_school_enrollment_rates(datadir, location=None, state_location=None, country_location=None, file_path=None, use_default=False): # pragma: no cover
    """
    Get dictionary of enrollment rates by age. If use_default, then we'll first
    try to look for location specific data and if that's not available we'll use
    default data from settings.location, settings.state_location,
    settings.country_location. This may not be appropriate for the
    population under study so it's best to provide as much data as you can for
    the specific population.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in
        file_path (string)        : file path to user specified school enrollment by age data
        use_default (bool)        : if True, try to first use the other parameters to find data specific to the location under study, otherwise returns default data drawing from settings.location, settings.state_location, settings.country_location.

    Returns:
        A dictionary of school enrollment rates by age.
    """
    if file_path is None:
        file_path = get_school_enrollment_rates_path(datadir, location, state_location, country_location)

    try:
        df = pd.read_csv(file_path)
    except:
        if use_default:
            file_path = get_school_enrollment_rates_path(datadir, location=defaults.settings.location, state_location=defaults.settings.state_location, country_location=defaults.settings.country_location)
            df = pd.read_csv(file_path)
        else:
            raise NotImplementedError(f"Data unavailable for the location specified. Please check input strings or set use_default to True to use default values from {defaults.settings.location}, {defaults.settings.state_location}, {defaults.settings.country_location}.")
    return dict(zip(df.Age, df.Percent))


# Generalized function for any location that has enrollment sizes

def get_school_size_brackets_path(datadir, location=None, state_location=None, country_location=None): # pragma: no cover
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
    # datadir = get_relative_path(datadir)
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


def get_school_size_brackets(datadir, location=None, state_location=None, country_location=None, file_path=None, use_default=False): # pragma: no cover
    """
    Get school size brackets: depends on the source/location of the data. If
    use_default, then we'll first try to look for location specific data and if
    that's not available we'll use default data from settings.location,
    settings.state_location, settings.country_location. This may not
    be appropriate for the population under study so it's best to provide as
    much data as you can for the specific population.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in
        file_path (string)        : file path to user specified school size brackets data
        use_default (bool)        : if True, try to first use the other parameters to find data specific to the location under study, otherwise returns default data drawing from settings.location, settings.state_location, settings.country_location.

    Returns:
        A dictionary of school size brackets.
    """
    if file_path is None:
        file_path = get_school_size_brackets_path(datadir, location, state_location, country_location)
    try:
        school_size_brackets = get_age_brackets_from_df(file_path)
    except:
        if use_default:
            file_path = get_school_size_brackets_path(datadir, location=defaults.settings.location, state_location=defaults.settings.state_location, country_location=defaults.settings.country_location)
            school_size_brackets = get_age_brackets_from_df(file_path)
        else:
            raise NotImplementedError(f"Data unavailable for the location specified. Please check input strings or set use_default to True to use default values from {defaults.settings.location}, {defaults.settings.state_location}, {defaults.settings.country_location}.")
    return school_size_brackets


def get_school_size_distr_by_brackets_path(datadir, location=None, state_location=None, country_location=None): # pragma: no cover
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
    # datadir = get_relative_path(datadir)
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


def get_school_size_distr_by_brackets(datadir, location=None, state_location=None, country_location=None, file_path=None, use_default=False): # pragma: no cover
    """
    Get distribution of school sizes by size bracket or bin. If use_default,
    then we'll first try to look for location specific data and if that's not
    available we'll use default data from settings.location,
    settings.state_location, settings.country_location. This may not
    be appropriate for the population under study so it's best to provide as
    much data as you can for the specific population.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in
        file_path (string)        : file path to user specified school size distribution data
        use_default (bool)        : if True, try to first use the other parameters to find data specific to the location under study, otherwise returns default data drawing from settings.location, settings.state_location, settings.country_location.

    Returns:
        A dictionary of the distribution of school sizes by bracket.
    """
    if file_path is None:
        file_path = get_school_size_distr_by_brackets_path(datadir, location, state_location, country_location)
    try:
        df = pd.read_csv(file_path)
    except:
        if use_default:
            file_path = get_school_size_distr_by_brackets_path(datadir, location=defaults.settings.location, state_location=defaults.settings.state_location, country_location=defaults.settings.country_location)
            df = pd.read_csv(file_path)
        else:
            raise NotImplementedError(f"Data unavailable for the location specified. Please check input strings or set use_default to True to use default values from {defaults.settings.location}, {defaults.settings.state_location}, {defaults.settings.country_location}.")
    size_distr = dict(zip(df.size_bracket, df.percent))
    size_distr = spb.norm_dic(size_distr)

    return size_distr


# ### Default school type data ### #

def get_default_school_type_age_ranges(): # pragma: no cover
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
    school_type_age_ranges['uv'] = np.arange(18, 101)

    return school_type_age_ranges


def get_default_school_types_distr_by_age(): # pragma: no cover
    """
    Define and return default probabilities of school type for each age.

    Return:
        A dictionary of default probabilities for the school type likely for each age.

    """
    school_type_age_ranges = get_default_school_type_age_ranges()

    school_types_distr_by_age = {}
    for a in range(101):
        school_types_distr_by_age[a] = dict.fromkeys(list(school_type_age_ranges.keys()), 0.)

    for k in school_type_age_ranges.keys():
        for a in school_type_age_ranges[k]:
            school_types_distr_by_age[a][k] = 1.

    return school_types_distr_by_age


def get_default_school_types_by_age_single(): # pragma: no cover
    """
    Define and return default school type by age by assigning the school type with the highest probability.

    Return:
        A dictionary of default school type by age.

    """
    school_types_distr_by_age = get_default_school_types_distr_by_age()
    school_types_by_age_single = {}
    for a in range(101):
        values_to_keys_dic = {school_types_distr_by_age[a][k]: k for k in school_types_distr_by_age[a]}
        max_v = max(values_to_keys_dic.keys())
        max_k = values_to_keys_dic[max_v]
        if max_v != 0:
            school_types_by_age_single[a] = max_k

    return school_types_by_age_single


def get_default_school_size_distr_brackets(): # pragma: no cover
    """
    Define and return default school size distribution brackets.

    Return:
        A dictionary of school size brackets.

    """
    return get_school_size_brackets(defaults.settings.datadir, country_location=defaults.settings.country_location, use_default=True)


def get_default_school_size_distr_by_type(): # pragma: no cover
    """
    Define and return default school size distribution for each school type. The school size distributions are binned to size groups or brackets.

    Return:
        A dictionary of school size distributions binned by size groups or brackets for each type of default school.

    """
    school_size_distr_by_type = {}

    school_types = ['pk', 'es', 'ms', 'hs', 'uv']

    for k in school_types:
        school_size_distr_by_type[k] = get_school_size_distr_by_brackets(defaults.settings.datadir, country_location=defaults.settings.country_location, use_default=True)

    return school_size_distr_by_type


def write_school_type_age_ranges(datadir, location, state_location, country_location, school_type_age_ranges): # pragma: no cover
    """
    Write to file the age range for each school type.

    Args:
        datadir (string)              : file path to the data directory
        location (string)             : name of the location
        state_location (string)       : name of the state the location is in
        country_location (string)     : name of the country the location is in
        school_type_age_ranges (dict) : a dictionary with the age range for each school type

    Returns:
        None.
    """
    school_type_age_ranges = sc.objdict(school_type_age_ranges)
    levels = [location, state_location, country_location]
    if all(level is None for level in levels):
        raise ValueError("Missing input strings. Try again.")
    elif country_location is None:
        raise ValueError("Missing country_location string. Please check that you have supplied this string.")
    elif state_location is None:
        file_path = os.path.join(datadir, country_location, 'schools', f'{country_location}_school_types_by_age_range.dat')
    elif location is None:
        file_path = os.path.join(datadir, country_location, state_location, 'schools', f'{state_location}_school_types_by_age_range.dat')
    else:
        file_path = os.path.join(datadir, country_location, state_location, location, 'schools', f'{location}_school_types_by_age_range.dat')

    with open(file_path, 'w') as f:
        f.write('school_type,age_range_min,age_range_max\n')
        for n, s, v in school_type_age_ranges.enumitems():
            f.write(f"{s},{v[0]:d},{v[-1]:d}\n")
    f.close()


def get_school_type_age_ranges_path(datadir, location=None, state_location=None, country_location=None): # pragma: no cover
    """
    Get file_path for the age range by school type.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in

    Returns:
        A file path to the age range for different school types.
    """
    # datadir = get_relative_path(datadir)
    levels = [location, state_location, country_location]
    if all(level is None for level in levels):
        raise NotImplementedError("Missing input strings. Try again.")
    elif country_location is None:
        raise NotImplementedError("Missing country_location string. Please check that you have supplied this string.")
    elif state_location is None:
        return os.path.join(datadir, country_location, 'schools', f'{country_location}_school_types_by_age_range.dat')
    elif location is None:
        return os.path.join(datadir, country_location, state_location, 'schools', f'{state_location}_school_types_by_age_range.dat')
    else:
        return os.path.join(datadir, country_location, state_location, location, 'schools', f'{location}_school_types_by_age_range.dat')


def get_school_type_age_ranges(datadir, location, state_location, country_location, file_path=None, use_default=None): # pragma: no cover
    """
    Get a dictionary of the school types and the age range for each for the location specified.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in
        file_path (string)        : file path to user specified distribution data
        use_default (bool)        : if True, try to first use the other parameters to find data specific to the location under study, otherwise returns default data drawing from Seattle, Washington.

    Returns:
        A dictionary of default school types and the age range for each.
    """

    if file_path is None:
        file_path = get_school_type_age_ranges_path(datadir, location, state_location, country_location)
    try:
        df = pd.read_csv(file_path)
    except:
        if use_default:
            return get_default_school_type_age_ranges()
        else:
            raise ValueError(f"Data unavailable for the location specified. Please check input strings or set use_default to True to use default values from {defaults.settings.location}, {defaults.settings.state_location}, {defaults.settings.country_location}.")

    z = zip(df.age_range_min, df.age_range_max)
    return dict(zip(df.school_type, [np.arange(i[0], i[1] + 1) for i in z]))


def get_school_size_distr_by_type_path(datadir, location=None, state_location=None, country_location=None): # pragma: no cover
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
    levels = [location, state_location, country_location]
    if all(level is None for level in levels):
        raise NotImplementedError("Missing input strings. Try again.")
    elif country_location is None:
        raise NotImplementedError("Missing country_location string. Please check that you have supplied this string.")
    elif state_location is None:
        return os.path.join(datadir, country_location, 'schools', f'{country_location}_school_size_distribution_by_type.dat')
    elif location is None:
        return os.path.join(datadir, country_location, state_location, 'schools', f'{state_location}_school_size_distribution_by_type.dat')
    else:
        return os.path.join(datadir, country_location, state_location, location, 'schools', f'{location}_school_size_distribution_by_type.dat')


def get_school_size_distr_by_type(datadir, location=None, state_location=None, country_location=None, file_path=None, use_default=None): # pragma: no cover
    """
    Get the school size distribution by school types. If use_default, then we'll
    try to look for location specific data first, and if that's not available
    we'll use default data from the set default locations (see sp.defaults.py).
    This may not be appropriate for the population under study so it's best to
    provide as much data as you can for the specific population.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in, which should be the 'usa'
        file_path (string)        : file path to user specified distribution data
        use_default (bool)        : if True, try to first use the other parameters to find data specific to the location under study, otherwise returns default data drawing from settings.location, settings.state_location, settings.country_location.

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
    except Exception as E:
        if use_default:
            data = get_default_school_size_distr_by_type()  # convert to a static data file and then you can move data clean up to the end of the function
            # file_path = get_school_size_distr_by_brackets_path(datadir, location=defaults.settings.location, state_location=defaults.settings.state_location, country_location=defaults.settings.country_location)
            # f = open(file_path, 'r')
            # data = json.load(f)
        else:
            raise ValueError(f"Data unavailable for the location specified ({str(E)}). Please check input strings or set use_default to True to use default values from {defaults.settings.location}, {defaults.settings.state_location}, {defaults.settings.country_location}.")

    return data


def get_employment_rates_path(datadir, location=None, state_location=None, country_location=None): # pragma: no cover
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
    # datadir = get_relative_path(datadir)
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


def get_employment_rates(datadir, location, state_location, country_location, file_path=None, use_default=False): # pragma: no cover
    """
    Get employment rates by age. If use_default, then we'll first try to look
    for location specific data and if that's not available we'll use default
    data from settings.location, settings.state_location,
    settings.country_location. This may not be appropriate for the
    population under study so it's best to provide as much data as you can for
    the specific population.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in, which should be the 'usa'
        file_path (string)        : file path to user specified employment by age data
        use_default (bool)        : if True, try to first use the other parameters to find data specific to the location under study, otherwise returns default data drawing from settings.location, settings.state_location, settings.country_location.

    Returns:
        A dictionary of employment rates by age.
    """
    if file_path is None:
        file_path = get_employment_rates_path(datadir, location, state_location, country_location)

    try:
        df = pd.read_csv(file_path)
    except:
        if use_default:
            file_path = get_employment_rates_path(datadir, location=defaults.settings.location, state_location=defaults.settings.state_location, country_location=defaults.settings.country_location)
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Data unavailable for the location specified. Please check input strings or set use_default to True to use default values from {defaults.settings.location}, {defaults.settings.state_location}, {defaults.settings.country_location}.")
    return dict(zip(df.Age, df.Percent))


def get_workplace_size_brackets_path(datadir, location=None, state_location=None, country_location=None): # pragma: no cover
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


def get_workplace_size_brackets(datadir, location=None, state_location=None, country_location=None, file_path=None, use_default=False): # pragma: no cover
    """
    Get workplace size brackets. If use_default, then we'll first try to look
    for location specific data and if that's not available we'll use default
    data from settings.location, settings.state_location,
    settings.country_location. This may not be appropriate for the
    population under study so it's best to provide as much data as you can for
    the specific population.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in, which should be the 'usa'
        file_path (string)        : file path to user specified workplace size brackets data
        use_default (bool)        : if True, try to first use the other parameters to find data specific to the location under study, otherwise returns default data drawing from settings.location, settings.state_location, settings.country_location.

    Returns:
        A dictionary of workplace size brackets.
    """

    if file_path is None:
        file_path = get_workplace_size_brackets_path(datadir, location, state_location, country_location)
    try:
        workplace_size_brackets = get_age_brackets_from_df(file_path)
    except:
        if use_default:
            file_path = get_workplace_size_brackets_path(datadir, location=defaults.settings.location, state_location=defaults.settings.state_location, country_location=defaults.settings.country_location)
            workplace_size_brackets = get_age_brackets_from_df(file_path)
        else:
            raise NotImplementedError(f"Data unavailable for the location specified. Please check input strings or set use_default to True to use default values from {defaults.settings.location}, {defaults.settings.state_location}, {defaults.settings.country_location}.")
    return workplace_size_brackets


def get_workplace_size_distr_by_brackets_path(datadir, location=None, state_location=None, country_location=None): # pragma: no cover
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
    # datadir = get_relative_path(datadir)
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


def get_workplace_size_distr_by_brackets(datadir, location=None, state_location=None, country_location=None, file_path=None, use_default=False): # pragma: no cover
    """
    Get the distribution of workplace size by brackets. If use_default, then
    we'll first try to look for location specific data and if that's not
    available we'll use default data from settings.location,
    settings.state_location, settings.country_location. This may not
    be appropriate for the population under study so it's best to provide as
    much data as you can for the specific population.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in
        file_path (string)        : file path to user specified workplace size distribution data
        use_default (bool)        : if True, try to first use the other parameters to find data specific to the location under study, otherwise returns default data drawing from settings.location, settings.state_location, settings.country_location.

    Returns:
        A dictionary of the distribution of workplace sizes by bracket.
    """
    if file_path is None:
        file_path = get_workplace_size_distr_by_brackets_path(datadir, location, state_location, country_location)

    try:
        df = pd.read_csv(file_path)
    except:
        if use_default:
            file_path = get_workplace_size_distr_by_brackets_path(datadir, location=defaults.settings.location, state_location=defaults.settings.state_location, country_location=defaults.settings.country_location)
            df = pd.read_csv(file_path)
        else:
            raise NotImplementedError(f"Data unavailable for the location specified. Please check input strings or set use_default to True to use default values from {defaults.settings.location}, {defaults.settings.state_location}, {defaults.settings.country_location}.")
    return dict(zip(df.work_size_bracket, df.size_count))


def get_state_postal_code(state_location, country_location): # pragma: no cover
    """
    Get the state postal code.

    Args:
        state_location (string)   : name of the state
        country_location (string) : name of the country the state is in

    Return:
        str: A postal code for the state_location.
    """
    file_path = os.path.join(defaults.settings.datadir, country_location, 'postal_codes.csv')

    df = pd.read_csv(file_path, delimiter=',')
    dic = dict(zip(df.state, df.postal_code))
    return dic[state_location]


def get_usa_long_term_care_facility_path(datadir, state_location=None, country_location=None, part=None): # pragma: no cover
    """
    Get file_path for state level data on Long Term Care Providers for the US
    from 2015-2016.

    Args:
        datadir (string)          : file path to the data directory
        state_location (string)   : name of the state
        country_location (string) : name of the country the state is in
        part (int)                : part 1 or 2 of the table

    Returns:
        str: A file path to data on Long Term Care Providers from 'Long-Term
        Care Providers and Services Users in the United States - State Estimates
        Supplement: National Study of Long-Term Care Providers, 2015-2016'. Part
        1 or 2 are available.
    """
    if country_location is None:
        raise NotImplementedError("Missing country_location string.")
    if state_location is None:
        raise NotImplementedError("Missing state_location string.")
    if part != 1 and part != 2:
        raise NotImplementedError("Part must be 1 or 2. Please try again.")
    postal_code = get_state_postal_code(state_location, country_location)
    return os.path.join(datadir, country_location, state_location, 'assisted_living', f'LongTermCare_Table_48_Part{part}_{postal_code}_2015_2016.csv')


def get_usa_long_term_care_facility_data(datadir, state_location=None, country_location=None, part=None, file_path=None, use_default=False): # pragma: no cover
    """
    Get state level data table from National survey on Long Term Care Providers
    for the US from 2015-2016.

    Args:
        datadir (string)          : file path to the data directory
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in
        part (int)                : part 1 or 2 of the table
        file_path (string)        : file path to user specified LTCF distribution data
        use_default (bool)        : if True, try to first use the other parameters to find data specific to the location under study, otherwise returns default data drawing from Seattle, Washington.

    Returns:
        str: A file path to data on the size distribution of residents per
        facility for Long Term Care Facilities.
    """
    if file_path is None:
        file_path = get_usa_long_term_care_facility_path(datadir, state_location, country_location, part)
    try:
        df = pd.read_csv(file_path, header=2)
    except:
        if use_default:
            file_path = get_usa_long_term_care_facility_path(datadir, state_location=defaults.settings.state_location, country_location=defaults.settings.country_location, part=part)
            df = pd.read_csv(file_path, header=2)
        else:
            raise NotImplementedError(f"Data unavailable for the location specified. Please check input strings or set use_default to True to use default values from {defaults.settings.state_location}, {defaults.settings.country_location}.")
    return df


def get_long_term_care_facility_residents_path(datadir, location=None, state_location=None, country_location=None): # pragma: no cover
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


def get_long_term_care_facility_residents_distr(datadir, location=None, state_location=None, country_location=None, file_path=None, use_default=None): # pragma: no cover
    """
    Get size distribution of residents per facility for Long Term Care
    Facilities.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in
        file_path (string)        : file path to user specified LTCF resident size distribution data
        use_default (bool)        : if True, try to first use the other parameters to find data specific to the location under study, otherwise returns default data drawing from Seattle, Washington.

    Returns:
        A dictionary of the distribution of residents per facility for Long Term
        Care Facilities.
    """
    if file_path is None:
        file_path = get_long_term_care_facility_residents_path(datadir, location=location, state_location=state_location, country_location=country_location)
    try:
        df = pd.read_csv(file_path, header=0)
    except:
        if use_default:
            file_path = get_long_term_care_facility_residents_path(datadir, location=defaults.settings.location, state_location=defaults.settings.state_location, country_location=defaults.settings.country_location)
            df = pd.read_csv(file_path, header=0)
        else:
            raise ValueError(f"Data unavailable for the location specified. Please check input strings or set use_default to True to use default values from {defaults.settings.location}, {defaults.settings.state_location}, {defaults.settings.country_location}.")
    return dict(zip(df.bin, df.percent))


def get_long_term_care_facility_residents_distr_brackets_path(datadir, location=None, state_location=None, country_location=None): # pragma: no cover
    """
    Get file_path for the size bins for the distribution of residents per
    facility for Long Term Care Facilities.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in

    Returns:
        A file path to data on the size bins for the distribution of residents
        per facility for Long Term Care Facilities.
    """
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


def get_long_term_care_facility_residents_distr_brackets(datadir, location=None, state_location=None, country_location=None, file_path=None, use_default=None): # pragma: no cover
    """
    Get size bins for the distribution of residents per facility for Long Term
    Care Facilities.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in, which should be the 'usa'
        file_path (string)        : file path to user specified LTCF resident size brackets data
        use_default (bool)        : if True, try to first use the other parameters to find data specific to the location under study, otherwise returns default data drawing from Seattle, Washington.

    Returns:
        A dictionary of size brackets or bins for residents per facility.
    """
    if file_path is None:
        file_path = get_long_term_care_facility_residents_distr_brackets_path(datadir, location, state_location, country_location)
    try:
        size_brackets = get_age_brackets_from_df(file_path)
    except:
        if use_default:
            file_path = get_long_term_care_facility_residents_distr_brackets_path(datadir, location=defaults.settings.location, state_location=defaults.settings.state_location, country_location=defaults.settings.country_location,)
            size_brackets = get_age_brackets_from_df(file_path)
        else:
            raise NotImplementedError(f"Data unavailable for the location specified. Please check input strings or set use_default to True to use default values from {defaults.settings.location}, {defaults.settings.state_location}, {defaults.settings.country_location}.")
    return size_brackets


def get_long_term_care_facility_resident_to_staff_ratios_path(datadir, location=None, state_location=None, country_location=None): # pragma: no cover
    """
    Get file_path for the distribution of resident to staff ratios per facility
    for Long Term Care Facilities.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in

    Returns:
        A file path to data on the distribution of resident to staff ratios per
        facility for Long Term Care Facilities.
    """
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


def get_long_term_care_facility_resident_to_staff_ratios_distr(datadir, location=None, state_location=None, country_location=None, file_path=None, use_default=None): # pragma: no cover
    """
    Get size distribution of resident to staff ratios per facility for Long Term
    Care Facilities.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in
        file_path (string)        : file path to user specified resident to staff ratio distribution data
        use_default (bool)        : if True, try to first use the other parameters to find data specific to the location under study, otherwise returns default data drawing from Seattle, Washington.

    Returns:
        A dictionary of the distribution of residents per facility for Long Term
        Care Facilities.
    """
    if file_path is None:
        file_path = get_long_term_care_facility_resident_to_staff_ratios_path(datadir, location=location, state_location=state_location, country_location=country_location)
    try:
        df = pd.read_csv(file_path, header=0)
    except:
        if use_default:
            file_path = get_long_term_care_facility_resident_to_staff_ratios_path(datadir, location=defaults.settings.location, state_location=defaults.settings.state_location, country_location=defaults.settings.country_location)
            df = pd.read_csv(file_path, header=0)
        else:
            raise ValueError(f"Data unavailable for the location specified. Please check input strings or set use_default to True to use default values from {defaults.settings.location}, {defaults.settings.state_location}, {defaults.settings.country_location}.")
    return dict(zip(df.bin, df.percent))


def get_long_term_care_facility_resident_to_staff_ratios_brackets_path(datadir, location=None, state_location=None, country_location=None): # pragma: no cover
    """
    Get file_path for the size bins for the distribution of residents to staff
    ratios per facility for Long Term Care Facilities.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in

    Returns:
        str: A file path to data on the size bins for the distribution of
        resident to staff ratios per facility for Long Term Care Facilities.
    """
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


def get_long_term_care_facility_resident_to_staff_ratios_brackets(datadir, location=None, state_location=None, country_location=None, file_path=None, use_default=None): # pragma: no cover
    """
    Get size bins for the distribution of resident to staff ratios per facility
    for Long Term Care Facilities.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in, which should be the 'usa'
        file_path (string)        : file path to user specified resident to staff ratio brackets data
        use_default (bool)        : if True, try to first use the other parameters to find data specific to the location under study, otherwise returns default data drawing from Seattle, Washington.

    Returns:
        A dictionary of size brackets or bins for resident to staff ratios per
        facility.
    """
    if file_path is None:
        file_path = get_long_term_care_facility_resident_to_staff_ratios_brackets_path(datadir, location, state_location, country_location)
    try:
        size_brackets = get_age_brackets_from_df(file_path)
    except:
        if use_default:
            file_path = get_long_term_care_facility_resident_to_staff_ratios_brackets_path(datadir, location=defaults.settings.location, state_location=defaults.settings.state_location, country_location=defaults.settings.country_location)
            size_brackets = get_age_brackets_from_df(file_path)
        else:
            raise NotImplementedError(f"Data unavailable for the location specified. Please check input strings or set use_default to True to use default values from {defaults.settings.location}, {defaults.settings.state_location}, {defaults.settings.country_location}.")
    return size_brackets


def get_long_term_care_facility_use_rates_path(datadir, state_location=None, country_location=None): # pragma: no cover
    """
    Get file_path for Long Term Care Facility use rates by age for a state.

    Args:
        datadir (str)          : file path to the data directory
        location_alias (str)   : more commonly known name of the location
        state_location (str)   : name of the state the location is in
        country_location (str) : name of the country the location is in

    Returns:
        str: A file path to the data on the Long Term Care Facility usage rates by age.

    Note:
        Currently only available for the United States.
    """
    levels = [state_location, country_location]
    if all(level is None for level in levels):
        raise NotImplementedError("Missing input strings. Try again.")
    if country_location is None:
        raise NotImplementedError("Missing country_location string. Please check that you have supplied this string.")
    elif state_location is None:
        return os.path.join(datadir, country_location, 'assisted_living', f'{country_location}_long_term_care_facility_use_rates_by_age.dat')
    return os.path.join(datadir, country_location, state_location, 'assisted_living', f'{state_location}_long_term_care_facility_use_rates_by_age.dat')


def get_long_term_care_facility_use_rates(datadir, state_location=None, country_location=None, file_path=None, use_default=None): # pragma: no cover
    """
    Get Long Term Care Facility use rates by age for a state.

    Args:
        datadir (str)          : file path to the data directory
        location_alias (str)   : more commonly known name of the location
        state_location (str)   : name of the state the location is in
        country_location (str) : name of the country the location is in
        file_path (string)     : file path to user specified gender by age bracket distribution data
        use_default (bool)     : if True, try to first use the other parameters to find data specific to the location under study, otherwise returns default data drawing from Seattle, Washington.


    Returns:
        dict: A dictionary of the Long Term Care Facility usage rates by age.

    Note:
        Currently only available for the United States.
    """
    if file_path is None:
        file_path = get_long_term_care_facility_use_rates_path(datadir, state_location, country_location)
    try:
        df = pd.read_csv(file_path)
    except:
        if use_default:
            file_path = get_long_term_care_facility_use_rates_path(datadir, state_location=defaults.settings.state_location, country_location=defaults.settings.country_location)
            df = pd.read_csv(file_path)
        else:
            raise NotImplementedError(f"Data unavailable for the location specified. Please check input strings or set use_default to True to use default values from {defaults.settings.state_location}, {defaults.settings.country_location}.")
    return dict(zip(df.Age, df.Percent))