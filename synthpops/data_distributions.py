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
from . import defaults
from . import logger
from . import data


def get_relative_path(datadir):
    """
    Get the path relative for the datadir.

    Args:
        datadir (str): path to a specified data directory

    Returns:
        str: A path relative to a specified data directory datadir
    """
    base_dir = datadir
    if len(defaults.settings.relative_path) > 1:
        base_dir = os.path.join(datadir, *defaults.settings.relative_path)
    return base_dir


def get_nbrackets():
    """Return the default number of age brackets."""
    return defaults.settings.nbrackets


def calculate_which_nbrackets_to_use(location_data, nbrackets=None):
    """
    Calculate the number of age brackets to use by default.

    Args:
        nbrackets (int): the number of age brackets to use

    Returns:
        int: The number of age brackets to use.
    """
    if nbrackets is None:
        nbrackets = [d.num_bins for d in location_data.population_age_distributions if d.num_bins is not None]
        if len(nbrackets):
            nbrackets = max(nbrackets)
        else:
            nbrackets = defaults.settings.nbrackets

    return nbrackets


def sanitize_location(location):
    """
    Process and return a valid name for a location.

    Args:
        location (str): name of the location

    Returns:
        str: A processed location name.
    """
    if location is None:
        return ""
    else:
        # No spaces in filenames.
        location = location.replace(" ", "_")
        # Our convention is to separate location segments with "-".
        location = location.replace("-", "_")
    return location


def calculate_location_filename(location, state_location, country_location):
    """
    Process a location filename.

    Args:
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in

    Returns:
        str: A filename for where the location data reside.
    """
    separator = "-"
    if location != "":
        filepath = separator.join([country_location, state_location, location])
    elif state_location != "":
        filepath = separator.join([country_location, state_location])
    else:
        filepath = country_location
    return filepath


def calculate_location_filepath(location, state_location, country_location):
    """
    Process a location filepath.

    Args:
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in

    Returns:
        str: A filename for where the location data reside.
    """
    logger.debug(f"Calculating filepath for (location, state_location, country_location) = "
                 f"({location}, {state_location}, {country_location})")
    location = sanitize_location(location)
    state_location = sanitize_location(state_location)
    country_location = sanitize_location(country_location)
    filename = calculate_location_filename(location, state_location, country_location)
    filename = f"{filename}.json"
    filepath = filename
    logger.debug(f"Filepath = {filepath}")
    return filepath


def load_location(specific_location, state_location, country_location, revert_to_default=None):
    """
    Loading json object for the location data.

    Args:
        specific_location (string) : name of the location
        state_location (string)    : name of the state the location is in
        country_location (string)  : name of the country the location is in
        revert_to_default (bool)   : If True, try to first find location specific data to return otherwise use default data specified by the default location

    Returns:
        str: A filename for where the location data reside.
    """
    if revert_to_default is None:
        revert_to_default = False
    location_filepath = calculate_location_filepath(specific_location, state_location, country_location)
    try:
        location_object = data.load_location_from_filepath(location_filepath)
        logger.debug(f"Loaded (location, state_location, country_location) = "
                     f"({specific_location}, {state_location}, {country_location}) "
                     f"from [{location_filepath}]")
        return location_object
    except:
        logger.warn(f"Failed to load location [{specific_location}], "
                    f"state_location [{state_location}], "
                    f"country_location [{country_location}], reverting to default.")
        if revert_to_default:
            return load_location(defaults.settings.location, defaults.settings.state_location, defaults.settings.country_location, revert_to_default=False)
        else:
            msg =   f"Data unavailable for " \
                    f"(location, state_location, country_location) = " \
                    f"({specific_location}, {state_location}, {country_location}). " \
                    f"Please check input strings, or set use_default to True to use the default values from " \
                    f"(location, state_location, country_location) = " \
                    f"({defaults.settings.location}, {defaults.settings.state_location}, {defaults.settings.country_location}). "
            raise NotImplementedError(msg)


def read_age_bracket_distr(datadir=None, location=None, state_location=None, country_location=None, nbrackets=None, file_path=None, use_default=False):
    """
    A dict of the age distribution by age brackets. If use_default, then we'll
    first try to look for location specific data and if that's not available
    we'll use default data from settings.location,
    settings.state_location, settings.country_location. This may not
    be appropriate for the population under study so it's best to provide as
    much data as you can for the specific population.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in
        file_path (string)        : file path to user specified age bracket distribution data
        use_default (bool)        : if True, try to first use the other parameters to find data specific to the location under study, otherwise returns default data drawing from the settings.location, settings.state_location, settings.country_location.

    Returns:
        dict: A dictionary of the age distribution by age bracket. Keys map to a
        range of ages in that age bracket.

    """
    # Use default if no file for this location.
    location_data = load_location(location, state_location, country_location, revert_to_default=use_default)
    nbrackets = calculate_which_nbrackets_to_use(location_data, nbrackets)
    age_brackets = location_data.get_population_age_distribution(nbrackets)
    # Use default if no data for this parameter.
    if use_default and (age_brackets is None or len(age_brackets) == 0):
        return read_age_bracket_distr(location=defaults.settings.location,
                                      state_location=defaults.settings.state_location,
                                      country_location=defaults.settings.country_location,
                                      use_default=False)
    percent = [age_bracket[2] for age_bracket in age_brackets]
    r = dict(zip(np.arange(len(age_brackets)), percent))
    return r


# TODO: need to adapt this to new data.py
def get_smoothed_single_year_age_distr(datadir=None, location=None, state_location=None, country_location=None, nbrackets=None, file_path=None, use_default=False, window_length=7):
    """
    A smoothed dict of the age distribution by single years. If use_default,
    then we'll first try to look for location specific data and if that's not
    available we'll use default data from settings.location,
    settings.state_location, settings.country_location. This may not
    be appropriate for the population under study so it's best to provide as
    much data as you can for the specific population. Using moving windows to
    smooth out the age distribution.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in
        file_path (string)        : file path to user specified age bracket distribution data
        use_default (bool)        : If True, try to first use the other parameters to find data specific to the location under study, otherwise returns default data drawing from the settings.location, settings.state_location, settings.country_location.
        window_length (int)       : length of window, in units of years, over which to average or smooth out age distribution

    Returns:
        dict: A dictionary of the age distribution by age bracket. Keys map to a
        range of ages in that age bracket.
    """
    age_bracket_distr = read_age_bracket_distr(datadir, location, state_location, country_location, nbrackets, file_path, use_default)
    age_brackets = get_census_age_brackets(datadir, country_location=country_location, state_location=state_location, location=location, nbrackets=nbrackets)
    age_by_brackets = spb.get_age_by_brackets(age_brackets)

    raw_age_distr = dict.fromkeys(age_by_brackets.keys(), 0)

    for a in raw_age_distr.keys():
        b = age_by_brackets[a]
        raw_age_distr[a] = age_bracket_distr[b] / len(age_brackets[b])

    smoothed_age_distr = raw_age_distr.copy()

    errormsg = f"The window_length should be a non-negative integer value less than 10. The supplied value is: {window_length}. Please try another value between 0 and 10."

    if not isinstance(window_length, (int, np.int32, np.int64)) or window_length < 0 or window_length >= 10:
        raise ValueError(errormsg)

    window_half = window_length // 2

    for a in range(window_half, max(smoothed_age_distr.keys()) - window_half + 1):

        smoothed_age_distr[a] = np.mean([raw_age_distr[ai] for ai in range(a - window_half, a + window_half + 1)])

    # check all values are greater than 0
    min_smoothed_val = min(smoothed_age_distr.values())
    if min_smoothed_val < 0:
        errormsg2 = f"The minimum value of the smoothed age distribution is: {min_smoothed_val}. All values of the distribution should be greater than or equal to 0. Check either the original age distribution or the window_length."
        raise ValueError(errormsg2)

    smoothed_age_distr = spb.norm_dic(smoothed_age_distr)

    return smoothed_age_distr


def get_household_size_distr(datadir=None, location=None, state_location=None, country_location=None, file_path=None, use_default=False):
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
        dict: A dictionary of the household size distribution data. Keys map to
        the household size as an integer, values are the percent of households
        of that size.
    """
    # Use default if no file for this location.
    location_data = load_location(location, state_location, country_location, revert_to_default=use_default)
    # Use default if no data for this parameter.
    if use_default and (location_data.household_size_distribution is None or len(location_data.household_size_distribution) == 0):
        return get_household_size_distr(location=defaults.settings.location,
                                        state_location=defaults.settings.state_location,
                                        country_location=defaults.settings.country_location,
                                        use_default=False)
    dist = [ [int(entry[0]), entry[1]] for entry in location_data.household_size_distribution ]
    r = dict(dist)
    return r


def get_head_age_brackets(datadir=None, location=None, state_location=None, country_location=None, file_path=None, use_default=False):
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
        dict: A dictionary of the age brackets for head of household
        distribution data. Keys map to the age bracket as an integer, values are
        the percent of households which head of household in that age bracket.
    """
    # Use default if no file for this location.
    location_data = load_location(location, state_location, country_location, revert_to_default=use_default)
    # Use default if no data for this parameter.
    if use_default and (location_data.household_head_age_brackets is None or len(location_data.household_head_age_brackets) == 0):
        return get_head_age_brackets(location=defaults.settings.location,
                                     state_location=defaults.settings.state_location,
                                     country_location=defaults.settings.country_location,
                                     use_default=False)
    age_brackets = {}
    for [bracket_index, bracket_minmax] in enumerate(location_data.household_head_age_brackets):
        age_brackets[bracket_index] = np.arange(int(bracket_minmax[0]), int(bracket_minmax[1]) + 1)
    return age_brackets


def get_head_age_by_size_distr(datadir=None, location=None, state_location=None, country_location=None, file_path=None, use_default=False):
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
        ndarray: An array where each row s represents the age distribution of
        the head of households for households of size s-1.
    """
    # Use default if no file for this location.
    location_data = load_location(location, state_location, country_location, revert_to_default=use_default)
    # Use default if no data for this parameter.
    if use_default and (location_data.household_head_age_distribution_by_family_size is None or len(location_data.household_head_age_distribution_by_family_size) == 0):
        return get_head_age_by_size_distr(location=defaults.settings.location,
                                          state_location=defaults.settings.state_location,
                                          country_location=defaults.settings.country_location,
                                          use_default=False)
    dist = [d[1:] for d in location_data.household_head_age_distribution_by_family_size]
    return np.array(dist)


def get_census_age_brackets(datadir=None, location=None, state_location=None, country_location=None, file_path=None, use_default=False, nbrackets=None):
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
        dict: A dictionary of the range of ages that map to each age bracket.

    """
    # Use default if no file for this location.
    location_data = load_location(location, state_location, country_location, revert_to_default=use_default)

    # nbrackets = calculate_which_nbrackets_to_use(nbrackets)
    nbrackets = calculate_which_nbrackets_to_use(location_data, nbrackets)

    dist = location_data.get_population_age_distribution(nbrackets)

    # Use default if no data for this parameter.
    if use_default and (dist is None or len(dist) == 0):
        return get_census_age_brackets(location=defaults.settings.location,
                                       state_location=defaults.settings.state_location,
                                       country_location=defaults.settings.country_location,
                                       use_default=False)

    age_brackets = {}
    for bracket_index, dist in enumerate(dist):
        age_min = int(dist[0])
        age_max = int(dist[1])
        age_brackets[bracket_index] = np.arange(age_min, age_max + 1)
    return age_brackets

# TODO: still open question on how to handle these.
def get_contact_matrix(datadir, setting_code, sheet_name=None, file_path=None, delimiter=' ', header=None):
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
        ndarray: Matrix of contact patterns where each row i is the average
        contact patterns for an individual in age bracket i and the columns
        represent the age brackets of their contacts. The matrix element i,j is
        then the contact rate, number, or frequency for the average individual
        in age bracket i with all of their contacts in age bracket j in that
        physical contact setting.
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

# TODO: still open question on how to handle these.
def get_contact_matrices(datadir=None, sheet_name=None, file_path_dic=None, delimiter=' ', header=None, use_default=False):
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
        dict: A dictionary of the different contact matrices for each
        population, given by the sheet name. Keys map to the different possible
        physical contact settings for which data are available.
    """
    matrices = {}
    if file_path_dic is None:
        file_path_dic = dict.fromkeys(['H', 'S', 'W', 'C'], None)
    try:
        for setting_code in ['H', 'S', 'W', 'C']:
            matrices[setting_code] = get_contact_matrix(datadir, setting_code, sheet_name, file_path_dic[setting_code], delimiter, header)
    except:
        if use_default:
            for setting_code in ['H', 'S', 'W', 'C']:
                matrices[setting_code] = get_contact_matrix(datadir, setting_code, sheet_name=defaults.settings.sheet_name)
        else:
            raise NotImplementedError(f"Data unavailable for the location specified. Please check input strings or set use_default to True to use default values from the {defaults.settings.sheet_name}.")
    return matrices


def get_school_enrollment_rates(datadir=None, location=None, state_location=None, country_location=None, file_path=None, use_default=False):
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
        dict: A dictionary of school enrollment rates by age.
    """
    # Use default if no file for this location.
    location_data = load_location(location, state_location, country_location, revert_to_default=use_default)
    # Use default if no data for this parameter.
    if use_default and (location_data.enrollment_rates_by_age is None or len(location_data.enrollment_rates_by_age) == 0):
        return get_school_enrollment_rates(location=defaults.settings.location,
                                           state_location=defaults.settings.state_location,
                                           country_location=defaults.settings.country_location,
                                           use_default=False)

    dist = [ [int(d[0]), d[1]] for d in location_data.enrollment_rates_by_age ]
    return dict(dist)


def get_school_size_brackets(datadir=None, location=None, state_location=None, country_location=None, file_path=None, use_default=False):
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
        dict: A dictionary of school size brackets.
    """
    # Use default if no file for this location.
    location_data = load_location(location, state_location, country_location, revert_to_default=use_default)
    # Use default if no data for this parameter.
    if use_default and (location_data.school_size_brackets is None or len(location_data.school_size_brackets) == 0):
        return get_school_size_brackets(location=defaults.settings.location,
                                        state_location=defaults.settings.state_location,
                                        country_location=defaults.settings.country_location,
                                        use_default=False)

    school_size_brackets = {}
    for bracket_index, bracket in enumerate(location_data.school_size_brackets):
        size_min = int(bracket[0])
        size_max = int(bracket[1])
        school_size_brackets[bracket_index] = np.arange(size_min, size_max + 1)
    return school_size_brackets


def get_school_size_distr_by_brackets(datadir=None, location=None, state_location=None, country_location=None, file_path=None, use_default=False):
    """
    Get distribution of school sizes by size bracket or bin. If use_default,
    then we'll first try to look for location specific data and if that's not
    available we'll use default data from settings.location, settings.state_location,
    settings.country_location. This may not be appropriate for the population under study
    so it's best to provide as much data as you can for the specific population.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in
        file_path (string)        : file path to user specified school size distribution data
        use_default (bool)        : if True, try to first use the other parameters to find data specific to the location under study, otherwise returns default data drawing from settings.location, settings.state_location, settings.country_location.

    Returns:
        dict: A dictionary of the distribution of school sizes by bracket.
    """
    # Use default if no file for this location.
    location_data = load_location(location, state_location, country_location, revert_to_default=use_default)
    # Use default if no data for this parameter.
    if use_default and (location_data.school_size_distribution is None or len(location_data.school_size_distribution) == 0):
        return get_school_size_distr_by_brackets(location=defaults.settings.location,
                                                 state_location=defaults.settings.state_location,
                                                 country_location=defaults.settings.country_location,
                                                 use_default=False)

    size_distr = dict(enumerate(location_data.school_size_distribution))
    size_distr = spb.norm_dic(size_distr)
    return size_distr


# ### Default school type data ### #
def get_default_school_type_age_ranges():
    """
    Define and return default school types and the age range for each.

    Return:
        dict: A dictionary of default school types and the age range for each.
    """
    school_type_age_ranges = {}
    school_type_age_ranges['pk'] = np.arange(3, 6)
    school_type_age_ranges['es'] = np.arange(6, 11)
    school_type_age_ranges['ms'] = np.arange(11, 14)
    school_type_age_ranges['hs'] = np.arange(14, 18)
    school_type_age_ranges['uv'] = np.arange(18, 101)

    return school_type_age_ranges


def get_default_school_types_distr_by_age():
    """
    Define and return default probabilities of school type for each age.

    Return:
        dict: A dictionary of default probabilities for the school type likely
        for each age.
    """
    school_type_age_ranges = get_default_school_type_age_ranges()

    school_types_distr_by_age = {}
    for a in range(101):
        school_types_distr_by_age[a] = dict.fromkeys(list(school_type_age_ranges.keys()), 0.)

    for k in school_type_age_ranges.keys():
        for a in school_type_age_ranges[k]:
            school_types_distr_by_age[a][k] = 1.

    return school_types_distr_by_age


def get_default_school_types_by_age_single():
    """
    Define and return default school type by age by assigning the school type
    with the highest probability.

    Return:
        dict: A dictionary of default school type by age.

    """
    school_types_distr_by_age = get_default_school_types_distr_by_age()
    school_types_by_age_single = {}
    for a in range(101):
        values_to_keys = {school_types_distr_by_age[a][k]: k for k in school_types_distr_by_age[a]}
        max_v = max(values_to_keys.keys())
        max_k = values_to_keys[max_v]
        if max_v != 0:
            school_types_by_age_single[a] = max_k

    return school_types_by_age_single


def get_default_school_size_distr_brackets():
    """
    Define and return default school size distribution brackets.

    Return:
        dict: A dictionary of school size brackets.

    """
    return get_school_size_brackets(defaults.settings.datadir, country_location=defaults.settings.country_location, state_location=defaults.settings.state_location, location=defaults.settings.location, use_default=True)


def get_default_school_size_distr_by_type():
    """
    Define and return default school size distribution for each school type. The
    school size distributions are binned to size groups or brackets.

    Return:
        dict: A dictionary of school size distributions binned by size groups or
        brackets for each type of default school.
    """
    school_size_distr_by_type = {}

    school_types = ['pk', 'es', 'ms', 'hs', 'uv']

    for k in school_types:
        school_size_distr_by_type[k] = get_school_size_distr_by_brackets(defaults.settings.datadir, country_location=defaults.settings.country_location, state_location=defaults.settings.state_location, location=defaults.settings.location, use_default=True)

    return school_size_distr_by_type


def get_school_type_age_ranges(datadir=None, location=None, state_location=None, country_location=None, file_path=None, use_default=False):
    """
    Get a dictionary of the school types and the age range for each for the
    location specified.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in
        file_path (string)        : file path to user specified distribution data
        use_default (bool)        : if True, try to first use the other parameters to find data specific to the location under study, otherwise returns default data drawing from Seattle, Washington.

    Returns:
        dict: A dictionary of default school types and the age range for each.
    """
    # Use default if no file for this location.
    location_data = load_location(location, state_location, country_location, revert_to_default=use_default)
    # Use default if no data for this parameter.
    if use_default and (location_data.school_types_by_age is None or len(location_data.school_types_by_age) == 0):
        return get_school_type_age_ranges(location=defaults.settings.location,
                                          state_location=defaults.settings.state_location,
                                          country_location=defaults.settings.country_location,
                                          use_default=False)

    school_type_age_ranges = dict()
    for school_type_by_age in location_data.school_types_by_age:
        age_min = school_type_by_age.age_range[0]
        age_max = school_type_by_age.age_range[1]
        school_type_age_ranges[school_type_by_age.school_type] = np.arange(age_min, age_max + 1)
    return school_type_age_ranges


def get_school_size_distr_by_type(datadir=None, location=None, state_location=None, country_location=None, file_path=None, use_default=False):
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
        country_location (string) : name of the country the location is in
        file_path (string)        : file path to user specified distribution data
        use_default (bool)        : if True, try to first use the other parameters to find data specific to the location under study, otherwise returns default data drawing from settings.location, settings.state_location, settings.country_location

    Returns:
        dict: A dictionary of school size distributions binned by size groups or
        brackets for each type of default school.
    """
    # Use default if no file for this location.
    location_data = load_location(location, state_location, country_location, revert_to_default=use_default)
    # Use default if no data for this parameter.
    if use_default and (location_data.school_size_distribution_by_type is None or len(location_data.school_size_distribution_by_type) == 0):
        return get_school_size_distr_by_type(location=defaults.settings.location,
                                             state_location=defaults.settings.state_location,
                                             country_location=defaults.settings.country_location,
                                             use_default=False)

    school_size_distr_by_type = {}
    for dist_by_type in location_data.school_size_distribution_by_type:
        size_dist = dict(enumerate(dist_by_type.size_distribution))
        school_size_distr_by_type[dist_by_type.school_type] = size_dist
    return school_size_distr_by_type


def get_employment_rates(datadir=None, location=None, state_location=None, country_location=None, file_path=None, use_default=False):
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
        dict: A dictionary of employment rates by age.
    """
    # Use default if no file for this location.
    location_data = load_location(location, state_location, country_location, revert_to_default=use_default)
    # Use default if no data for this parameter.
    if use_default and (location_data.employment_rates_by_age is None or len(location_data.employment_rates_by_age) == 0):
        return get_employment_rates(location=defaults.settings.location,
                                    state_location=defaults.settings.state_location,
                                    country_location=defaults.settings.country_location,
                                    use_default=False)

    return dict(location_data.employment_rates_by_age)


def get_workplace_size_brackets(datadir=None, location=None, state_location=None, country_location=None, file_path=None, use_default=False):
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
        dict: A dictionary of workplace size brackets.
    """
    # Use default if no file for this location.
    location_data = load_location(location, state_location, country_location, revert_to_default=use_default)
    # Use default if no data for this parameter.
    if use_default and (location_data.workplace_size_counts_by_num_personnel is None or len(location_data.workplace_size_counts_by_num_personnel) == 0):
        return get_workplace_size_brackets(location=defaults.settings.location,
                                           state_location=defaults.settings.state_location,
                                           country_location=defaults.settings.country_location,
                                           use_default=False)

    workplace_size_brackets = dict()
    for bracket_index, bracket in enumerate(location_data.workplace_size_counts_by_num_personnel):
        size_min = int(bracket[0])
        size_max = int(bracket[1])
        workplace_size_brackets[bracket_index] = np.arange(size_min, size_max + 1)
    return workplace_size_brackets


def get_workplace_size_distr_by_brackets(datadir=None, location=None, state_location=None, country_location=None, file_path=None, use_default=False):
    """
    Get the distribution of workplace size by brackets. If use_default, then
    we'll first try to look for location specific data and if that's not
    available we'll use default data from settings.location, settings.state_location,
    settings.country_location. This may not be appropriate for the population under study
    so it's best to provide as much data as you can for the specific population.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in
        file_path (string)        : file path to user specified workplace size distribution data
        use_default (bool)        : if True, try to first use the other parameters to find data specific to the location under study, otherwise returns default data drawing from settings.location, settings.state_location, settings.country_location.

    Returns:
        dict: A dictionary of the distribution of workplace sizes by bracket.
    """
    # Use default if no file for this location.
    location_data = load_location(location, state_location, country_location, revert_to_default=use_default)
    # Use default if no data for this parameter.
    if use_default and (location_data.workplace_size_counts_by_num_personnel is None or len(location_data.workplace_size_counts_by_num_personnel) == 0):
        return get_workplace_size_distr_by_brackets(location=defaults.settings.location,
                                                    state_location=defaults.settings.state_location,
                                                    country_location=defaults.settings.country_location,
                                                    use_default=False)

    bracket_sizes = [ [bracket[0], bracket[1][2]]
                      for bracket in enumerate(location_data.workplace_size_counts_by_num_personnel) ]
    dist = dict(bracket_sizes)
    return dist


def get_state_postal_code(state_location, country_location):
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


def get_long_term_care_facility_residents_distr(datadir=None, location=None, state_location=None, country_location=None, file_path=None, use_default=False):
    """
    Get size distribution of residents per facility for Long Term Care
    Facilities.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in
        file_path (string)        : file path to user specified LTCF resident size distribution data
        use_default (bool)        : if True, try to first use the other parameters to find data specific to the location under study, otherwise returns default data drawing from settings.location, settings.state_location, settings.country_location.

    Returns:
        dict: A dictionary of the distribution of residents per facility for
        Long Term Care Facilities.
    """
    # Use default if no file for this location.
    location_data = load_location(location, state_location, country_location, revert_to_default=use_default)
    # Use default if no data for this parameter.
    if use_default and (location_data.ltcf_num_residents_distribution is None or len(location_data.ltcf_num_residents_distribution) == 0):
        return get_long_term_care_facility_residents_distr(location=defaults.settings.location,
                                                           state_location=defaults.settings.state_location,
                                                           country_location=defaults.settings.country_location,
                                                           use_default=False)

    bin_dist = [ [bracket[0], bracket[1][2]] for bracket in enumerate(location_data.ltcf_num_residents_distribution)]
    dist = dict(bin_dist)
    return dist


def get_long_term_care_facility_residents_distr_brackets(datadir=None, location=None, state_location=None, country_location=None, file_path=None, use_default=False):
    """
    Get size bins for the distribution of residents per facility for Long Term
    Care Facilities.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in, which should be the 'usa'
        file_path (string)        : file path to user specified LTCF resident size brackets data
        use_default (bool)        : if True, try to first use the other parameters to find data specific to the location under study, otherwise returns default data drawing from settings.location, settings.state_location, settings.country_location.

    Returns:
        dict: A dictionary of size brackets or bins for residents per facility.
    """
    # Use default if no file for this location.
    location_data = load_location(location, state_location, country_location, revert_to_default=use_default)
    # Use default if no data for this parameter.
    if use_default and (location_data.ltcf_num_residents_distribution is None or len(location_data.ltcf_num_residents_distribution) == 0):
        return get_long_term_care_facility_residents_distr_brackets(location=defaults.settings.location,
                                                                    state_location=defaults.settings.state_location,
                                                                    country_location=defaults.settings.country_location,
                                                                    use_default=False)

    num_residents_brackets = dict()
    for bracket_index, bracket in enumerate(location_data.ltcf_num_residents_distribution):
        min_num_residents = int(bracket[0])
        max_num_residents = int(bracket[1])
        num_residents_brackets[bracket_index] = np.arange(min_num_residents, max_num_residents + 1)
    return num_residents_brackets


def get_long_term_care_facility_resident_to_staff_ratios_distr(datadir=None, location=None, state_location=None, country_location=None, file_path=None, use_default=False):
    """
    Get size distribution of resident to staff ratios per facility for Long Term
    Care Facilities.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in
        file_path (string)        : file path to user specified resident to staff ratio distribution data
        use_default (bool)        : if True, try to first use the other parameters to find data specific to the location under study, otherwise returns default data drawing from settings.location, settings.state_location, settings.country_location.

    Returns:
        dict: A dictionary of the distribution of residents per facility for
        Long Term Care Facilities.
    """
    # Use default if no file for this location.
    location_data = load_location(location, state_location, country_location, revert_to_default=use_default)
    # Use default if no data for this parameter.
    if use_default and (location_data.ltcf_resident_to_staff_ratio_distribution is None or len(location_data.ltcf_resident_to_staff_ratio_distribution) == 0):
        return get_long_term_care_facility_resident_to_staff_ratios_distr(location=defaults.settings.location,
                                                                          state_location=defaults.settings.state_location,
                                                                          country_location=defaults.settings.country_location,
                                                                          use_default=False)

    bin_dist = [ [bracket[0], bracket[1][2]] for bracket in enumerate(location_data.ltcf_resident_to_staff_ratio_distribution)]
    dist = dict(bin_dist)
    return dist


def get_long_term_care_facility_resident_to_staff_ratios_brackets(datadir=None, location=None, state_location=None, country_location=None, file_path=None, use_default=False):
    """
    Get size bins for the distribution of resident to staff ratios per facility
    for Long Term Care Facilities.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in, which should be the 'usa'
        file_path (string)        : file path to user specified resident to staff ratio brackets data
        use_default (bool)        : if True, try to first use the other parameters to find data specific to the location under study, otherwise returns default data drawing from settings.location, settings.state_location, settings.country_location.

    Returns:
        dict: A dictionary of size brackets or bins for resident to staff ratios
        per facility.
    """
    # Use default if no file for this location.
    location_data = load_location(location, state_location, country_location, revert_to_default=use_default)
    # Use default if no data for this parameter.
    if use_default and (location_data.ltcf_resident_to_staff_ratio_distribution is None or len(location_data.ltcf_resident_to_staff_ratio_distribution) == 0):
        return get_long_term_care_facility_resident_to_staff_ratios_brackets(location=defaults.settings.location,
                                                                             state_location=defaults.settings.state_location,
                                                                             country_location=defaults.settings.country_location,
                                                                             use_default=False)

    ltcf_ratio_brackets = dict()
    for bracket_index, bracket in enumerate(location_data.ltcf_resident_to_staff_ratio_distribution):
        size_min = bracket[0]
        size_max = bracket[1]
        ltcf_ratio_brackets[bracket_index] = np.arange(size_min, size_max + 1)
    return ltcf_ratio_brackets


def get_long_term_care_facility_use_rates(datadir=None, location=None, state_location=None, country_location=None, file_path=None, use_default=False):
    """
    Get Long Term Care Facility use rates by age for a state.

    Args:
        datadir (str)          : file path to the data directory
        location_alias (str)   : more commonly known name of the location
        state_location (str)   : name of the state the location is in
        country_location (str) : name of the country the location is in
        file_path (string)     : file path to user specified gender by age bracket distribution data
        use_default (bool)     : if True, try to first use the other parameters to find data specific to the location under study, otherwise returns default data drawing from settings.location, settings.state_location, settings.country_location.

    Returns:
        dict: A dictionary of the Long Term Care Facility usage rates by age.

    Note:
        Currently only available for the United States.
    """
    # Use default if no file for this location.
    location_data = load_location(location, state_location, country_location, revert_to_default=use_default)
    # Use default if no data for this parameter.
    if use_default and (location_data.ltcf_use_rate_distribution is None or len(location_data.ltcf_use_rate_distribution) == 0):
        return get_long_term_care_facility_use_rates(location=defaults.settings.location,
                                                     state_location=defaults.settings.state_location,
                                                     country_location=defaults.settings.country_location,
                                                     use_default=False)

    dist = [[int(d[0]), d[1]] for d in location_data.ltcf_use_rate_distribution]
    return dict(dist)
