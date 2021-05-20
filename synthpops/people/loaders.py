'''
Load data
'''

#%% Housekeeping
import numpy as np
import sciris as sc
from . import country_age_data    as cad
from . import state_age_data      as sad
from . import household_size_data as hsd


__all__ = ['default_age_data', 'get_country_aliases', 'map_entries', 'show_locations', 'get_age_distribution', 'get_household_size']


# Default age data, based on Seattle 2018 census data -- used in population.py
default_age_data = np.array([
    [ 0,  4, 0.0605],
    [ 5,  9, 0.0607],
    [10, 14, 0.0566],
    [15, 19, 0.0557],
    [20, 24, 0.0612],
    [25, 29, 0.0843],
    [30, 34, 0.0848],
    [35, 39, 0.0764],
    [40, 44, 0.0697],
    [45, 49, 0.0701],
    [50, 54, 0.0681],
    [55, 59, 0.0653],
    [60, 64, 0.0591],
    [65, 69, 0.0453],
    [70, 74, 0.0312],
    [75, 79, 0.02016], # Calculated based on 0.0504 total for >=75
    [80, 84, 0.01344],
    [85, 89, 0.01008],
    [90, 99, 0.00672],
])


def get_country_aliases():
    ''' Define aliases for countries with odd names in the data '''
    country_mappings = {
       'Bolivia':        'Bolivia (Plurinational State of)',
       'Burkina':        'Burkina Faso',
       'Cape Verde':     'Cabo Verdeo',
       'Hong Kong':      'China, Hong Kong Special Administrative Region',
       'Macao':          'China, Macao Special Administrative Region',
       "Cote d'Ivore":   'Côte d’Ivoire',
       "Ivory Coast":    'Côte d’Ivoire',
       'DRC':            'Democratic Republic of the Congo',
       'Iran':           'Iran (Islamic Republic of)',
       'Laos':           "Lao People's Democratic Republic",
       'Micronesia':     'Micronesia (Federated States of)',
       'Korea':          'Republic of Korea',
       'South Korea':    'Republic of Korea',
       'Moldova':        'Republic of Moldova',
       'Russia':         'Russian Federation',
       'Palestine':      'State of Palestine',
       'Syria':          'Syrian Arab Republic',
       'Taiwan':         'Taiwan Province of China',
       'Macedonia':      'The former Yugoslav Republic of Macedonia',
       'UK':             'United Kingdom of Great Britain and Northern Ireland',
       'United Kingdom': 'United Kingdom of Great Britain and Northern Ireland',
       'Tanzania':       'United Republic of Tanzania',
       'USA':            'United States of America',
       'United States':  'United States of America',
       'Venezuela':      'Venezuela (Bolivarian Republic of)',
       'Vietnam':        'Viet Nam',
        }

    return country_mappings # Convert to lowercase


def map_entries(json, location):
    '''
    Find a match between the JSON file and the provided location(s).

    Args:
        json (list or dict): the data being loaded
        location (list or str): the list of locations to pull from
    '''

    # The data have slightly different formats: list of dicts or just a dict
    countries = [key.lower() for key in json.keys()]

    # Set parameters
    if location is None:
        location = countries
    else:
        location = sc.promotetolist(location)

    # Define a mapping for common mistakes
    mapping = get_country_aliases()
    mapping = {key.lower(): val.lower() for key, val in mapping.items()}

    entries = {}
    for loc in location:
        lloc = loc.lower()
        if lloc not in countries and lloc in mapping:
            lloc = mapping[lloc]
        try:
            ind = countries.index(lloc)
            entry = list(json.values())[ind]
            entries[loc] = entry
        except ValueError as E:
            suggestions = sc.suggest(loc, countries, n=4)
            if suggestions:
                errormsg = f'Location "{loc}" not recognized, did you mean {suggestions}? ({str(E)})'
            else:
                errormsg = f'Location "{loc}" not recognized ({str(E)})'
            raise ValueError(errormsg)

    return entries


def show_locations(location=None, output=False):
    '''
    Print a list of available locations.

    Args:
        location (str): if provided, only check if this location is in the list
        output (bool): whether to return the list (else print)

    **Examples**::

        sp.people.show_locations() # Print a list of valid locations
        sp.people.show_locations('lithuania') # Check if Lithuania is a valid location
        sp.people.show_locations('Viet-Nam') # Check if Viet-Nam is a valid location

    New in version 1.10.0.
    '''
    country_json   = sc.dcp(cad.data)
    state_json     = sc.dcp(sad.data)
    aliases        = get_country_aliases()

    age_data       = sc.mergedicts(state_json, country_json, aliases) # Countries will overwrite states, e.g. Georgia
    household_data = sc.dcp(hsd.data)

    loclist = sc.objdict()
    loclist.age_distributions = sorted(list(age_data.keys()))
    loclist.household_size_distributions = sorted(list(household_data.keys()))

    if location is not None:
        age_available = location.lower() in [v.lower() for v in loclist.age_distributions]
        hh_available = location.lower() in [v.lower() for v in loclist.household_size_distributions]
        age_sugg = ''
        hh_sugg = ''
        age_sugg = f'(closest match: {sc.suggest(location, loclist.age_distributions)})' if not age_available else ''
        hh_sugg = f'(closest match: {sc.suggest(location, loclist.household_size_distributions)})' if not hh_available else ''
        print(f'For location "{location}":')
        print(f'  Population age distribution is available: {age_available} {age_sugg}')
        print(f'  Household size distribution is available: {hh_available} {hh_sugg}')
        return

    if output:
        return loclist
    else:
        print(f'There are {len(loclist.age_distributions)} age distributions and {len(loclist.household_size_distributions)} household size distributions.')
        print('\nList of available locations (case insensitive):\n')
        sc.pp(loclist)
        return


def get_age_distribution(location=None):
    '''
    Load age distribution for a given country or countries.

    Args:
        location (str or list): name of the country or countries to load the age distribution for

    Returns:
        age_data (array): Numpy array of age distributions, or dict if multiple locations

    New in version 1.10.0.
    '''

    # Load the raw data
    country_json   = sc.dcp(cad.data)
    state_json     = sc.dcp(sad.data)
    json = sc.mergedicts(state_json, country_json) # Countries will overwrite states, e.g. Georgia
    entries = map_entries(json, location)

    max_age = 99
    result = {}
    for loc,age_distribution in entries.items():
        total_pop = sum(list(age_distribution.values()))
        local_pop = []

        for age, age_pop in age_distribution.items():
            if age[-1] == '+':
                val = [int(age[:-1]), max_age, age_pop/total_pop]
            else:
                ages = age.split('-')
                val = [int(ages[0]), int(ages[1]), age_pop/total_pop]
            local_pop.append(val)
        result[loc] = np.array(local_pop)

    if len(result) == 1:
        result = list(result.values())[0]

    return result


def get_household_size(location=None):
    '''
    Load average household size distribution for a given country or countries.

    Args:
        location (str or list): name of the country or countries to load the age distribution for

    Returns:
        house_size (float): Size of household, or dict if multiple locations

    New in version 1.10.0.
    '''
    # Load the raw data
    json = sc.dcp(hsd.data)

    result = map_entries(json, location)
    if len(result) == 1:
        result = list(result.values())[0]

    return result
