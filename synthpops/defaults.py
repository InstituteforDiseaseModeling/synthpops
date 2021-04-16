"""
Defaults for synthpops files and data types.
"""
import sciris as sc
import os

# specify default valid probability distributions - users can easily supply their own list if interested
valid_probability_distributions = [
    'population_age_distributions',
    'household_size_distribution',
    'ltcf_resident_to_staff_ratio_distribution',
    'ltcf_num_residents_distribution',
    'school_size_distribution',
]

default_data = {
    'Senegal': {
        'country_location' : 'Senegal',
        'state_location'   : 'Dakar',
        'location'         : 'Dakar',
        'sheet_name'       : 'Senegal',
        'nbrackets'        : 18,
    },
    'defaults': {
        'country_location': 'usa',
        'state_location': 'Washington',
        'location': 'seattle_metro',
        'sheet_name': 'United States of America',
        'nbrackets' : 20,
    },
    'usa': {
        'country_location': 'usa',
        'state_location': 'Washington',
        'location': 'seattle_metro',
        'sheet_name': 'United States of America',
        'nbrackets': 20,
    },
}


def default_datadir_path():
    """Return the path to synthpops internal data folder."""
    thisdir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(thisdir, os.pardir, 'data')


# available globally if needed or via defaults.py
settings_config = sc.objdict()

settings_config.thisdir = os.path.dirname(os.path.abspath(__file__))
settings_config.localdatadir = default_datadir_path()
settings_config.datadir = settings_config.localdatadir

settings_config.relative_path = []


settings_config.max_age = 101
settings_config.nbrackets = 20
settings_config.valid_nbracket_ranges = [16, 18, 20]
# settings_config.household_size_1_included = 1

settings_config.country_location = None
settings_config.state_location = None
settings_config.location = None
settings_config.sheet_name = None


def reset_settings_config_by_key(key, value):
    """
    Reset a key in the globally available settings_config dictionary with a new value.

    Returns:
        None
    """
    settings_config[key] = value


def reset_settings_config(new_config):
    """
    Reset multiple keys in the globally available settings_config dictionary based on a new
    dictionary of values.

    Args:
        new_config (dict) : a dictionary with new values mapped to keys

    Returns:
        None.
    """
    for key, value in new_config.items():
        reset_settings_config_by_key(key, value)
