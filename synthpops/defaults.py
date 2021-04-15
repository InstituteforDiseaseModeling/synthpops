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
        # 'household_size_1': True,
    },
    'defaults': {
        'country_location': 'usa',
        'state_location': 'Washington',
        'location': 'seattle_metro',
        'sheet_name': 'United States of America',
        'nbrackets' : 20,
        # 'household_size_1': True,
    },
    'usa': {
        'country_location': 'usa',
        'state_location': 'Washington',
        'location': 'seattle_metro',
        'sheet_name': 'United States of America',
        'nbrackets': 20,
        # 'household_size_1': True,
    },
}

defaults_config = sc.objdict()

defaults_config.thisdir = os.path.dirname(os.path.abspath(__file__))
defaults_config.localdatadir = os.path.join(defaults_config.thisdir, os.pardir, 'data')
defaults_config.datadir = defaults_config.localdatadir

defaults_config.relative_path = []

datadir = defaults_config.datadir

defaults_config.max_age = 101
defaults_config.nbrackets = 20
defaults_config.valid_nbracket_ranges = [16, 18, 20]
# defaults_config.household_size_1_included = 1

defaults_config.country_location = None
defaults_config.state_location = None
defaults_config.location = None
defaults_config.sheet_name = None
# defaults_config.default_country = None
# defaults_config.default_state = None
# defaults_config.default_location = None
# defaults_config.default_sheet_name = None


def reset_defaults_config_by_key(key, value):
    defaults_config[key] = value


def reset_defaults_config(new_config):
    for key, value in new_config.items():
        reset_defaults_config_by_key(key, value)
