"""
Defaults for synthpops files and data types.
"""

# specify default valid probability distributions - users can easily supply their own list if interested
valid_probability_distributions = [
    'population_age_distributions',
    'household_size_distribution',
    'ltcf_resident_to_staff_ratio_distribution',
    'ltcf_num_residents_distribution',
    'school_size_distribution',
]

defaults_config = {}
defaults_config['default_country'] = 'usa'


def reset_defaults_config(key, value):
    defaults_config[key] = value