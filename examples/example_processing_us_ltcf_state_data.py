"""
An example of how to use functions in sp.process_census to process some data
tables downloaded from the US Census Bureau into distribution tables that
sp.data_distribution functions might expect to work with.

See example_processing_us_census_bureau_data.py for further instructions.
"""

# Comment out this line after downloading the data
raise Exception('You must download the Census data (see above) before running this script')


import synthpops as sp


datadir = sp.datadir

state_location = 'Washington'
country_location = 'usa'
acs_period = 1

# save = True
save = False


age_brackets = sp.get_census_age_brackets(datadir, state_location=state_location, country_location=country_location, nbrackets=18)
age_by_brackets_dic = sp.get_age_by_brackets_dic(age_brackets)


ltcf_rates_by_age = sp.process_long_term_care_facility_rates_by_age(datadir, state_location, country_location)
if save:
    sp.write_long_term_care_facility_use_rates(datadir, state_location, country_location, ltcf_rates_by_age)
ltcf_rates_by_age = sp.get_long_term_care_facility_use_rates(datadir, state_location=state_location, country_location=country_location)


# use the data to estimate the number of long term care facility users for a local region and a given population size
local_population_size = 225e3
location = 'Seattle-Tacoma-Bellevue-WA-Metro-Area'
location = 'Washington'

local_age_distr, local_age_brackets = sp.process_us_census_age_counts(datadir, location, state_location, country_location, year=2018, acs_period=acs_period)
local_age_distr = sp.norm_dic(local_age_distr)

local_users = {}

for a in sorted(ltcf_rates_by_age.keys()):
    b = age_by_brackets_dic[a]
    local_users.setdefault(b, 0)
    local_users[b] += local_population_size * local_age_distr[b] / len(local_age_brackets[b]) * ltcf_rates_by_age[a]

print(f'Total long term care facility users for {location} with population size {local_population_size:.0f} is: {sum(local_users.values()):.0f}.')


# # use the Kaiser Health News data to estimate ratios
# location_alias = 'seattle_metro'  # saving data only for Seattle
# location_list = ['Seattle']

# location_alias = 'King_County'  # saving data for all locations in King County
# location_list = ['Seattle', 'Bellevue', 'Renton', 'Kent',
#                  'Federal Way', 'Redmond', 'Issaquah', 'Kirkland',
#                  'Shoreline', 'Burien', 'Tukwila', 'Sammamish',
#                  'Enumclaw', 'Snoqualmie',
#                  'North Bend', 'Seatac', 'Des Moines', 'Vashon',
#                  'Mercer Island', 'Kenmore', 'Lake Forest Park',
#                  'Normandy Park', 'White Center',
#                  ]

# location_alias = 'Washington'  # saving data for all locations in Washington state
# location_list = []

location_alias = 'Spokane_County'
location_list = ['Cheney', 'Spokane']

hist_df, bins_df = sp.process_usa_ltcf_resident_to_staff_ratios(datadir, country_location, state_location, location_alias, location_list, save)
print(f"Kaiser Health News resident to staff ratios for {location_alias}")
print(hist_df)
print(bins_df)
