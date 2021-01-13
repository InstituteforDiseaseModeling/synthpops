"""
An example of how to use functions in sp.process_census to process some data
tables downloaded from the US Census Bureau into distribution tables that
sp.data_distribution functions might expect to work with.

Note: Before processing US Census Bureau data you should search for it on their
website, download it, rename it according to the pattern specified below and,
place it in the folder specified.

The raw data files can be downloaded from the US Census Bureau using their
customized data explorer tables (https://data.census.gov/cedsci/).
The files of interest will have names with patterns following the format

'ACSXXX{acs_period}Y{year}.YYYYY_data_with_overlays_ZZZZZZ.csv'

where XXX are some letters to indicate the main data type, YYYYY are a
combination of letters and numbers to indicate the specific data table, and
ZZZZZZ are another combination of letters and numbers to reference the
locations included in the table.

You should replace ZZZZZZ with the name of the location you want to process in
the data file (for this reason, download a single location's table at a time).

Next, place the table in the following folder path:

datadir/demographics/contact_matrices_152_countries/country_location/state_location/data_type/

where data_type might be one of the following strings:

age_distributions
household_size_distributions
employment
enrollment

Now you should be able to process raw data from the US Census Bureau with
the synthpops functions shown below.


Notice:

"This product uses the Census Bureau Data API but is not endorsed or certified by the Census Bureau."

"""

# Comment out this line after downloading the data
raise Exception('You must download the Census data (see above) before running this script')


import synthpops as sp

datadir = sp.datadir


# location, location_alias = 'Oregon', 'Oregon'
# location, location_alias = 'Portland-Vancouver-Hillsboro-OR-WA-Metro-Area', 'portland_metro'  # what shortened name do you want to use for this metro location
# state_location = 'Oregon'
# country_location = 'usa'


location, location_alias = 'King_County', 'King_County'
# location, location_alias = 'Spokane_County', 'Spokane_County'
# location, location_alias = 'Pierce_County', 'Pierce_County'
# location, location_alias = 'Yakima_County', 'Yakima_County'

state_location = 'Washington'


country_location = 'usa'

year = 2019
acs_period = 1

# process and write age distribution data
age_bracket_count, age_brackets = sp.process_us_census_age_counts(datadir, location, state_location, country_location, year, acs_period)
sp.write_age_bracket_distr_18(datadir, location_alias, state_location, country_location, age_bracket_count, age_brackets)
sp.write_age_bracket_distr_16(datadir, location_alias, state_location, country_location, age_bracket_count, age_brackets)

# process and write gender by age distribution data
age_bracket_count_by_gender, age_brackets = sp.process_us_census_age_counts_by_gender(datadir, location, state_location, country_location, year, acs_period)
sp.write_gender_age_bracket_distr_18(datadir, location_alias, state_location, country_location, age_bracket_count_by_gender, age_brackets)
sp.write_gender_age_bracket_distr_16(datadir, location_alias, state_location, country_location, age_bracket_count_by_gender, age_brackets)

# process and write household size distribution data
household_size_count = sp.process_us_census_household_size_count(datadir, location, state_location, country_location, year, acs_period)
sp.write_household_size_distr(datadir, location_alias, state_location, country_location, household_size_count)
sp.write_household_size_count(datadir, location_alias, state_location, country_location, household_size_count)

# process and write employment rates by age data
employment_rates = sp.process_us_census_employment_rates(datadir, location, state_location, country_location, year, acs_period)
sp.write_employment_rates(datadir, location_alias, state_location, country_location, employment_rates)

# process and write enrollment rates by age data
enrollment_rates = sp.process_us_census_enrollment_rates(datadir, location, state_location, country_location, year, acs_period)
sp.write_enrollment_rates(datadir, location_alias, state_location, country_location, enrollment_rates)

# process and write count of workplace sizes by bin
size_label_mappings, establishment_size_counts = sp.process_us_census_workplace_sizes(datadir, location, state_location, country_location, 2018)
sp.write_workplace_size_counts(datadir, location_alias, state_location, country_location, size_label_mappings, establishment_size_counts)
