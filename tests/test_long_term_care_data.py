import synthpops as sp
import numpy as np
import pandas as pd
import os
import math


datadir = sp.datadir
state_location = 'Washington'
country_location = 'usa'

part = 2
df = sp.get_usa_long_term_care_facility_data(datadir, state_location, part)

# df keys
age_bracket_keys = ['Under 65', '65–74', '75–84', '85 and over']
facility_keys = ['Nursing home', 'Residential care community']

# state numbers
facillity_users = {}
for fk in facility_keys:
    facillity_users[fk] = {}
    facillity_users[fk]['Total'] = int(df[df.iloc[:, 0] == 'Number of users2, 5'][fk].values[0].replace(',', ''))
    for ab in age_bracket_keys:
        facillity_users[fk][ab] = float(df[df.iloc[:, 0] == ab][fk].values[0].replace(',', ''))/100.

total_facility_users = np.sum([facillity_users[fk]['Total'] for fk in facillity_users])

# Census Bureau numbers
state_pop_2016 = 7288000
state_age_distr_2016 = {}
state_age_distr_2016['60-64'] = 6.3
state_age_distr_2016['65-74'] = 9.0
state_age_distr_2016['75-84'] = 4.0
state_age_distr_2016['85-100'] = 1.8

state_pop_2018 = 7535591
state_age_distr_2018 = {}
state_age_distr_2018['60-64'] = 6.3
state_age_distr_2018['65-74'] = 9.5
state_age_distr_2018['75-84'] = 4.3
state_age_distr_2018['85-100'] = 1.8

for a in state_age_distr_2016:
    state_age_distr_2016[a] = state_age_distr_2016[a]/100.
    state_age_distr_2018[a] = state_age_distr_2018[a]/100.

num_state_elderly_2016 = 0
num_state_elderly_2018 = 0
for a in state_age_distr_2016:
    num_state_elderly_2016 += state_pop_2016 * state_age_distr_2016[a]
    num_state_elderly_2018 += state_pop_2018 * state_age_distr_2018[a]

print('number of elderly',num_state_elderly_2016, num_state_elderly_2018)
print('growth in elderly', num_state_elderly_2018/num_state_elderly_2016)
print('users in 2016',total_facility_users, '% of elderly', total_facility_users/num_state_elderly_2016)

expected_users_2018 = total_facility_users * num_state_elderly_2018/num_state_elderly_2016
print('users in 2018', expected_users_2018)

# local Seattle metro stats (should be King County only but for now we'll use Seattle metro which includes Bellevue and Tacoma)
location = 'seattle_metro'
age_bracket_distr = sp.read_age_bracket_distr(datadir, location=location, state_location=state_location, country_location=country_location)
age_brackets = sp.get_census_age_brackets(datadir, state_location=state_location, country_location=country_location)

for ab in age_brackets:
    print(ab, age_brackets[ab][0], age_brackets[ab][-1], age_bracket_distr[ab])

# seattle_pop_2018 = 2233163
seattle_pop_2018 = 2250000
local_elderly_2018 = 0
for ab in range(12, 16):
    local_elderly_2018 += age_bracket_distr[ab] * seattle_pop_2018
print('number of local elderly', local_elderly_2018)

growth_since_2016 = num_state_elderly_2018/num_state_elderly_2016
local_perc_elderly_2018 = local_elderly_2018/num_state_elderly_2018

print('local users in 2018?', total_facility_users * local_elderly_2018/num_state_elderly_2018 * num_state_elderly_2018/num_state_elderly_2016)
seattle_users_est_from_state = total_facility_users * local_perc_elderly_2018 * growth_since_2016


# KC facilities reporting cases - should account for 70% of all facilities
KC_snf_df = pd.read_csv(os.path.join('/home', 'dmistry', 'Dropbox (IDM)', 'dmistry_COVID-19', 'secure_King_County', 'IDM_CASE_FACILITY.csv'))

d = KC_snf_df.groupby(['FACILITY_ID']).mean()
# print(sorted(d['RESIDENT_TOTAL_COUNT'].values), d['RESIDENT_TOTAL_COUNT'].values.mean(), np.median(d['RESIDENT_TOTAL_COUNT'].values))

KC_residential_users = d['RESIDENT_TOTAL_COUNT'].values.sum()
# print(KC_residential_users)

facilities_reporting_cases = 0.7  # not under reporting but rather the fraction that are reporting any case
seattle_users_est_from_KC_facilities_reporting = KC_residential_users/facilities_reporting_cases
print('second estimate of local users in 2018', seattle_users_est_from_KC_facilities_reporting)


est_seattle_users_2018 = dict.fromkeys(['60-64', '65-74', '75-84', '85-100'], 0)
for fk in facillity_users:
    for ab in facillity_users[fk]:
        if ab != 'Total':
            print(fk, ab, facillity_users[fk][ab], facillity_users[fk][ab] * facillity_users[fk]['Total'], facillity_users[fk][ab] * facillity_users[fk]['Total'] * seattle_pop_2018/state_pop_2018)
            if ab == 'Under 65':
                b = '60-64'
            elif ab == '65–74':
                b = '65-74'
            elif ab == '75–84':
                b = '75-84'
            elif ab == '85 and over':
                b = '85-100'
            est_seattle_users_2018[b] += facillity_users[fk][ab] * facillity_users[fk]['Total'] * seattle_pop_2018/state_pop_2018

for ab in est_seattle_users_2018:
    print(ab, est_seattle_users_2018[ab], est_seattle_users_2018[ab] / (state_age_distr_2018[ab] * seattle_pop_2018))

for ab in est_seattle_users_2018:
    est_seattle_users_2018[ab] = int(math.ceil(est_seattle_users_2018[ab]))

print(np.sum([est_seattle_users_2018[ab] for ab in est_seattle_users_2018]))




"""
From PH KC:
Abbreviation Description

SNF     Skilled nursing facility
AL      Assisted living
IL      Independent living
MC      Memory care


Notes:

National report at state level from 2015-2016 reports average capacity for different types of care
but this isn't facility capacity in total, rather the average capacity for that type of care in the
facility which may be added to capacity with other types of residential care for mixed types facilities.

For example, PH KC shows several Assisted living and Independent living combined in the same facility,
also very common to find Assisted living and Memory care together in the same facility. Memory care
requires more care hours so the capacity for it is lower per staff, but assisted living and Independent
living can require far less, so a combination is common. In fact, for 70% of reported KC facilities,
no Memory care facility is strictly a memory care facility.


"""


# est_users_perc = {}
# est_users_perc['65-74'] = 0.01
# est_users_perc['75-84'] = 0.05
# est_users_perc['85-100'] = 0.20

# for a in est_users_perc:
#     print(est_users_perc[a] * state_age_distr_2018[a] * seattle_pop_2018)
