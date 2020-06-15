"""
Modeling Seattle Metro Long Term Care Facilities

"""
import synthpops as sp
import numpy as np
import pandas as pd
import os
import math

# for pop of 2.25 million of Seattle
est_ltcf_user_by_age_brackets = {}
est_ltcf_user_by_age_brackets['60-64'] = 1390
est_ltcf_user_by_age_brackets['65-74'] = 1942
est_ltcf_user_by_age_brackets['75-84'] = 5285
est_ltcf_user_by_age_brackets['85-100'] = 7207

est_ltcf_user_by_age_brackets_perc = {}
est_ltcf_user_by_age_brackets_perc['55-59'] = 0.0098
est_ltcf_user_by_age_brackets_perc['60-64'] = 0.0098
est_ltcf_user_by_age_brackets_perc['65-69'] = 0.0098
est_ltcf_user_by_age_brackets_perc['70-74'] = 0.0098
est_ltcf_user_by_age_brackets_perc['75-79'] = 0.0546
est_ltcf_user_by_age_brackets_perc['80-84'] = 0.0546
est_ltcf_user_by_age_brackets_perc['85-100'] = 0.1780


pop = 2.25e6

country_location = 'usa'
state_location = 'Washington'
location = 'seattle_metro'

age_distr_18_fp = os.path.join(sp.datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, 'age distributions', 'seattle_metro_age_bracket_distr_18.dat')
age_distr_18 = sp.read_age_bracket_distr(sp.datadir, file_path=age_distr_18_fp)

age_brackets_18_fp = os.path.join(sp.datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, 'age distributions', 'census_age_brackets_18.dat')
age_brackets_18 = sp.get_census_age_brackets(sp.datadir, file_path=age_brackets_18_fp)

age_distr_18_2 = {}
age_distr_18_2[0] = 0.058
age_distr_18_2[1] = 0.058
age_distr_18_2[2] = 0.055
age_distr_18_2[3] = 0.053
age_distr_18_2[4] = 0.060
age_distr_18_2[5] = 0.092
age_distr_18_2[6] = 0.091
age_distr_18_2[7] = 0.079
age_distr_18_2[8] = 0.071
age_distr_18_2[9] = 0.069
age_distr_18_2[10] = 0.063
age_distr_18_2[11] = 0.056
age_distr_18_2[12] = 0.063
age_distr_18_2[13] = 0.048
age_distr_18_2[14] = 0.037
age_distr_18_2[15] = 0.021
age_distr_18_2[16] = 0.022
age_distr_18_2[17] = 0.018

print(np.sum([v for v in age_distr_18_2.values()]))

age_by_brackets_dic_18 = sp.get_age_by_brackets_dic(age_brackets_18)


gen_pop_size = 2.25e6
gen_pop_size = int(gen_pop_size)

exp_users_by_age = {}
exp_users_by_age_2 = {}

# for ab in est_ltcf_user_by_age_brackets_perc:
#     print(ab)
#     ab_split = int(ab.split('-')[0])
#     b = age_by_brackets_dic_18[ab_split]
#     for a in age_brackets_18[b]:
#         exp_users_by_age[a] = int(gen_pop_size * age_distr_18[b] / len(age_brackets_18[b]) * est_ltcf_user_by_age_brackets_perc[ab])

#         print(a, exp_users_by_age[a])

# print(np.sum([exp_users_by_age[a] for a in exp_users_by_age]))


# print(est_ltcf_user_by_age_brackets['60-64']/ pop / (age_distr_18[12]))
# print(est_ltcf_user_by_age_brackets['65-74']/ pop / (age_distr_18[13] + age_distr_18[14]))
# print(est_ltcf_user_by_age_brackets['75-84']/ pop / (age_distr_18[15] + age_distr_18[16]))
# print(est_ltcf_user_by_age_brackets['85-100']/ pop / (age_distr_18[17]))

# for b in age_brackets_18:
    # print(age_brackets_18[b])

for a in range(60, 101):

    if a < 65:
        b = age_by_brackets_dic_18[a]
        # exp_users_by_age[a] = gen_pop_size * age_distr_18[b]/len(age_brackets_18[b])
        # exp_users_by_age[a] = exp_users_by_age[a] * est_ltcf_user_by_age_brackets['60-64'] / pop / age_distr_18[12]
        # exp_users_by_age[a] = int(math.ceil(exp_users_by_age[a]))

        # exp_users_by_age_2[a] = gen_pop_size * age_distr_18_2[b]/len(age_brackets_18[b])
        # exp_users_by_age_2[a] = exp_users_by_age_2[a] * est_ltcf_user_by_age_brackets_perc['60-64']
        # exp_users_by_age_2[a] = int(math.ceil(exp_users_by_age_2[a]))

        exp_users_by_age_2[a] = gen_pop_size * age_distr_18[b]/len(age_brackets_18[b])
        exp_users_by_age_2[a] = exp_users_by_age_2[a] * est_ltcf_user_by_age_brackets_perc['60-64']
        exp_users_by_age_2[a] = int(math.ceil(exp_users_by_age_2[a]))

    elif a < 75:
        b = age_by_brackets_dic_18[a]
        # exp_users_by_age[a] = gen_pop_size * age_distr_18[b]/len(age_brackets_18[b])
        # exp_users_by_age[a] = exp_users_by_age[a] * est_ltcf_user_by_age_brackets['65-74'] / pop / (age_distr_18[13] + age_distr_18[14])
        # exp_users_by_age[a] = int(math.ceil(exp_users_by_age[a]))

        # exp_users_by_age_2[a] = gen_pop_size * age_distr_18_2[b]/len(age_brackets_18[b])
        # exp_users_by_age_2[a] = exp_users_by_age_2[a] * est_ltcf_user_by_age_brackets_perc['70-74']
        # exp_users_by_age_2[a] = int(math.ceil(exp_users_by_age_2[a]))

        exp_users_by_age_2[a] = gen_pop_size * age_distr_18[b]/len(age_brackets_18[b])
        exp_users_by_age_2[a] = exp_users_by_age_2[a] * est_ltcf_user_by_age_brackets_perc['70-74']
        exp_users_by_age_2[a] = int(math.ceil(exp_users_by_age_2[a]))

    elif a < 85:
        b = age_by_brackets_dic_18[a]
        # exp_users_by_age[a] = gen_pop_size * age_distr_18[b]/len(age_brackets_18[b])
        # exp_users_by_age[a] = exp_users_by_age[a] * est_ltcf_user_by_age_brackets['75-84'] / pop / (age_distr_18[15] + age_distr_18[16])
        # exp_users_by_age[a] = int(math.ceil(exp_users_by_age[a]))

        # exp_users_by_age_2[a] = gen_pop_size * age_distr_18_2[b]/len(age_brackets_18[b])
        # exp_users_by_age_2[a] = exp_users_by_age_2[a] * est_ltcf_user_by_age_brackets_perc['80-84']
        # exp_users_by_age_2[a] = int(math.ceil(exp_users_by_age_2[a]))

        exp_users_by_age_2[a] = gen_pop_size * age_distr_18[b]/len(age_brackets_18[b])
        exp_users_by_age_2[a] = exp_users_by_age_2[a] * est_ltcf_user_by_age_brackets_perc['80-84']
        exp_users_by_age_2[a] = int(math.ceil(exp_users_by_age_2[a]))

    elif a < 101:
        b = age_by_brackets_dic_18[a]
        # exp_users_by_age[a] = gen_pop_size * age_distr_18[b]/len(age_brackets_18[b])
        # exp_users_by_age[a] = exp_users_by_age[a] * est_ltcf_user_by_age_brackets['85-100'] / pop / age_distr_18[17]
        # exp_users_by_age[a] = int(math.ceil(exp_users_by_age[a]))

        # exp_users_by_age_2[a] = gen_pop_size * age_distr_18_2[b]/len(age_brackets_18[b])
        # exp_users_by_age_2[a] = exp_users_by_age_2[a] * est_ltcf_user_by_age_brackets_perc['85-100']
        # exp_users_by_age_2[a] = int(math.ceil(exp_users_by_age_2[a]))

        exp_users_by_age_2[a] = gen_pop_size * age_distr_18[b]/len(age_brackets_18[b])
        exp_users_by_age_2[a] = exp_users_by_age_2[a] * est_ltcf_user_by_age_brackets_perc['85-100']
        exp_users_by_age_2[a] = int(math.ceil(exp_users_by_age_2[a]))


    # try:
    # print(a, exp_users_by_age[a])

# print(np.sum([exp_users_by_age[a] for a in exp_users_by_age]))
print(np.sum([exp_users_by_age_2[a] for a in exp_users_by_age_2]))


# KC facilities reporting cases - should account for 70% of all facilities
KC_snf_df = pd.read_csv(os.path.join('/home', 'dmistry', 'Dropbox (IDM)', 'dmistry_COVID-19', 'secure_King_County', 'IDM_CASE_FACILITY.csv'))
d = KC_snf_df.groupby(['FACILITY_ID']).mean()
# print(sorted(d['RESIDENT_TOTAL_COUNT'].values), d['RESIDENT_TOTAL_COUNT'].values.mean(), np.median(d['RESIDENT_TOTAL_COUNT'].values))

KC_residential_users = d['RESIDENT_TOTAL_COUNT'].values.sum()

KC_ltcf_sizes = d['RESIDENT_TOTAL_COUNT'].values

# print(KC_ltcf_sizes)
