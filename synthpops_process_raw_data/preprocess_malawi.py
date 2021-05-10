"""Pre-process Malawi demographic data."""
import numpy as np
import pandas as pd
import sciris as sc
import synthpops as sp
import os


dir_path = os.path.join(sp.settings.datadir, 'Malawi')
file_path = os.path.join(dir_path, 'Series A. Population Tables.xlsx')

df = pd.read_excel(file_path, sheet_name='A5', header=1, skiprows=[2, 3], skipfooter=303)

ages = df['Age in single Years'].values[1:]
age_count = np.array(df['National'].values[1:])
age_range = np.arange(len(ages))

age_dist = age_count / age_count.sum()
age_dist_mapping = dict(zip(age_range, age_dist))

data = dict(age_min=sc.dcp(age_range), age_max=sc.dcp(age_range), age_dist=age_dist)
data['age_max'][-1] = 100
new_df = pd.DataFrame.from_dict(data)
print(new_df)

new_file_path = os.path.join(dir_path, 'Malawi_national_ages.csv')
new_df.to_csv(new_file_path, index=False)

census_age_brackets = sp.get_census_age_brackets(sp.settings.datadir, location='seattle-metro', state_location='Washington', country_location='usa', nbrackets=16)
census_age_by_brackets = sp.get_age_by_brackets(census_age_brackets)

print(census_age_brackets)
agg_ages = sp.get_aggregate_ages(age_dist_mapping, census_age_by_brackets)
print(agg_ages)
agg_data = dict()
agg_data['age_min'] = np.array([census_age_brackets[b][0] for b in census_age_brackets])
agg_data['age_max'] = np.array([census_age_brackets[b][-1] for b in census_age_brackets])
agg_data['age_dist'] = np.array([agg_ages[b] for b in sorted(census_age_brackets.keys())])
agg_df = pd.DataFrame.from_dict(agg_data)
print(agg_df)
agg_path = os.path.join(dir_path, 'Malawi_national_ages_16.csv')
agg_df.to_csv(agg_path, index=False)
