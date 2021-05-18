"""Create nepal json."""
import sciris as sc
import synthpops as sp
import numpy as np
import pandas as pd
import os


def process_age_dists():
    """Process and add age distribution data."""

    filepath = os.path.join(sp.settings.datadir, 'Nepal.json')
    location_data = sp.Location()
    location_data.location_name = 'Nepal'

    raw_data_path = os.path.join(sp.settings.datadir, 'Nepal')
    age_count_df = pd.read_csv(os.path.join(raw_data_path, 'Nepal-2019.csv'))

    age_count = np.array(age_count_df['M']) + np.array(age_count_df['F'])
    age_dist = age_count / age_count.sum()

    age_bin_labels = age_count_df['Age'].values
    data = dict()
    data['age_min'] = []
    data['age_max'] = []
    data['age_dist'] = []
    for bi, bl in enumerate(age_bin_labels):
        try:
            b = bl.split('-')
            b0, b1 = int(b[0]), int(b[1])
        except:
            b = bl.split('+')
            b0, b1 = int(b[0]), int(b[0])

        data['age_min'].append(b0)
        data['age_max'].append(b1)
        data['age_dist'].append(age_dist[bi])

    for k in data:
        data[k] = np.array(data[k])

    df = pd.DataFrame.from_dict(data)
    age_dist_arr = sp.convert_df_to_json_array(df, cols=df.columns, int_cols=['age_min', 'age_max'])

    location_data.population_age_distributions.append(sp.PopulationAgeDistribution())
    location_data.population_age_distributions[0].num_bins = len(age_dist_arr)
    location_data.population_age_distributions[0].distribution = age_dist_arr

    data_16 = sc.dcp(data)
    data_16['age_min'] = data_16['age_min'][:-5]
    data_16['age_max'] = data_16['age_max'][:-5]
    data_16['age_max'][-1] = 100
    data_16['age_dist'][-6] = data_16['age_dist'][-6:].sum()
    data_16['age_dist'] = data_16['age_dist'][:-5]

    df_16 = pd.DataFrame.from_dict(data_16)
    age_dist_arr_16 = sp.convert_df_to_json_array(df_16, cols=df_16.columns, int_cols=['age_min', 'age_max'])
    location_data.population_age_distributions.append(sp.PopulationAgeDistribution())
    location_data.population_age_distributions[1].num_bins = len(age_dist_arr_16)
    location_data.population_age_distributions[1].distribution = age_dist_arr_16

    sp.save_location_to_filepath(location_data, filepath)


def process_employment_rates():
    """Process and add employment data."""
    filepath = os.path.join(sp.settings.datadir, 'Nepal.json')
    location_data = sp.load_location_from_filepath(filepath)

    raw_data_path = os.path.join(sp.settings.datadir, 'Nepal')
    em_df = pd.read_csv(os.path.join(raw_data_path, 'employment_by_age.csv'))
    age_bin_labels = em_df['Age'].values
    binned_rates = em_df['EmploymentRate'].values
    employment_rates = dict.fromkeys(np.arange(101), 0)
    for bi, bl in enumerate(age_bin_labels):
        try:
            b = bl.split('-')
            b0, b1 = int(b[0]), int(b[1])
        except:
            b = bl.split('+')
            b0, b1 = int(b[0]), int(b[0]) + 10

        for a in np.arange(b0, b1 + 1):
            employment_rates[a] = binned_rates[bi]

    employment_rates_df = pd.DataFrame.from_dict(dict(age=np.arange(len(employment_rates)), percent=[employment_rates[a] for a in sorted(employment_rates.keys())]))
    location_data.employment_rates_by_age = sp.convert_df_to_json_array(employment_rates_df, cols=employment_rates_df.columns, int_cols=['age'])

    sp.save_location_to_filepath(location_data, filepath)


def process_enrollment_rates():
    """Process and add enrollment data."""
    filepath = os.path.join(sp.settings.datadir, 'Nepal.json')
    location_data = sp.load_location_from_filepath(filepath)

    raw_data_path = os.path.join(sp.settings.datadir, 'Nepal')
    en_df = pd.read_csv(os.path.join(raw_data_path, 'enrollment_by_age.csv'))
    age_bin_labels = en_df['Age'].values
    binned_rates = en_df['EnrollmentRate'].values
    enrollment_rates = dict.fromkeys(np.arange(101), 0)
    for bi, bl in enumerate(age_bin_labels):
        b = bl.split('-')
        b0, b1 = int(b[0]), int(b[1])

        for a in range(b0, b1 + 1):
            enrollment_rates[a] = binned_rates[bi]

    enrollment_rates_df = pd.DataFrame.from_dict(dict(age=np.arange(len(enrollment_rates)), percent=[enrollment_rates[a] for a in sorted(enrollment_rates.keys())]))
    location_data.enrollment_rates_by_age = sp.convert_df_to_json_array(enrollment_rates_df, cols=enrollment_rates_df.columns, int_cols=['age'])
    sp.save_location_to_filepath(location_data, filepath)

if __name__ == '__main__':

    # process_age_dists()
    # process_employment_rates()
    # process_enrollment_rates()
    filepath = os.path.join(sp.settings.datadir, 'Nepal.json')
    location_data = sp.load_location_from_filepath(filepath)
    print('loaded.')
