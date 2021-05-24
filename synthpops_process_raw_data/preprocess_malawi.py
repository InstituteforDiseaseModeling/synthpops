"""Pre-process Malawi demographic data."""
import numpy as np
import pandas as pd
import sciris as sc
import synthpops as sp
import os


dir_path = os.path.join(sp.settings.datadir, 'Malawi')


def process_age_tables():
    """Function to preprocess age tables."""
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

    new_file_path = os.path.join(dir_path, 'Malawi_national_ages.csv')
    new_df.to_csv(new_file_path, index=False)

    census_age_brackets = sp.get_census_age_brackets(sp.settings.datadir, location='seattle-metro', state_location='Washington', country_location='usa', nbrackets=16)
    census_age_by_brackets = sp.get_age_by_brackets(census_age_brackets)

    agg_ages = sp.get_aggregate_ages(age_dist_mapping, census_age_by_brackets)

    agg_data = dict()
    agg_data['age_min'] = np.array([census_age_brackets[b][0] for b in census_age_brackets])
    agg_data['age_max'] = np.array([census_age_brackets[b][-1] for b in census_age_brackets])
    agg_data['age_dist'] = np.array([agg_ages[b] for b in sorted(census_age_brackets.keys())])
    agg_df = pd.DataFrame.from_dict(agg_data)
    print(agg_df)
    agg_path = os.path.join(dir_path, 'Malawi_national_ages_16.csv')
    agg_df.to_csv(agg_path, index=False)


def process_labor_tables():
    """Function to process labor tables for employment rates."""
    file_path = os.path.join(dir_path, 'Series D. Economic Tables.xlsx')
    df = pd.read_excel(file_path, sheet_name='D1', header=1, skiprows=[2, 3])

    age_labels_binned = df['Age'][1:]
    age_count_binned = np.array(df['Population'][1:])
    employed_ages_binned = np.array(df['Economically Active (Labour Force)'][1:])
    employment_rates_binned = employed_ages_binned / age_count_binned
    employment_rates = dict.fromkeys(np.arange(101), 0.)
    for bi, bl in enumerate(age_labels_binned):
        b = bl.split(' - ')
        b0, b1 = int(b[0]), int(b[1])

        for a in np.arange(b0, b1 + 1):
            employment_rates[a] = employment_rates_binned[bi]
            print(a, employment_rates[a])
    data = dict(age=np.arange(101), percent=np.array([employment_rates[a] for a in range(101)]))
    new_df = pd.DataFrame.from_dict(data)
    new_file_path = os.path.join(dir_path, 'Malawi_employment_rates_by_age.csv')
    new_df.to_csv(new_file_path, index=False)


def process_education_tables():
    """Function to process education tables for enrollment rates."""
    file_path = os.path.join(dir_path, 'Series C. Education Tables.xlsx')
    df = pd.read_excel(file_path, sheet_name='TABLE C2', header=1, skiprows=[2, 3], skipfooter=24)

    age_labels_binned = np.array(df['Age and Sex (5 Years and Older)'])
    age_count_binned = np.array(df['Total '])
    enrolled_ages_binned = np.array(df['Unnamed: 5'])
    enrollment_rates_binned = enrolled_ages_binned / age_count_binned
    enrollment_rates = dict.fromkeys(np.arange(101), 0.)
    for bi, bl in enumerate(age_labels_binned):
        try:
            b = bl.split(' - ')
            b0, b1 = int(b[0]), int(b[1])
        except:
            b = bl.split('+')
            b0 = int(b[0])
            b1 = 64

        for a in np.arange(b0, b1 + 1):
            enrollment_rates[a] = enrollment_rates_binned[bi]
            print(a, enrollment_rates[a])
    data = dict(age=np.arange(101), percent=np.array([enrollment_rates[a] for a in range(101)]))
    new_df = pd.DataFrame.from_dict(data)
    new_file_path = os.path.join(dir_path, 'Malawi_enrollment_rates_by_age.csv')
    new_df.to_csv(new_file_path, index=False)


if __name__ == '__main__':
    print(f"processing files from {dir_path}.")

    # # turn the following on as needed to process raw data files
    # process_age_tables()
    # process_labor_tables()
    # process_education_tables()

