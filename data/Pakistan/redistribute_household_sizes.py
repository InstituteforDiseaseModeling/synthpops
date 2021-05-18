import numpy as np
import pandas as pd

import os

# user name
username = os.path.split(os.path.expanduser('~'))[-1]

# directory where code lives
codedir = os.path.dirname(os.path.abspath(__file__))

# directory where data lives
datadir = codedir.replace('Code', 'Data')


def get_household_size_distr():
    file_path = os.path.join(datadir, 'Pakistan_household_size_distribution_original.dat')
    df = pd.read_csv(file_path)
    return dict(zip(df.household_size, df.percent))


def redistribute_last_size(max_size):
    size_distr = get_household_size_distr()
    original_max_size = max(size_distr.keys())

    size_distr[original_max_size] = size_distr[original_max_size] / (max_size + 1 - original_max_size)
    for s in range(original_max_size + 1, max_size + 1):
        size_distr[s] = size_distr[original_max_size]

    return size_distr


def get_average_household_size(max_size):
    size_distr = redistribute_last_size(max_size)
    s = [size_distr[s] * s for s in size_distr]
    return np.sum(s)


def write_redistributed_household_size_distr(max_size):
    size_distr = redistribute_last_size(max_size)
    file_path = os.path.join(datadir, 'Pakistan_household_size_distribution_extended.dat')
    f = open(file_path, 'w')
    f.write('household_size,percent\n')
    for s in sorted(size_distr.keys()):
        f.write(str(s) + ',' + str(size_distr[s]) + '\n')
    f.close()


if __name__ == '__main__':

    avg_s = get_average_household_size(14)
    print(avg_s)

    write_redistributed_household_size_distr(14)
