import synthpops as sp
import numpy as np


if __name__ == '__main__':

    np.random.seed(0)
    sp.set_seed(0)

    m1 = np.zeros((3, 5))
    m2 = np.zeros((3, 5))

    d = sp.get_canberra_distance(m1, m2)

    datadir = sp.settings.datadir

    sheet_name1 = 'United States of America'
    sheet_name2 = 'United Kingdom of Great Britain'

    cm1 = sp.get_contact_matrices(datadir, sheet_name=sheet_name1)
    cm2 = sp.get_contact_matrices(datadir, sheet_name=sheet_name2)

    nm1 = {k: v / v.sum() for k, v in cm1.items()}
    nm2 = {k: v / v.sum() for k, v in cm2.items()}

    # for k in cm1:
    #     print(k, cm1[k].sum(), cm2[k].sum())

    # print(sp.get_canberra_distance(cm1['H'], cm1['S']))
    # print(sp.get_canberra_distance(cm1['H'], cm2['H']))
    # print(sp.get_canberra_distance(cm1['S'], cm2['S']))

    # print()

    # print(sp.get_canberra_distance(nm1['H'], nm1['S']))
    # print(sp.get_canberra_distance(nm1['H'], nm2['H']))
    # print(sp.get_canberra_distance(nm1['S'], nm2['S']))

    print(sp.get_canberra_distance(nm1['S'], nm2['S'], dim_norm=1))

    r, c = nm1['H'].shape
    z = 1e-19
    zm1 = np.random.rand(r, c) * z
    zm2 = np.random.rand(r, c) * z

    print()
    print(sp.get_canberra_distance(nm1['S'] + zm1, nm1['S'] + zm2, dim_norm=1))
    print(sp.get_canberra_distance(nm1['H'] + zm1, nm1['H'] + zm2, dim_norm=1))
    print(sp.get_canberra_distance(nm2['S'] + zm1, nm2['S'] + zm2, dim_norm=1))
    print(sp.get_canberra_distance(nm2['H'] + zm1, nm2['H'] + zm2, dim_norm=1))

    # r = 8
    # c = 5
    # z = 0.01

    # m1 = np.ones((r, c)) + np.random.rand(r, c) * z
    # m2 = np.ones((r, c)) + np.random.rand(r, c) * z

    # print()

    # print(sp.get_canberra_distance(m1, m2))
    # print(sp.get_canberra_distance(m1 / m1.sum(), m2 / m2.sum()))
    # print(sp.get_canberra_distance(m1, m2, dim_norm=1))
    # print(sp.get_canberra_distance(m1 / m1.sum(), m2 / m2.sum(), dim_norm=1))

