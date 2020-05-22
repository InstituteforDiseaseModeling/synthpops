import sciris as sc
import numpy as np
import networkx as nx
# from .base import *
# from . import data_distributions as spdata
# from . import sampling as spsamp
# from .config import datadir
import os
import pandas as pd

import synthpops as sp

seed = 0
np.random.seed(seed)


if __name__ == '__main__':

    datadir = sp.datadir
    country_location = 'usa'
    state_location = 'Washington'
    location = 'seattle_metro'
    sheet_name = 'United States of America'

    with_facilities = True
    with_industry_code = False

    n = 20e3
    n = int(n)

    options_args = {'use_microstructure': True, 'use_industry_code': with_industry_code, 'use_long_term_care_facilities': with_facilities}
    network_distr_args = {'Npop': int(n)}


    # population = sp.make_contacts(location=location, state_location=state_location, country_location=country_location, options_args=options_args, network_distr_args=network_distr_args)

    # # sp.show_layers(population, show_n=20)
    # p = 1900
    # for i in range(p):
    #     person = population[i]

    #     if person['snf_res'] is not None or person['snf_staff'] is not None:
    #         print(i, person['age'],person['snf_res'], person['snf_staff'])

    #         

    file_path = os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, 'contact_networks_facilities')

    age_by_uid_path = os.path.join(file_path, location + '_' + str(n) + '_age_by_uid.dat')

    facilities_by_uid_path = os.path.join(file_path, location + '_' + str(n) + '_synthetic_facilities_with_uids.dat')
    facilities_staff_by_uid_path = os.path.join(file_path, location + '_' + str(n) + '_synthetic_facilities_staff_with_uids.dat')

    df = pd.read_csv(age_by_uid_path, delimiter=' ', header=None)

    age_by_uid_dic = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))

    facilities_by_uids = open(facilities_by_uid_path, 'r')
    facilities_staff_uids = open(facilities_staff_by_uid_path, 'r')

    popdict = {}

    p_matrix = np.zeros((2,2))
    p_matrix[0,0] = 0.1
    p_matrix[0,1] = 0.1
    p_matrix[1,0] = 0.1
    p_matrix[1,1] = 0.5

    groups = []
    groups.append([60,6])
    groups.append([60,10])
    groups.append([60,20])

    k = 15

    # share_k_matrix = np.ones((2,2))
    # share_k_matrix[0][0] = 0.25
    # share_k_matrix[1][1] = 0.25
    # share_k_matrix[0][1] = 0.25
    # share_k_matrix[1][0] = 0.25


    for nf, (line1, line2) in enumerate(zip(facilities_by_uids, facilities_staff_uids)):
        r1 = line1.strip().split(' ')
        r2 = line2.strip().split(' ')
        r1 = [int(i) for i in r1]
        r2 = [int(i) for i in r2]

        n1 = list(np.arange(len(r1)).astype(int))
        n2 = list(np.arange(len(r1), len(r1) + len(r2)).astype(int))
        # print(len(r1), len(r2))

        facility = r1 + r2

        sizes = [len(r1), len(r2)]

        for i in r1:
            popdict[i] = {'contacts': {}}
            for l in ['H', 'S', 'W', 'C', 'LTCF']:
                popdict[i]['contacts'][l] = set()

        for i in r2:
            popdict[i] = {'contacts': {}}
            for l in ['H', 'S', 'W', 'C', 'LTCF']:
                popdict[i]['contacts'][l] = set()

        sp.create_reduced_contacts_with_group_types(popdict, r1, r2, 'LTCF', average_degree=k, force_cross_edges=True)

    # # for c,gr in enumerate(groups):

    #     # n1 = list(np.arange(gr[0]).astype(int))
    #     # n2 = list(np.arange(gr[0],gr[0] + gr[1]).astype(int))
    #     # sizes = gr

    #     # p_matrix[0][0] = 
        share_k_matrix = np.ones((2,2))
        share_k_matrix *= k/np.sum(sizes)

    #     # share_k_matrix *= k/np.sum(gr)
    #     # share_k_matrix[0][1] *= 2
    #     # share_k_matrix[1][0] *= 2
    #     # share_k_matrix[0][0] *= 0.5
    #     # share_k_matrix[1][1] *= 0.5

        p_matrix = share_k_matrix.copy()

        G = nx.stochastic_block_model(sizes,p_matrix, selfloops=False)

        E = G.edges()
        print(len(E),len(G))

    #     print(np.mean([G.degree(i) for i in G.nodes()]))
    #     # print(np.mean([i for i in G.degree()]))
    #     # print([i for i in G.degree()])
    #     r1_edges = []
    #     r2_edges = []
    #     cross_edges = []
    #     for e in E:
    #         i, j = e
    #         if i in n1 and j in n1:
    #             r1_edges.append(e)
    #         elif i in n2 and j in n2:
    #             r2_edges.append(e)
    #         else:
    #             cross_edges.append(e)

    #         # print(facility[i], facility[j])
    #         # popdict[facility[i]]['contacts']['LTCF'].add(facility[j])
    #         # popdict[facility[j]]['contacts']['LTCF'].add(facility[i])

    #     # print(n1, n2)

    #     print( 'in res',2 * len(r1_edges) / sizes[0])
    #     print( 'in staff',2 * len(r2_edges) / sizes[1])
    #     # print( 2 * len(cross_edges) / (len(r1)+len(r2) ))
    #     print( 'staff per res',len(cross_edges) / sizes[0])
    #     print( 'res per staff',len(cross_edges) / sizes[1])

        # group_1_to_1_only = []
        # group_2_to_2_only = []

        # for i in n1:
        #     cross_neighbors = [j for j in G.neighbors(i) if j in n2]
        #     if len(cross_neighbors) == 0:
        #         group_1_to_1_only.append(i)

        # for i in n2:
        #     cross_neighbors = [j for j in G.neighbors(i) if j in n1]
        #     if len(cross_neighbors) == 0:
        #         group_2_to_2_only.append(i)

        # print('those with only within group contacts')
        # print(len(n1), len(n2))
        # print(len(group_1_to_1_only), len(group_2_to_2_only))

        # for i in n1:
        #     neighbors = [j for j in G.neighbors(i)]
        #     # staff_neighbors = set(neighbors).intersection(set(n2))
        #     staff_neighbors = [j for j in G.neighbors(i) if j in n2]

        #     if len(staff_neighbors) == 0:
        #         random_neighbor = np.random.choice(neighbors)

        #         random_staff = np.random.choice(n2)
        #         print('rs',random_staff)

        #         random_staff_neighbors = [ii for ii in G.neighbors(random_staff) if ii in n2]
        #         random_staff_neighbor_cut = np.random.choice(random_staff_neighbors)
        #         print('rsc',random_staff_neighbor_cut)

        #         G.add_edge(i, random_staff)
        #         G.remove_edge(random_staff, random_staff_neighbor_cut)

        # print('facility', nf, len(r1), len(r2))
        
        # for i in n1:
        #     # neighbors = [j for j in popdict[i]['contacts']['LTCF'] if j in r2]
        #     neighbors = [j for j in G.neighbors(i) if j in n2]
        #     if len(neighbors) == 0:
        #         print('still no staff', i, len(popdict[i]['contacts']['LTCF']))


    #     E = G.edges()
    #     for e in E:
    #         i, j = e

    #         popdict[facility[i]]['contacts']['LTCF'].add(facility[j])
    #         popdict[facility[j]]['contacts']['LTCF'].add(facility[i])


    #     print()

        for i in r1:
            contacts = popdict[i]['contacts']['LTCF']

            cross_neighbors = set(r2).intersection(set(contacts))
            if len(cross_neighbors) == 0:
                print('god no', contacts)



    # for i in popdict:
    #     print(i)


