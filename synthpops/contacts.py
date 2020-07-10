"""
Generate contacts between people in the population, with many options possible
"""

import os
import numpy as np
import pandas as pd
import numba as nb
import sciris as sc
import networkx as nx
from . import data_distributions as spdata
from . import sampling as spsamp
from . import base as spb
from . import contact_networks as spcnx
from . import read_write as sprw
from .config import datadir


def make_popdict(n=None, uids=None, ages=None, sexes=None, location=None, state_location=None, country_location=None, use_demography=False, id_len=16):
    """
    Create a dictionary of n people with age, sex and loc keys

    Args:
        n (int)                   : number of people
        uids (list)               : supplied uids of individuals
        ages (list)               : supplied ages of individuals
        sexes (list)              : supplied sexes of individuals
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in
        use_demography (bool)     : If True, use demographic data
        id_len (int)              : length of the uid

    Returns:
        A dictionary where keys are the uid of each person and the values are another dictionary containing values for other attributes of the person
    """

    min_people = 100

    if location             is None: location = 'seattle_metro'
    if state_location is None: state_location = 'Washington'

    # A list of UIDs was supplied as the first argument
    if uids is not None:  # UIDs were supplied, use them
        n = len(uids)
        # uid_mapping = {uids[i]: i for i in range(len(uids))}
        try:
            uid_mapping = {uid: int(uid) for u, uid in enumerate(uids)}
        except:
            uid_mapping = {uid: u for u, uid in enumerate(uids)}  # replacing uids for uid_mapping since uids might be strings
    else:  # Not supplied, generate
        n = int(n)
        # default to using ints for ids from now on
        uids = [i for i in range(n)] 
        uid_mapping = {i: i for i in range(n)}

        # using strings for uids
        # uids = []
        # for i in range(n):
            # uids.append(sc.uuid(length=id_len))

    # Check that there are enough people
    if n < min_people:
        print(f'Warning: with {n}<{min_people} people, contact matrices will be approximate')

    # Optionally take in either ages or sexes, too
    if ages is None and sexes is None:
        if use_demography:
            if country_location != 'usa':
                gen_ages = spsamp.get_age_n(datadir, n=n, location=location, state_location=state_location, country_location=country_location)
                gen_sexes = list(np.random.binomial(1, p=0.5, size=n))
            else:
                if location is None: location, state_location = 'seattle_metro', 'Washington'
                gen_ages, gen_sexes = spsamp.get_usa_age_sex_n(datadir, location=location, state_location=state_location, country_location=country_location, n_people=n)
        else:
            # if location is None:
            gen_ages, gen_sexes = spsamp.get_age_sex_n(None, None, None, n_people=n)

    # you only have ages...
    elif ages is not None and sexes is None:
        if country_location == 'usa':
            if location is None: location, state_location = 'seattle_metro', 'Washington'
            gen_ages, gen_sexes = spsamp.get_usa_sex_n(datadir, ages, location=location, state_location=state_location, country_location=country_location)
        else:
            gen_ages = ages
            gen_sexes = list(np.random.binomial(1, p=0.5, size=n))

    # you only have sexes...
    elif ages is None and sexes is not None:
        if country_location == 'usa':
            if location is None: location, state_location = 'seattle_metro', 'Washington'
            gen_ages, gen_sexes = spsamp.get_usa_age_n(datadir, sexes, location=location, state_location=state_location, country_location=country_location)
        else:
            # gen_sexes = sexes
            # gen_ages = sp.get_age_n(datadir,n=n,location=location,state_location=state_location,country_location=country_location)
            raise NotImplementedError('Currently, only locations in the US are supported')

    # randomize your generated ages and sexes
    if ages is None or sexes is None:
        random_inds = np.random.permutation(n)
        ages = [gen_ages[r] for r in random_inds]
        sexes = [gen_sexes[r] for r in random_inds]

    # you have both ages and sexes so we'll just populate that for you...
    popdict = {}
    for i, uid in enumerate(uids):
        uid = uid_mapping[uid]
        popdict[uid] = {}
        popdict[uid]['age'] = int(ages[i])
        popdict[uid]['sex'] = sexes[i]
        popdict[uid]['loc'] = None
        popdict[uid]['contacts'] = {'M': set()}

    return popdict


def make_contacts_generic(popdict, network_distr_args):
    """
    Create contact network regardless of age, according to network distribution properties. Can be used by webapp.

    Args:
        popdict (dict)           : dictionary of all individuals
        network_distr_args (dict): network distribution parameters dictionary for average_degree, network_type, and directionality

    Returns:
        A dictionary of individuals with contacts drawn from given network distribution parameters.
    """

    n_contacts = network_distr_args['average_degree']
    network_type = network_distr_args['network_type']
    directed = network_distr_args['directed']

    uids = popdict.keys()
    uids = [uid for uid in uids]

    # if isinstance(uids[0], str):
        # uid_mapping = {uid: u for u, uid in enumerate(uids)}

    # elif isinstance(uids[0], int):
        # uid_mapping = {i: i for i in range(len(uids))}

    N = len(popdict)

    if network_type == 'poisson_degree':
        p = float(n_contacts)/N

        G = nx.erdos_renyi_graph(N, p, directed=directed)

    A = [a for a in G.adjacency()]

    for n, uid in enumerate(uids):
        targets = [t for t in A[n][1].keys()]
        target_uids = [uids[target] for target in targets]  # if using uids which may be strings or ints
        popdict[uid]['contacts']['M'] = set(target_uids)

    return popdict


def make_contacts_without_social_layers_152(popdict, n_contacts_dic, location, state_location, country_location, sheet_name, network_distr_args):
    """
    Create contact network according to overall age-mixing contact matrices. Does not capture clustering or microstructure,
    therefore exact households, schools, or workplaces are not created. However, this does separate agents according to their
    age and gives them contacts likely for their age. For all ages, the average number of contacts is constant with this
    method, although location specific data may prove this to not be true.

    Args:
        popdict (dict)            : dictionary of all individuals
        n_contacts_dic (dict)     : number of contacts to draw on average by setting
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in
        sheet_name (string)       : name of the sheet in the excel file with contact patterns
        network_distr_args (dict) : network distribution parameters dictionary for average_degree, network_type, and directionality, can also include powerlaw exponents,
                                    block sizes (re: SBMs), clustering distribution, or other properties needed to generate network structures. Checkout
                                    https://networkx.github.io/documentation/stable/reference/generators.html#module-networkx.generators for what's possible
                                    Default 'network_type' is 'poisson_degree' for Erdos-Renyi random graphs in large n limit.

    Returns:
        A dictionary of individuals with attributes, including their age and the ids of their contacts drawn from given network distribution parameters and the ages of contacts drawn according to overall age mixing data.
        A single social setting or layer of contacts.

    """

    uids_by_age_dic = spb.get_uids_by_age_dic(popdict)
    age_brackets = spdata.get_census_age_brackets(datadir, state_location=state_location, country_location=country_location)
    num_agebrackets = len(age_brackets)
    age_by_brackets_dic = spb.get_age_by_brackets_dic(age_brackets)

    age_mixing_matrix_dic = spdata.get_contact_matrix_dic(datadir, sheet_name=sheet_name)
    age_mixing_matrix_dic['M'] = spb.combine_matrices(age_mixing_matrix_dic, n_contacts_dic, num_agebrackets)  # may need to normalize matrices before applying this method to K. Prem et al matrices because of the difference in what the raw matrices represent

    n_contacts = network_distr_args['average_degree']
    directed = network_distr_args['directed']
    network_type = network_distr_args['network_type']

    k = 'M'
    if directed:
        if network_type == 'poisson_degree':
            for i in popdict:
                nc = spsamp.pt(n_contacts)
                contact_ages = spsamp.sample_n_contact_ages_with_matrix(nc, popdict[i]['age'], age_brackets, age_by_brackets_dic, age_mixing_matrix_dic[k])
                popdict[i]['contacts'][k] = popdict[i]['contacts'][k].union(spsamp.get_n_contact_ids_by_age(uids_by_age_dic, contact_ages, age_brackets, age_by_brackets_dic))
    else:
        if network_type == 'poisson_degree':
            n_contacts = n_contacts/2
            for i in popdict:
                nc = spsamp.pt(n_contacts)
                contact_ages = spsamp.sample_n_contact_ages_with_matrix(nc, popdict[i]['age'], age_brackets, age_by_brackets_dic, age_mixing_matrix_dic[k])
                popdict[i]['contacts'][k] = popdict[i]['contacts'][k].union(spsamp.get_n_contact_ids_by_age(uids_by_age_dic, contact_ages, age_brackets, age_by_brackets_dic))
                for c in popdict[i]['contacts'][k]:
                    popdict[c]['contacts'][k].add(i)

    return popdict


def make_contacts_with_social_layers_152(popdict, n_contacts_dic, location, state_location, country_location, sheet_name, activity_args, network_distr_args):
    """
    Create contact network according to overall age-mixing contact matrices. Does not capture clustering or microstructure,
    therefore exact households, schools, or workplaces are not created. However, this does separate agents according to their
    age and gives them contacts likely for their age specified by the social settings they are likely to participate in.
    For all ages, the average number of contacts is constant with this method, although location specific data very wel may
    prove this to not be true. In general, college students may also be workers, however here they are only students and we
    assume that any of their contacts in the work environment are likely to look like their contacts at school.

    Essentially recreates an age-specific compartmental model's concept of contacts but for an agent based modeling framework.

    Args:
        popdict (dict)            : dictionary of all individuals
        n_contacts_dic (dict)     : number of contacts to draw on average by setting
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in
        sheet_name (string)       : name of the sheet in the excel file with contact patterns
        activity_args (dict)      : dictionary of age bounds for participating in different activities like going to school or working, also student-teacher ratio
        network_distr_args (dict) : network distribution parameters dictionary for average_degree, network_type, and directionality, can also include powerlaw exponents,
                                    block sizes (re: SBMs), clustering distribution, or other properties needed to generate network structures. Checkout
                                    https://networkx.github.io/documentation/stable/reference/generators.html#module-networkx.generators for what's possible
                                    Default 'network_type' is 'poisson_degree' for Erdos-Renyi random graphs in large n limit.

    Returns:
        A dictionary of individuals with contacts with attributes, including their age and the ids of their contacts drawn from given network distribution parameters and the ages of contacts drawn according to age mixing data.
        Multiple social settings or layers so contacts are listed for different layers.

    """

    uids_by_age_dic = spb.get_uids_by_age_dic(popdict)
    age_brackets = spdata.get_census_age_brackets(datadir, state_location=state_location, country_location=country_location)
    num_agebrackets = len(age_brackets)
    age_by_brackets_dic = spb.get_age_by_brackets_dic(age_brackets)

    age_mixing_matrix_dic = spdata.get_contact_matrix_dic(datadir, sheet_name=sheet_name)
    age_mixing_matrix_dic['M'] = spb.combine_matrices(age_mixing_matrix_dic, n_contacts_dic, num_agebrackets)  # may need to normalize matrices before applying this method to K. Prem et al matrices because of the difference in what the raw matrices represent

    directed = network_distr_args['directed']
    network_type = network_distr_args['network_type']

    # currently not set to capture school enrollment rates or work enrollment rates
    student_n_dic = sc.dcp(n_contacts_dic)
    non_student_n_dic = sc.dcp(n_contacts_dic)

    # this might not be needed because students will choose their teachers, but if directed then this makes teachers point to students as well
    n_students = np.sum([len(uids_by_age_dic[a]) for a in range(activity_args['student_age_min'], activity_args['student_age_max']+1)])
    n_workers = np.sum([len(uids_by_age_dic[a]) for a in range(activity_args['worker_age_min'], activity_args['worker_age_max']+1)])
    n_teachers = n_students/activity_args['student_teacher_ratio']
    teachers_school_weight = n_teachers/n_workers

    student_n_dic['W'] = 0
    non_student_n_dic['S'] = teachers_school_weight  # make some teachers
    # 5 categories as defined by activity_args:
    # infants & toddlers : H, C
    # school-aged students : H, S, C
    # college-aged students / workers : H, S, W, C
    # non-student workers : H, W, C
    # retired elderly : H, C - some may be workers too but it's low. directed means they won't have contacts in the workplace, but undirected means they will at least a little.

    # will follow degree distribution well
    for uid in popdict:
        for k in n_contacts_dic:
            popdict[uid]['contacts'][k] = set()

    if directed:

        for uid in popdict:
            age = popdict[uid]['age']
            if age < activity_args['student_age_min']:
                for k in ['H', 'C']:
                    if network_type == 'poisson_degree':
                        nc = spsamp.pt(n_contacts_dic[k])
                        contact_ages = spsamp.sample_n_contact_ages_with_matrix(nc, age, age_brackets, age_by_brackets_dic, age_mixing_matrix_dic[k])
                        popdict[uid]['contacts'][k] = popdict[uid]['contacts'][k].union(spsamp.get_n_contact_ids_by_age(uids_by_age_dic, contact_ages, age_brackets, age_by_brackets_dic))

            elif age >= activity_args['student_age_min'] and age < activity_args['student_age_max']:
                for k in ['H', 'S', 'C']:
                    if network_type == 'poisson_degree':
                        nc = spsamp.pt(n_contacts_dic[k])
                        contact_ages = spsamp.sample_n_contact_ages_with_matrix(nc, age, age_brackets, age_by_brackets_dic, age_mixing_matrix_dic[k])
                        popdict[uid]['contacts'][k] = popdict[uid]['contacts'][k].union(spsamp.get_n_contact_ids_by_age(uids_by_age_dic, contact_ages, age_brackets, age_by_brackets_dic))

            elif age >= activity_args['college_age_min'] and age < activity_args['college_age_max']:
                for k in ['H', 'S', 'C']:
                    if network_type == 'poisson_degree':
                        nc = spsamp.pt(n_contacts_dic[k])
                        contact_ages = spsamp.sample_n_contact_ages_with_matrix(nc, age, age_brackets, age_by_brackets_dic, age_mixing_matrix_dic[k])
                        popdict[uid]['contacts'][k] = popdict[uid]['contacts'][k].union(spsamp.get_n_contact_ids_by_age(uids_by_age_dic, contact_ages, age_brackets, age_by_brackets_dic))

            elif age >= activity_args['worker_age_min'] and age < activity_args['worker_age_max']:
                for k in ['H', 'S', 'W', 'C']:
                    if network_type == 'poisson_degree':
                        nc = spsamp.pt(non_student_n_dic[k])
                        contact_ages = spsamp.sample_n_contact_ages_with_matrix(nc, age, age_brackets, age_by_brackets_dic, age_mixing_matrix_dic[k])
                        popdict[uid]['contacts'][k] = popdict[uid]['contacts'][k].union(spsamp.get_n_contact_ids_by_age(uids_by_age_dic, contact_ages, age_brackets, age_by_brackets_dic))

            elif age >= activity_args['worker_age_max']:
                for k in ['H', 'C']:
                    if network_type == 'poisson_degree':
                        nc = spsamp.pt(n_contacts_dic[k])
                        contact_ages = spsamp.sample_n_contact_ages_with_matrix(nc, age, age_brackets, age_by_brackets_dic, age_mixing_matrix_dic[k])
                        popdict[uid]['contacts'][k] = popdict[uid]['contacts'][k].union(spsamp.get_n_contact_ids_by_age(uids_by_age_dic, contact_ages, age_brackets, age_by_brackets_dic))

    else:
        for uid in popdict:
            age = popdict[uid]['age']
            if age < activity_args['student_age_min']:
                for k in ['H', 'C']:
                    if network_type == 'poisson_degree':
                        nc = spsamp.pt(n_contacts_dic[k]/2)
                        contact_ages = spsamp.sample_n_contact_ages_with_matrix(nc, age, age_brackets, age_by_brackets_dic, age_mixing_matrix_dic[k])
                        popdict[uid]['contacts'][k] = popdict[uid]['contacts'][k].union(spsamp.get_n_contact_ids_by_age(uids_by_age_dic, contact_ages, age_brackets, age_by_brackets_dic))
                        for c in popdict[uid]['contacts'][k]:
                            popdict[c]['contacts'][k].add(uid)

            elif age >= activity_args['student_age_min'] and age < activity_args['student_age_max']:
                for k in ['H', 'S', 'C']:
                    if network_type == 'poisson_degree':
                        nc = spsamp.pt(n_contacts_dic[k]/2)
                        contact_ages = spsamp.sample_n_contact_ages_with_matrix(nc, age, age_brackets, age_by_brackets_dic, age_mixing_matrix_dic[k])
                        popdict[uid]['contacts'][k] = popdict[uid]['contacts'][k].union(spsamp.get_n_contact_ids_by_age(uids_by_age_dic, contact_ages, age_brackets, age_by_brackets_dic))
                        for c in popdict[uid]['contacts'][k]:
                            popdict[c]['contacts'][k].add(uid)

            elif age >= activity_args['college_age_min'] and age < activity_args['college_age_max']:
                for k in ['H', 'S', 'C']:
                    if network_type == 'poisson_degree':
                        nc = spsamp.pt(n_contacts_dic[k]/2)
                        contact_ages = spsamp.sample_n_contact_ages_with_matrix(nc, age, age_brackets, age_by_brackets_dic, age_mixing_matrix_dic[k])
                        popdict[uid]['contacts'][k] = popdict[uid]['contacts'][k].union(spsamp.get_n_contact_ids_by_age(uids_by_age_dic, contact_ages, age_brackets, age_by_brackets_dic))
                        for c in popdict[uid]['contacts'][k]:
                            popdict[c]['contacts'][k].add(uid)

            elif age >= activity_args['worker_age_min'] and age < activity_args['worker_age_max']:
                for k in ['H', 'W', 'C']:
                    if network_type == 'poisson_degree':
                        nc = spsamp.pt(non_student_n_dic[k]/2)
                        contact_ages = spsamp.sample_n_contact_ages_with_matrix(nc, age, age_brackets, age_by_brackets_dic, age_mixing_matrix_dic[k])
                        popdict[uid]['contacts'][k] = popdict[uid]['contacts'][k].union(spsamp.get_n_contact_ids_by_age(uids_by_age_dic, contact_ages, age_brackets, age_by_brackets_dic))
                        for c in popdict[uid]['contacts'][k]:
                            popdict[c]['contacts'][k].add(uid)

            elif age >= activity_args['worker_age_max']:
                for k in ['H', 'C']:
                    if network_type == 'poisson_degree':
                        nc = spsamp.pt(n_contacts_dic[k]/2)
                        contact_ages = spsamp.sample_n_contact_ages_with_matrix(nc, age, age_brackets, age_by_brackets_dic, age_mixing_matrix_dic[k])
                        popdict[uid]['contacts'][k] = popdict[uid]['contacts'][k].union(spsamp.get_n_contact_ids_by_age(uids_by_age_dic, contact_ages, age_brackets, age_by_brackets_dic))
                        for c in popdict[uid]['contacts'][k]:
                            popdict[c]['contacts'][k].add(uid)

    return popdict


def make_contacts_without_social_layers_and_sex(popdict, n_contacts_dic, location, state_location, country_location, sheet_name, network_distr_args):
    """
    Create contact network according to overall age-mixing contact matrices for the US. Does not capture clustering or microstructure, therefore
    exact households, schools, or workplaces are not created. However, this does separate agents according to their age and gives them contacts
    likely for their age. For all ages, the average number of contacts is constant, although location specific data may prove this to not be true.
    Individuals also have a sex, though this in general does not have an impact on their contact patterns.

    Args:
        popdict (dict)            : dictionary of all individuals
        n_contacts_dic (dict)     : number of contacts to draw on average by setting
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in
        sheet_name (string)       : name of the sheet in the excel file with contact patterns
        network_distr_args (dict) : network distribution parameters dictionary for average_degree, network_type, and directionality, can also include powerlaw exponents,
                                    block sizes (re: SBMs), clustering distribution, or other properties needed to generate network structures. Checkout
                                    https://networkx.github.io/documentation/stable/reference/generators.html#module-networkx.generators for what's possible
                                    Default 'network_type' is 'poisson_degree' for Erdos-Renyi random graphs in large n limit.

    Returns:
        A dictionary of individuals with attributes, including their age and the ids of their contacts drawn from given network distribution parameters and the ages of contacts drawn according to overall age mixing data.
        A single social setting or layer of contacts.

    """

    # using a single contact matrix combined from the other settings available
    uids_by_age_dic = spb.get_uids_by_age_dic(popdict)
    age_brackets = spdata.get_census_age_brackets(datadir, state_location=state_location, country_location=country_location)
    age_by_brackets_dic = spb.get_age_by_brackets_dic(age_brackets)
    num_agebrackets = len(age_brackets)

    age_mixing_matrix_dic = spdata.get_contact_matrix_dic(datadir, sheet_name)
    age_mixing_matrix_dic['M'] = spb.combine_matrices(age_mixing_matrix_dic, n_contacts_dic, num_agebrackets)  # may need to normalize matrices before applying this method to K. Prem et al matrices because of the difference in what the raw matrices represent

    n_contacts = network_distr_args['average_degree']
    directed = network_distr_args['directed']
    network_type = network_distr_args['network_type']

    k = 'M'

    if directed:
        if network_type == 'poisson_degree':
            for i in popdict:
                nc = spsamp.pt(n_contacts)
                contact_ages = spsamp.sample_n_contact_ages_with_matrix(nc, popdict[i]['age'], age_brackets, age_by_brackets_dic, age_mixing_matrix_dic[k])
                popdict[i]['contacts'][k] = popdict[i]['contacts'][k].union(spsamp.get_n_contact_ids_by_age(uids_by_age_dic, contact_ages, age_brackets, age_by_brackets_dic))

    else:
        if network_type == 'poisson_degree':
            n_contacts = n_contacts/2
            for i in popdict:
                nc = spsamp.pt(n_contacts)
                contact_ages = spsamp.sample_n_contact_ages_with_matrix(nc, popdict[i]['age'], age_brackets, age_by_brackets_dic, age_mixing_matrix_dic[k])
                popdict[i]['contacts'][k] = popdict[i]['contacts'][k].union(spsamp.get_n_contact_ids_by_age(uids_by_age_dic, contact_ages, age_brackets, age_by_brackets_dic))
                for c in popdict[i]['contacts'][k]:
                    popdict[c]['contacts'][k].add(i)

    return popdict


def make_contacts_with_social_layers_and_sex(popdict, n_contacts_dic, location, state_location, country_location, sheet_name, activity_args, network_distr_args):
    """
    Create contact network according to overall age-mixing contact matrices for the US.
    Does not capture clustering or microstructure, therefore exact households, schools, or workplaces are not created.
    However, this does separate agents according to their age and gives them contacts likely for their age specified by
    the social settings they are likely to participate in. College students may also be workers, however here they are
    only students and we assume that any of their contacts in the work environment are likely to look like their contacts at school.

    Args:
        popdict (dict)            : dictionary of all individuals
        n_contacts_dic (dict)     : number of contacts to draw on average by setting
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in
        sheet_name (string)       : name of the sheet in the excel file with contact patterns
        activity_args (dict)      : dictionary of age bounds for participating in different activities like going to school or working, also student-teacher ratio
        network_distr_args (dict) : network distribution parameters dictionary for average_degree, network_type, and directionality, can also include powerlaw exponents,
                                    block sizes (re: SBMs), clustering distribution, or other properties needed to generate network structures. Checkout
                                    https://networkx.github.io/documentation/stable/reference/generators.html#module-networkx.generators for what's possible
                                    Default 'network_type' is 'poisson_degree' for Erdos-Renyi random graphs in large n limit.
    Returns:
        A dictionary of individuals with attributes, including their age and the ids of their contacts drawn from given network distribution parameters and the ages of contacts drawn according to age mixing data.
        Multiple social settings or layers so contacts are listed for different layers.

    """

    # use a contact matrix dictionary and n_contacts_dic for the average number of contacts in each layer
    uids_by_age_dic = spb.get_uids_by_age_dic(popdict)

    age_brackets = spdata.get_census_age_brackets(datadir, state_location=state_location, country_location=country_location)
    age_by_brackets_dic = spb.get_age_by_brackets_dic(age_brackets)

    age_mixing_matrix_dic = spdata.get_contact_matrix_dic(datadir, sheet_name)

    directed = network_distr_args['directed']
    network_type = network_distr_args['network_type']

    student_n_dic = sc.dcp(n_contacts_dic)
    non_student_n_dic = sc.dcp(n_contacts_dic)

    # this might not be needed because students will choose their teachers, but if directed then this makes teachers point to students as well
    n_students = np.sum([len(uids_by_age_dic[a]) for a in range(activity_args['student_age_min'], activity_args['student_age_max']+1)])
    n_workers = np.sum([len(uids_by_age_dic[a]) for a in range(activity_args['worker_age_min'], activity_args['worker_age_max']+1)])
    n_teachers = n_students/activity_args['student_teacher_ratio']
    teachers_school_weight = n_teachers/n_workers

    student_n_dic['W'] = 0
    non_student_n_dic['S'] = teachers_school_weight  # make some teachers
    # 5 categories :
    # infants & toddlers : H, C
    # school-aged students : H, S, C
    # college-aged students / workers : H, S, W, C
    # non-student workers : H, W, C
    # retired elderly : H, C - some may be workers too but it's low. directed means they won't have contacts in the workplace, but undirected means they will at least a little.

    # will follow degree distribution well
    for uid in popdict:
        for k in n_contacts_dic:
            popdict[uid]['contacts'][k] = set()

    if directed:

        for uid in popdict:
            age = popdict[uid]['age']
            if age < activity_args['student_age_min']:
                for k in ['H', 'C']:
                    if network_type == 'poisson_degree':
                        nc = spsamp.pt(n_contacts_dic[k])
                        contact_ages = spsamp.sample_n_contact_ages_with_matrix(nc, age, age_brackets, age_by_brackets_dic, age_mixing_matrix_dic[k])
                        popdict[uid]['contacts'][k] = popdict[uid]['contacts'][k].union(spsamp.get_n_contact_ids_by_age(uids_by_age_dic, contact_ages, age_brackets, age_by_brackets_dic))

            elif age >= activity_args['student_age_min'] and age < activity_args['student_age_max']:
                for k in ['H', 'S', 'C']:
                    if network_type == 'poisson_degree':
                        nc = spsamp.pt(n_contacts_dic[k])
                        contact_ages = spsamp.sample_n_contact_ages_with_matrix(nc, age, age_brackets, age_by_brackets_dic, age_mixing_matrix_dic[k])
                        popdict[uid]['contacts'][k] = popdict[uid]['contacts'][k].union(spsamp.get_n_contact_ids_by_age(uids_by_age_dic, contact_ages, age_brackets, age_by_brackets_dic))

            elif age >= activity_args['college_age_min'] and age < activity_args['college_age_max']:
                for k in ['H', 'S', 'C']:
                    if network_type == 'poisson_degree':
                        # people at school and work? how??? college students going to school might actually look like their work environments anyways so for now this is just going to have schools and no work
                        nc = spsamp.pt(n_contacts_dic[k])
                        contact_ages = spsamp.sample_n_contact_ages_with_matrix(nc, age, age_brackets, age_by_brackets_dic, age_mixing_matrix_dic[k])
                        popdict[uid]['contacts'][k] = popdict[uid]['contacts'][k].union(spsamp.get_n_contact_ids_by_age(uids_by_age_dic, contact_ages, age_brackets, age_by_brackets_dic))

            elif age >= activity_args['worker_age_min'] and age < activity_args['worker_age_max']:
                for k in ['H', 'S', 'W', 'C']:
                    if network_type == 'poisson_degree':
                        nc = spsamp.pt(non_student_n_dic[k])
                        contact_ages = spsamp.sample_n_contact_ages_with_matrix(nc, age, age_brackets, age_by_brackets_dic, age_mixing_matrix_dic[k])
                        popdict[uid]['contacts'][k] = popdict[uid]['contacts'][k].union(spsamp.get_n_contact_ids_by_age(uids_by_age_dic, contact_ages, age_brackets, age_by_brackets_dic))

            elif age >= activity_args['worker_age_max']:
                for k in ['H', 'C']:
                    if network_type == 'poisson_degree':
                        nc = spsamp.pt(n_contacts_dic[k])
                        contact_ages = spsamp.sample_n_contact_ages_with_matrix(nc, age, age_brackets, age_by_brackets_dic, age_mixing_matrix_dic[k])
                        popdict[uid]['contacts'][k] = popdict[uid]['contacts'][k].union(spsamp.get_n_contact_ids_by_age(uids_by_age_dic, contact_ages, age_brackets, age_by_brackets_dic))

    else:
        for uid in popdict:
            age = popdict[uid]['age']
            if age < activity_args['student_age_min']:
                for k in ['H', 'C']:
                    if network_type == 'poisson_degree':
                        nc = spsamp.pt(n_contacts_dic[k]/2)
                        contact_ages = spsamp.sample_n_contact_ages_with_matrix(nc, age, age_brackets, age_by_brackets_dic, age_mixing_matrix_dic[k])
                        popdict[uid]['contacts'][k] = popdict[uid]['contacts'][k].union(spsamp.get_n_contact_ids_by_age(uids_by_age_dic, contact_ages, age_brackets, age_by_brackets_dic))
                        for c in popdict[uid]['contacts'][k]:
                            popdict[c]['contacts'][k].add(uid)

            elif age >= activity_args['student_age_min'] and age < activity_args['student_age_max']:
                for k in ['H', 'S', 'C']:
                    if network_type == 'poisson_degree':
                        nc = spsamp.pt(n_contacts_dic[k]/2)
                        contact_ages = spsamp.sample_n_contact_ages_with_matrix(nc, age, age_brackets, age_by_brackets_dic, age_mixing_matrix_dic[k])
                        popdict[uid]['contacts'][k] = popdict[uid]['contacts'][k].union(spsamp.get_n_contact_ids_by_age(uids_by_age_dic, contact_ages, age_brackets, age_by_brackets_dic))
                        for c in popdict[uid]['contacts'][k]:
                            popdict[c]['contacts'][k].add(uid)

            elif age >= activity_args['college_age_min'] and age < activity_args['college_age_max']:
                for k in ['H', 'S', 'C']:
                    if network_type == 'poisson_degree':
                        nc = spsamp.pt(n_contacts_dic[k]/2)
                        contact_ages = spsamp.sample_n_contact_ages_with_matrix(nc, age, age_brackets, age_by_brackets_dic, age_mixing_matrix_dic[k])
                        popdict[uid]['contacts'][k] = popdict[uid]['contacts'][k].union(spsamp.get_n_contact_ids_by_age(uids_by_age_dic, contact_ages, age_brackets, age_by_brackets_dic))
                        for c in popdict[uid]['contacts'][k]:
                            popdict[c]['contacts'][k].add(uid)

            elif age >= activity_args['worker_age_min'] and age < activity_args['worker_age_max']:
                for k in ['H', 'W', 'C']:
                    if network_type == 'poisson_degree':
                        nc = spsamp.pt(non_student_n_dic[k]/2)
                        contact_ages = spsamp.sample_n_contact_ages_with_matrix(nc, age, age_brackets, age_by_brackets_dic, age_mixing_matrix_dic[k])
                        popdict[uid]['contacts'][k] = popdict[uid]['contacts'][k].union(spsamp.get_n_contact_ids_by_age(uids_by_age_dic, contact_ages, age_brackets, age_by_brackets_dic))
                        for c in popdict[uid]['contacts'][k]:
                            popdict[c]['contacts'][k].add(uid)

            elif age >= activity_args['worker_age_max']:
                for k in ['H', 'C']:
                    if network_type == 'poisson_degree':
                        nc = spsamp.pt(n_contacts_dic[k]/2)
                        contact_ages = spsamp.sample_n_contact_ages_with_matrix(nc, age, age_brackets, age_by_brackets_dic, age_mixing_matrix_dic[k])
                        popdict[uid]['contacts'][k] = popdict[uid]['contacts'][k].union(spsamp.get_n_contact_ids_by_age(uids_by_age_dic, contact_ages, age_brackets, age_by_brackets_dic))
                        for c in popdict[uid]['contacts'][k]:
                            popdict[c]['contacts'][k].add(uid)

    return popdict


def rehydrate(data):
    """
    Populate popdict with uids, ages and contacts from generated microstructure data
    that was saved to data object

    Args:
        data (pop object)

    Returns:
        Popdict (sc.objdict)
    """
    popdict = sc.dcp(data['popdict'])
    mapping = {'H': 'households', 'S': 'schools', 'W': 'workplaces'}
    for key, label in mapping.items():
        for r in data[label]:
            for uid in r:
                popdict[uid]['contacts'][key] = set(r)
                popdict[uid]['contacts'][key].remove(uid)

    return popdict


def save_synthpop(datadir, contacts, location):
    """
    Save pop data object to file.

    Args:
        datadir (string)  : file path to the data directory
        contacts (dict)   : dictionary of people with contacts
        location (string) : name of the location

    Returns:
        None
    """

    filename = os.path.join(datadir, location + '_synthpop_' + str(len(contacts)) + '.pop')
    sc.saveobj(filename=filename, obj=contacts)


def create_reduced_contacts_with_group_types(popdict, group_1, group_2, setting, average_degree=20, p_matrix=None, force_cross_edges=True):
    """
    Create contacts between members of group 1 and group 2, fixing the average degree, and the
    probability of an edge between any two groups controlled by p_matrix if provided.
    Forces inter group edge for each individual in group 1 with force_cross_groups equal to True.
    This means not everyone in group 2 will have a contact with group 1.

    The members of group 1 and group 2 should be distinct and non-overlapping.

    Args:
        group_1 (list)            : list of ids for group 1
        group_2 (list)            : list of ids for group 2
        average_degree (int)      : average degree across group 1 and 2
        p_matrix (np.ndarray)     : probability matrix for edges between any two groups
        force_cross_groups (bool) : If True, force each individual to have at least one contact with a member from the other group

    Returns:
        Popdict with edges added for nodes in the two groups.

    Notes:
        This method uses the Stochastic Block Model algorithm to generate contacts both between nodes in different groups
    and for nodes within the same group. In the current version, fixing the average degree and p_matrix, the matrix of probabilities
    for edges between any two groups is not supported. Future versions may add support for this.
    """

    if len(group_1) == 0 or len(group_2) == 0:
        errormsg = f'This method requires that both groups are populated. If one of the two groups has size 0, then consider using the synthpops.trim_contacts() method, or checking that the groups provided to this method are correct.'
        raise ValueError(errormsg)

    if average_degree < 2:
        errormsg = f'This method is likely to create disconnected graphs with average_degree < 2. In order to keep the group connected, use a higher average_degree for nodes across the two groups.'
        raise ValueError(errormsg)

    r1 = [int(i) for i in group_1]
    r2 = [int(i) for i in group_2]

    n1 = list(np.arange(len(r1)).astype(int))
    n2 = list(np.arange(len(r1), len(r1)+len(r2)).astype(int))

    group = r1 + r2
    sizes = [len(r1), len(r2)]

    for i in popdict:
        popdict[i]['contacts'].setdefault(setting, set())

    # group is less than the average degree, so return a fully connected graph instead
    if len(group) <= average_degree:
        G = nx.complete_graph(len(group))

    # group 2 is less than 2 people so everyone in group 1 must be connected to that lone group 2 individual, create a fully connected graph then remove some edges at random to preserve the degree distribution
    elif len(group_2) < 2:
        G = nx.complete_graph(len(group))
        for i in n1:
            group_1_neighbors = [j for j in G.neighbors(i) if j in n1]

            # if the person's degree is too high, cut out some contacts
            if len(group_1_neighbors) > average_degree:
                ncut = len(group_1_neighbors) - average_degree # rough number to cut
                # ncut = spsamp.pt(ncut) # sample from poisson that number
                # ncut = min(len(group_1_neighbors), ncut)  # make sure the number isn't greater than the people available to cut
                for k in range(ncut):
                    j = np.random.choice(group_1_neighbors)
                    G.remove_edge(i, j)
                    group_1_neighbors.remove(j)

    else:
        share_k_matrix = np.ones((2, 2))
        share_k_matrix *= average_degree/np.sum(sizes)

        if p_matrix is None:
            p_matrix = share_k_matrix.copy()

        # create a graph with edges within each groups and between members of different groups using the probability matrix
        G = nx.stochastic_block_model(sizes, p_matrix)

        # how many people in group 2 have connections they could cut to preserve the degree distribution
        group_2_to_group_2_connections = []
        for i in n2:
            group_2_neighbors = [j for j in G.neighbors(i) if j in n2]
            if len(group_2_neighbors) > 0:
                group_2_to_group_2_connections.append(i)

        # there are no people in group 2 who can remove edges to other group 2 people, so instead, just add edges
        if len(group_2_to_group_2_connections) == 0:
            for i in n1:
                group_2_neighbors = [j for j in G.neighbors(i) if j in n2]

                # need to add a contact in group 2
                if len(group_2_neighbors) == 0:

                    random_group_2_j = np.random.choice(n2)
                    G.add_edge(i, random_group_2_j)

        # some in group 2 have contacts to remove to preserve the degree distribution
        else:
            for i in n1:
                group_2_neighbors = [j for j in G.neighbors(i) if j in n2]

                # increase the degree of the node in group 1, while decreasing the degree of a member of group 2 at random
                if len(group_2_neighbors) == 0:

                    random_group_2_j = np.random.choice(n2)
                    random_group_2_neighbors = [ii for ii in G.neighbors(random_group_2_j) if ii in n2]

                    # add an edge to random_group_2_j
                    G.add_edge(i, random_group_2_j)

                    # if the group 2 person has an edge they can cut to their own group, remove it
                    if len(random_group_2_neighbors) > 0:
                        random_group_2_neighbor_cut = np.random.choice(random_group_2_neighbors)
                        G.remove_edge(random_group_2_j, random_group_2_neighbor_cut)

    E = G.edges()
    for e in E:
        i, j = e

        id_i = group[i]
        id_j = group[j]

        popdict[id_i]['contacts'][setting].add(id_j)
        popdict[id_j]['contacts'][setting].add(id_i)

    return popdict


def make_contacts_from_microstructure(datadir, location, state_location, country_location, n, with_industry_code=False):
    """
    Make a popdict from synthetic household, school, and workplace files with uids. If with_industry_code is True, then individuals
    will have a workplace industry code as well (default value is -1 to represent that this data is unavailable). Currently, industry
    codes are only available to assign to populations within the US.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in
        n (int)                   : number of people in the population
        with_industry_code (bool) : If True, assign workplace industry code read in from cached file

    Returns:
        A popdict of people with attributes. Dictionary keys are the IDs of individuals in the population and the values are a dictionary
        for each individual with their attributes, such as age, household ID (hhid), school ID (scid), workplace ID (wpid), workplace
        industry code (wpindcode) if available, and the IDs of their contacts in different layers. Different layers available are
        households ('H'), schools ('S'), and workplaces ('W'). Contacts in these layers are clustered and thus form a network composed of
        groups of people interacting with each other. For example, all household members are contacts of each other, and everyone in the
        same school is considered a contact of each other.

    Notes:
        Methods to trim large groups of contacts down to better approximate a sense of close contacts (such as classroom sizes or
        smaller work groups are available via sp.trim_contacts() - see below).
    """
    file_path = os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, 'contact_networks')

    households_by_uid_path = os.path.join(file_path, location + '_' + str(n) + '_synthetic_households_with_uids.dat')

    if with_industry_code:
        workplaces_by_uid_path = os.path.join(file_path, location + '_' + str(n) + '_synthetic_workplaces_by_industry_with_uids.dat')
        workplaces_by_industry_code_path = os.path.join(file_path, location + '_' + str(n) + '_synthetic_workplaces_by_industry_codes.dat')
    else:
        workplaces_by_uid_path = os.path.join(file_path, location + '_' + str(n) + '_synthetic_workplaces_with_uids.dat')
    schools_by_uid_path = os.path.join(file_path, location + '_' + str(n) + '_synthetic_schools_with_uids.dat')
    teachers_by_uid_path = os.path.join(file_path, location + '_' + str(n) + '_synthetic_teachers_with_uids.dat')

    age_by_uid_dic = sprw.read_in_age_by_uid(datadir, location, state_location, country_location, 'contact_networks', n)
    uids = age_by_uid_dic.keys()
    uids = [uid for uid in uids]

    # uid are strings or ints
    if isinstance(uids[0], str):
        uid_mapping = {uid: u for u, uid in enumerate(uids)}
    elif isinstance(uids[0], np.integer):
        uid_mapping = {u: u for u in uids}
    else:
        errormsg = f'Agent IDs should be either a string or an integer. The IDs being read in do not match either of those types. Please check that the data files being read in are correct.'
        raise ValueError(errormsg)

    # you have ages but not sexes so we'll just populate that for you at random
    popdict = {}
    for i, uid in enumerate(uids):
        u = uid_mapping[uid]
        popdict[u] = {}
        popdict[u]['age'] = int(age_by_uid_dic[uid])
        popdict[u]['sex'] = np.random.binomial(1, p=0.5)
        popdict[u]['loc'] = None
        popdict[u]['contacts'] = {}
        popdict[u]['hhid'] = None
        popdict[u]['scid'] = None
        popdict[u]['sc_student'] = None
        popdict[u]['sc_teacher'] = None
        popdict[u]['wpid'] = None
        popdict[u]['wpindcode'] = None
        for k in ['H', 'S', 'W', 'C']:
            popdict[u]['contacts'][k] = set()

    fh = open(households_by_uid_path, 'r')
    fs = open(schools_by_uid_path, 'r')
    ft = open(teachers_by_uid_path, 'r')
    fw = open(workplaces_by_uid_path, 'r')

    if with_industry_code:
        fi = open(workplaces_by_industry_code_path, 'r')
        workplaces_by_industry_codes = np.loadtxt(fi)
        fi.close()

    # map uids to ints in the popdict created
    if isinstance(uids[0], str):
        # read in home contacts
        for nh, line in enumerate(fh):
            r = line.strip().split(' ')
            household = [uid_mapping[uid] for uid in r]

            for u in household:
                popdict[u]['contacts']['H'] = set(household)
                popdict[u]['contacts']['H'].remove(u)
                popdict[u]['hhid'] = nh

        # read in school contacts
        for ns, (line1, line2) in enumerate(zip(fs, ft)):
            r1 = line1.strip().split(' ')
            r2 = line2.strip().split(' ')

            students = [uid_mapping[uid] for uid in r1]
            teachers = [uid_mapping[uid] for uid in r2]

            # for uid in school:
            for u in students:
                popdict[u]['contacts']['S'] = set(students)
                popdict[u]['contacts']['S'] = popdict[u]['contacts']['S'].union(set(teachers))
                popdict[u]['contacts']['S'].remove(u)
                popdict[u]['scid'] = ns
                popdict[u]['sc_student'] = 1

            for u in teachers:
                popdict[u]['contacts']['S'] = set(students)
                popdict[u]['contacts']['S'] = popdict[u]['contacts']['S'].union(set(teachers))
                popdict[u]['contacts']['S'].remove(u)
                popdict[u]['scid'] = ns
                popdict[u]['sc_teacher'] = 1

        # read in workplace contacts
        for nw, line in enumerate(fw):
            r = line.strip().split(' ')
            workplace = [uid_mapping[uid] for uid in r]

            for u in workplace:
                popdict[u]['contacts']['W'] = set(workplace)
                popdict[u]['contacts']['W'].remove(u)
                popdict[u]['wpid'] = nw
                if with_industry_code:
                    popdict[u]['wpindcode'] = int(workplaces_by_industry_codes[nw])

    # uids are ints so no need to do any mapping
    # elif isinstance(uids[0], int) or isinstance(uids[0], np.int64):
    elif isinstance(uids[0], np.integer):
        # read in home contacts
        for nh, line in enumerate(fh):
            r = line.strip().split(' ')
            home = [int(u) for u in r]

            for u in home:
                popdict[u]['contacts']['H'] = set(home)
                popdict[u]['contacts']['H'].remove(u)
                popdict[u]['hhid'] = nh

        # read in school contacts
        for ns, (line1, line2) in enumerate(zip(fs, ft)):
            r1 = line1.strip().split(' ')
            r2 = line2.strip().split(' ')

            students = [int(u) for u in r1]
            teachers = [int(u) for u in r2]

            for u in students:
                popdict[u]['contacts']['S'] = set(students)
                popdict[u]['contacts']['S'] = popdict[u]['contacts']['S'].union(set(teachers))
                popdict[u]['contacts']['S'].remove(u)
                popdict[u]['scid'] = ns
                popdict[u]['sc_student'] = 1

            for u in teachers:
                popdict[u]['contacts']['S'] = set(students)
                popdict[u]['contacts']['S'] = popdict[u]['contacts']['S'].union(set(teachers))
                popdict[u]['contacts']['S'].remove(u)
                popdict[u]['scid'] = ns
                popdict[u]['sc_teacher'] = 1

        # read in workplace contacts
        for nw, line in enumerate(fw):
            r = line.strip().split(' ')
            workplace = [int(u) for u in r]

            for u in workplace:
                popdict[u]['contacts']['W'] = set(workplace)
                popdict[u]['contacts']['W'].remove(u)
                popdict[u]['wpid'] = nw
                if with_industry_code:
                    popdict[u]['wpindcode'] = int(workplaces_by_industry_codes[nw])

    fh.close()
    fs.close()
    ft.close()
    fw.close()

    return popdict


def make_contacts_from_microstructure_objects(age_by_uid_dic, homes_by_uids, schools_by_uids, teachers_by_uids, workplaces_by_uids, workplaces_by_industry_codes=None):
    """
    From microstructure objects (dictionary mapping ID to age, lists of lists in different settings, etc.), create a dictionary of individuals.
    Each key is the ID of an individual which maps to a dictionary for that individual with attributes such as their age, household ID (hhid),
    school ID (scid), workplace ID (wpid), workplace industry code (wpindcode) if available, and contacts in different layers.

    Args:
        age_by_uid_dic (dict)                             : dictionary mapping id to age for all individuals in the population
        homes_by_uids (list)                              : A list of lists where each sublist is a household and the IDs of the household members.
        schools_by_uids (list)                            : list of lists, where each sublist represents a school and the ids of the students and teachers within it
        workplaces_by_uids (list)                         : list of lists, where each sublist represents a workplace and the ids of the workers within it
        workplaces_by_industry_codes (np.ndarray or None) : array with workplace industry code for each workplace

    Returns:
        A popdict of people with attributes. Dictionary keys are the IDs of individuals in the population and the values are a dictionary
        for each individual with their attributes, such as age, household ID (hhid), school ID (scid), workplace ID (wpid), workplace
        industry code (wpindcode) if available, and the IDs of their contacts in different layers. Different layers available are
        households ('H'), schools ('S'), and workplaces ('W'). Contacts in these layers are clustered and thus form a network composed of
        groups of people interacting with each other. For example, all household members are contacts of each other, and everyone in the
        same school is considered a contact of each other.

    Notes:
        Methods to trim large groups of contacts down to better approximate a sense of close contacts (such as classroom sizes or
        smaller work groups are available via sp.trim_contacts() - see below).
    """
    uids = age_by_uid_dic.keys()
    uids = [uid for uid in uids]

    # uid are strings or ints
    if isinstance(uids[0], str):
        uid_mapping = {uid: u for u, uid in enumerate(uids)}
    elif isinstance(uids[0], np.integer):
        uid_mapping = {u: u for u in uids}
    else:
        errormsg = f'Agent IDs should be either a string or an integer. The IDs being read in do not match either of those types. Please check that the data files being read in are correct.'
        raise ValueError(errormsg)

    popdict = {}
    for uid in age_by_uid_dic:
        u = uid_mapping[uid]
        popdict[u] = {}
        popdict[u]['age'] = int(age_by_uid_dic[uid])
        popdict[u]['sex'] = np.random.binomial(1, p=0.5)
        popdict[u]['loc'] = None
        popdict[u]['contacts'] = {}
        popdict[u]['hhid'] = None
        popdict[u]['scid'] = None
        popdict[u]['sc_student'] = None
        popdict[u]['sc_teacher'] = None
        popdict[u]['wpid'] = None
        popdict[u]['wpindcode'] = None
        for k in ['H', 'S', 'W', 'C']:
            popdict[u]['contacts'][k] = set()

    if isinstance(uids[0], str):
        # read in home contacts
        for nh, household in enumerate(homes_by_uids):
            household = [uid_mapping[uid] for uid in household]

            for u in household:
                popdict[u]['contacts']['H'] = set(household)
                popdict[u]['contacts']['H'].remove(u)
                popdict[u]['hhid'] = nh

        # read in school contacts
        for ns, students in enumerate(schools_by_uids):
            students = [uid_mapping[uid] for uid in students]
            teachers = teachers_by_uids[ns]
            teachers = [uid_mapping[uid] for uid in teachers]

            for u in students:
                popdict[u]['contacts']['S'] = set(students)
                popdict[u]['contacts']['S'] = popdict[u]['contacts']['S'].union(set(teachers))
                popdict[u]['contacts']['S'].remove(u)
                popdict[u]['scid'] = ns
                popdict[u]['sc_student'] = 1

            for u in teachers:
                popdict[u]['contacts']['S'] = set(students)
                popdict[u]['contacts']['S'] = popdict[u]['contacts']['S'].union(set(teachers))
                popdict[u]['contacts']['S'].remove(u)
                popdict[u]['scid'] = ns
                popdict[u]['sc_teacher'] = 1

        # read in workplace contacts
        for nw, workplace in enumerate(workplaces_by_uids):
            workplace = [uid_mapping[uid] for uid in workplace]

            for u in workplace:
                popdict[u]['contacts']['W'] = set(workplace)
                popdict[u]['contacts']['W'].remove(u)
                popdict[u]['wpid'] = nw
                if workplaces_by_industry_codes is not None:
                    popdict[u]['wpindcode'] = int(workplaces_by_industry_codes[nw])

    # elif isinstance(uids[0], int) or isinstance(uids[0], np.int64):
    elif isinstance(uids[0], np.integer):
        # read in home contacts
        for nh, household in enumerate(homes_by_uids):

            for u in household:
                popdict[u]['contacts']['H'] = set(household)
                popdict[u]['contacts']['H'].remove(u)
                popdict[u]['hhid'] = nh

        # read in school contacts
        for ns, students in enumerate(schools_by_uids):
            teachers = teachers_by_uids[ns]

            for u in students:
                popdict[u]['contacts']['S'] = set(students)
                popdict[u]['contacts']['S'] = popdict[u]['contacts']['S'].union(set(teachers))
                popdict[u]['contacts']['S'].remove(u)
                popdict[u]['scid'] = ns
                popdict[u]['sc_student'] = 1

            for u in teachers:
                popdict[u]['contacts']['S'] = set(students)
                popdict[u]['contacts']['S'] = popdict[u]['contacts']['S'].union(set(teachers))
                popdict[u]['contacts']['S'].remove(u)
                popdict[u]['scid'] = ns
                popdict[u]['sc_teacher'] = 1

        # read in workplace contacts
        for nw, workplace in enumerate(workplaces_by_uids):

            for u in workplace:
                popdict[u]['contacts']['W'] = set(workplace)
                popdict[u]['contacts']['W'].remove(u)
                popdict[u]['wpid'] = nw
                if workplaces_by_industry_codes is not None:
                    popdict[u]['wpindcode'] = int(workplaces_by_industry_codes[nw])

    return popdict


def make_contacts_with_facilities_from_microstructure(datadir, location, state_location, country_location, n, use_two_group_reduction=False, average_LTCF_degree=20):
    """
    Make a popdict from synthetic household, school, and workplace files with uids. If with_industry_code is True, then individuals
    will have a workplace industry code as well (default value is -1 to represent that this data is unavailable). Currently, industry
    codes are only available to assign to populations within the US.

    Args:
        datadir (string)               : file path to the data directory
        location (string)              : name of the location
        state_location (string)        : name of the state the location is in
        country_location (string)      : name of the country the location is in
        n (int)                        : number of people in the population
        with_industry_code (bool)      : If True, assign workplace industry code read in from cached file
        use_two_group_reduction (bool) : If True, create long term care facilities with reduced contacts across both groups
        average_LTCF_degree (int)      : default average degree in long term care facilities

    Returns:
        A popdict of people with attributes. Dictionary keys are the IDs of individuals in the population and the values are a dictionary
        for each individual with their attributes, such as age, household ID (hhid), school ID (scid), workplace ID (wpid), workplace
        industry code (wpindcode) if available, and the IDs of their contacts in different layers. Different layers available are
        households ('H'), schools ('S'), and workplaces ('W'), and long term care facilities ('LTCF'). Contacts in these layers are clustered and thus form a network composed of
        groups of people interacting with each other. For example, all household members are contacts of each other, and everyone in the
        same school is considered a contact of each other. If use_two_group_reduction is True, then contracts within 'LTCF' are reduced
        from fully connected.

    Notes:
        Methods to trim large groups of contacts down to better approximate a sense of close contacts (such as classroom sizes or
        smaller work groups are available via sp.trim_contacts() or sp.create_reduced_contacts_with_group_types(): see these methods for more details).
    """
    file_path = os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, 'contact_networks_facilities')

    # age_by_uid_path = os.path.join(file_path, location + '_' + str(n) + '_age_by_uid.dat')

    households_by_uid_path = os.path.join(file_path, location + '_' + str(n) + '_synthetic_households_with_uids.dat')
    workplaces_by_uid_path = os.path.join(file_path, location + '_' + str(n) + '_synthetic_workplaces_with_uids.dat')
    schools_by_uid_path = os.path.join(file_path, location + '_' + str(n) + '_synthetic_schools_with_uids.dat')
    teachers_by_uid_path = os.path.join(file_path, location + '_' + str(n) + '_synthetic_teachers_with_uids.dat')
    facilities_by_uid_path = os.path.join(file_path, location + '_' + str(n) + '_synthetic_facilities_with_uids.dat')
    facilities_staff_by_uid_path = os.path.join(file_path, location + '_' + str(n) + '_synthetic_facilities_staff_with_uids.dat')

    # df = pd.read_csv(age_by_uid_path, delimiter=' ', header=None)
    # age_by_uid_dic = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
    age_by_uid_dic = sprw.read_in_age_by_uid(datadir, location, state_location, country_location, 'contact_networks_facilities', n)
    uids = age_by_uid_dic.keys()
    uids = [uid for uid in uids]

    # uid are strings or ints
    if isinstance(uids[0], str):
        uid_mapping = {uid: u for u, uid in enumerate(uids)}
    elif isinstance(uids[0], np.integer):
        uid_mapping = {u: u for u in uids}
    else:
        errormsg = f'Agent IDs should be either a string or an integer. The IDs being read in do not match either of those types. Please check that the data files being read in are correct.'
        raise ValueError(errormsg)

    # you have ages but not sexes so we'll just populate that for you at random
    popdict = {}
    for i, uid in enumerate(uids):
        u = uid_mapping[uid]
        popdict[u] = {}
        popdict[u]['age'] = int(age_by_uid_dic[uid])
        popdict[u]['sex'] = np.random.binomial(1, p=0.5)
        popdict[u]['loc'] = None
        popdict[u]['contacts'] = {}
        popdict[u]['snf_res'] = None
        popdict[u]['snf_staff'] = None
        popdict[u]['hhid'] = None
        popdict[u]['scid'] = None
        popdict[u]['sc_student'] = None
        popdict[u]['sc_teacher'] = None
        popdict[u]['wpid'] = None
        popdict[u]['snfid'] = None
        for k in ['H', 'S', 'W', 'C', 'LTCF']:
            popdict[u]['contacts'][k] = set()

    fh = open(households_by_uid_path, 'r')
    fs = open(schools_by_uid_path, 'r')
    ft = open(teachers_by_uid_path, 'r')
    fw = open(workplaces_by_uid_path, 'r')
    ffres = open(facilities_by_uid_path, 'r')
    ffstaff = open(facilities_staff_by_uid_path, 'r')

    if isinstance(uids[0], str):
        # read in facility residents and staff
        for nf, (line1, line2) in enumerate(zip(ffres, ffstaff)):
            r1 = line1.strip().split(' ')
            r2 = line2.strip().split(' ')

            facility = [uid_mapping[uid] for uid in r1]
            facility_staff = [uid_mapping[uid] for uid in r2]

            for u in facility:
                popdict[u]['snf_res'] = 1
                popdict[u]['snfid'] = nf

            for u in facility_staff:
                popdict[u]['snf_staff'] = 1
                popdict[u]['snfid'] = nf

            if use_two_group_reduction:
                popdict = create_reduced_contacts_with_group_types(popdict, r1, r2, 'LTCF', average_degree=average_LTCF_degree, force_cross_edges=True)

            else:
                for u in facility:
                    popdict[u]['contacts']['LTCF'] = set(facility)
                    popdict[u]['contacts']['LTCF'] = popdict[u]['contacts']['LTCF'].union(set(facility_staff))
                    popdict[u]['contacts']['LTCF'].remove(u)

                for u in facility_staff:
                    popdict[u]['contacts']['LTCF'] = set(facility)
                    popdict[u]['contacts']['LTCF'] = popdict[u]['contacts']['LTCF'].union(set(facility_staff))
                    popdict[u]['contacts']['LTCF'].remove(u)

        # read in home contacts
        for nh, line in enumerate(fh):
            r = line.strip().split(' ')
            household = [uid_mapping[uid] for uid in r]

            for u in household:
                popdict[u]['contacts']['H'] = set(household)
                popdict[u]['contacts']['H'].remove(u)
                popdict[u]['hhid'] = nh

        # read in school contacts
        for ns, (line1, line2) in enumerate(zip(fs, ft)):
            r1 = line1.strip().split(' ')
            r2 = line2.strip().split(' ')

            students = [uid_mapping[uid] for uid in r1]
            teachers = [uid_mapping[uid] for uid in r2]

            for u in students:
                popdict[u]['contacts']['S'] = set(students)
                popdict[u]['contacts']['S'] = popdict[u]['contacts']['S'].union(set(teachers))
                popdict[u]['contacts']['S'].remove(u)
                popdict[u]['scid'] = ns
                popdict[u]['sc_student'] = 1

            for u in teachers:
                popdict[u]['contacts']['S'] = set(students)
                popdict[u]['contacts']['S'] = popdict[u]['contacts']['S'].union(set(teachers))
                popdict[u]['contacts']['S'].remove(u)
                popdict[u]['scid'] = ns
                popdict[u]['sc_teacher'] = 1

        # read in workplace contacts
        for nw, line in enumerate(fw):
            r = line.strip().split(' ')
            workplace = [uid_mapping[uid] for uid in r]

            for u in workplace:
                popdict[u]['contacts']['W'] = set(workplace)
                popdict[u]['contacts']['W'].remove(u)
                popdict[u]['wpid'] = nw

    # elif isinstance(uids[0], int) or isinstance(uids[0], np.int64):
    elif isinstance(uids[0], np.integer):
        # read in facility residents and staff
        for nf, (line1, line2) in enumerate(zip(ffres, ffstaff)):
            r1 = line1.strip().split(' ')
            r2 = line2.strip().split(' ')

            facility = [int(u) for u in r1]
            facility_staff = [int(u) for u in r2]

            for u in facility:
                popdict[u]['snf_res'] = 1
                popdict[u]['snfid'] = nf

            for u in facility_staff:
                popdict[u]['snf_staff'] = 1
                popdict[u]['snfid'] = nf

            if use_two_group_reduction:
                popdict = create_reduced_contacts_with_group_types(popdict, r1, r2, 'LTCF', average_degree=average_LTCF_degree, force_cross_edges=True)

            else:
                for u in facility:
                    popdict[u]['contacts']['LTCF'] = set(facility)
                    popdict[u]['contacts']['LTCF'] = popdict[u]['contacts']['LTCF'].union(set(facility_staff))
                    popdict[u]['contacts']['LTCF'].remove(u)

                for u in facility_staff:
                    popdict[u]['contacts']['LTCF'] = set(facility)
                    popdict[u]['contacts']['LTCF'] = popdict[u]['contacts']['LTCF'].union(set(facility_staff))
                    popdict[u]['contacts']['LTCF'].remove(u)

        # read in home contacts
        for nh, line in enumerate(fh):
            r = line.strip().split(' ')
            household = [int(u) for u in r]

            for u in household:
                popdict[u]['contacts']['H'] = set(household)
                popdict[u]['contacts']['H'].remove(u)
                popdict[u]['hhid'] = nh

        # read in school contacts
        for ns, (line1, line2) in enumerate(zip(fs, ft)):
            r1 = line1.strip().split(' ')
            r2 = line2.strip().split(' ')

            students = [int(u) for u in r1]
            teachers = [int(u) for u in r2]

            for u in students:
                popdict[u]['contacts']['S'] = set(students)
                popdict[u]['contacts']['S'] = popdict[u]['contacts']['S'].union(set(teachers))
                popdict[u]['contacts']['S'].remove(u)
                popdict[u]['scid'] = ns
                popdict[u]['sc_student'] = 1

            for u in teachers:
                popdict[u]['contacts']['S'] = set(students)
                popdict[u]['contacts']['S'] = popdict[u]['contacts']['S'].union(set(teachers))
                popdict[u]['contacts']['S'].remove(u)
                popdict[u]['scid'] = ns
                popdict[u]['sc_teacher'] = 1

        # read in workplace contacts
        for nw, line in enumerate(fw):
            r = line.strip().split(' ')
            workplace = [int(u) for u in r]

            for u in workplace:
                popdict[u]['contacts']['W'] = set(workplace)
                popdict[u]['contacts']['W'].remove(u)
                popdict[u]['wpid'] = nw

    fh.close()
    fs.close()
    ft.close()
    fw.close()
    ffres.close()
    ffstaff.close()

    return popdict


def make_contacts_with_facilities_from_microstructure_objects(age_by_uid_dic, homes_by_uids, schools_by_uids, teachers_by_uids, workplaces_by_uids, facilities_by_uids, facilities_staff_uids, workplaces_by_industry_codes=None, use_two_group_reduction=False, average_LTCF_degree=20):
    """
    From microstructure objects (dictionary mapping ID to age, lists of lists in different settings, etc.), create a dictionary of individuals.
    Each key is the ID of an individual which maps to a dictionary for that individual with attributes such as their age, household ID (hhid),
    school ID (scid), workplace ID (wpid), workplace industry code (wpindcode) if available, and contacts in different layers.

    Args:
        age_by_uid_dic (dict)                             : dictionary mapping id to age for all individuals in the population
        homes_by_uids (list)                              : A list of lists where each sublist is a household and the IDs of the household members.
        schools_by_uids (list)                            : A list of lists, where each sublist represents a school and the ids of the students and teachers within it
        workplaces_by_uids (list)                         : A list of lists, where each sublist represents a workplace and the ids of the workers within it
        facilities_by_uids (list): A list of lists, where each sublist represents a skilled nursing or long term care facility and the ids of the residents living within it
        facilities_staff_uids (list): A list of lists, where each sublist represents a skilled nursing or long term care facility and the ids of the staff working within it
        workplaces_by_industry_codes (np.ndarray or None) : array with workplace industry code for each workplace
        use_two_group_reduction (bool) : If True, create long term care facilities with reduced contacts across both groups
        average_LTCF_degree (int)      : default average degree in long term care facilities

    Returns:
        A popdict of people with attributes. Dictionary keys are the IDs of individuals in the population and the values are a dictionary
        for each individual with their attributes, such as age, household ID (hhid), school ID (scid), workplace ID (wpid), workplace
        industry code (wpindcode) if available, and the IDs of their contacts in different layers. Different layers available are
        households ('H'), schools ('S'), and workplaces ('W'), and long term care facilities ('LTCF'). Contacts in these layers are clustered and thus form a network composed of
        groups of people interacting with each other. For example, all household members are contacts of each other, and everyone in the
        same school is considered a contact of each other. If use_two_group_reduction is True, then contracts within 'LTCF' are reduced
        from fully connected.

    Notes:
        Methods to trim large groups of contacts down to better approximate a sense of close contacts (such as classroom sizes or
        smaller work groups are available via sp.trim_contacts() or sp.create_reduced_contacts_with_group_types(): see these methods for more details).
    """

    uids = age_by_uid_dic.keys()
    uids = [uid for uid in uids]

    # uid are strings or ints
    if isinstance(uids[0], str):
        uid_mapping = {uid: u for u, uid in enumerate(uids)}
    elif isinstance(uids[0], np.integer):
        uid_mapping = {u: u for u in uids}
    else:
        errormsg = f'Agent IDs should be either a string or an integer. The IDs being read in do not match either of those types. Please check that the data files being read in are correct.'
        raise ValueError(errormsg)

    popdict = {}
    for uid in age_by_uid_dic:
        u = uid_mapping[uid]
        popdict[u] = {}
        popdict[u]['age'] = int(age_by_uid_dic[uid])
        popdict[u]['sex'] = np.random.randint(2)
        popdict[u]['loc'] = None
        popdict[u]['contacts'] = {}
        popdict[u]['snf_res'] = None
        popdict[u]['snf_staff'] = None
        popdict[u]['hhid'] = None
        popdict[u]['scid'] = None
        popdict[u]['wpid'] = None
        popdict[u]['snfid'] = None
        for k in ['H', 'S', 'W', 'C', 'LTCF']:
            popdict[u]['contacts'][k] = set()

    if isinstance(uids[0], str):
        # read in facility residents and staff
        for nf, facility in enumerate(facilities_by_uids):
            facility = [uid_mapping[uid] for uid in facility]
            facility_staff = facilities_staff_uids[nf]
            facility_staff = [uid_mapping[uid] for uid in facility_staff]

            for u in facility:
                popdict[u]['snf_res'] = 1
                popdict[u]['snfid'] = nf

            for u in facility_staff:
                popdict[u]['snf_staff'] = 1
                popdict[u]['snfid'] = nf

            if use_two_group_reduction:
                popdict = create_reduced_contacts_with_group_types(popdict, facility, facility_staff, 'LTCF', average_degree=average_LTCF_degree, force_cross_edges=True)

            else:
                for u in facility:
                    popdict[u]['contacts']['LTCF'] = set(facility)
                    popdict[u]['contacts']['LTCF'] = popdict[u]['contacts']['LTCF'].union(set(facility_staff))
                    popdict[u]['contacts']['LTCF'].remove(u)

                for u in facility_staff:
                    popdict[u]['contacts']['LTCF'] = set(facility)
                    popdict[u]['contacts']['LTCF'] = popdict[u]['contacts']['LTCF'].union(set(facility_staff))
                    popdict[u]['contacts']['LTCF'].remove(u)

        # read in home contacts
        for nh, household in enumerate(homes_by_uids):
            household = [uid_mapping[uid] for uid in household]

            for u in household:
                popdict[u]['contacts']['H'] = set(household)
                popdict[u]['contacts']['H'].remove(u)
                popdict[u]['hhid'] = nh

        # read in school contacts
        for ns, students in enumerate(schools_by_uids):
            students = [uid_mapping[uid] for uid in students]
            teachers = teachers_by_uids[ns]
            teachers = [uid_mapping[uid] for uid in teachers]

            for u in students:
                popdict[u]['contacts']['S'] = set(students)
                popdict[u]['contacts']['S'] = popdict[u]['contacts']['S'].union(set(teachers))
                popdict[u]['contacts']['S'].remove(u)
                popdict[u]['scid'] = ns
                popdict[u]['sc_student'] = 1

            for u in teachers:
                popdict[u]['contacts']['S'] = set(students)
                popdict[u]['contacts']['S'] = popdict[u]['contacts']['S'].union(set(teachers))
                popdict[u]['contacts']['S'].remove(u)
                popdict[u]['scid'] = ns
                popdict[u]['sc_teacher'] = 1

        # read in workplace contacts
        for nw, workplace in enumerate(workplaces_by_uids):
            workplace = [uid_mapping[uid] for uid in workplace]

            for u in workplace:
                popdict[u]['contacts']['W'] = set(workplace)
                popdict[u]['contacts']['W'].remove(u)
                popdict[u]['wpid'] = nw
                if workplaces_by_industry_codes is not None:
                    popdict[u]['wpindcode'] = int(workplaces_by_industry_codes[nw])

    # elif isinstance(uids[0], int) or isinstance(uids[0], np.int64):
    elif isinstance(uids[0], np.integer):
        # read in facility residents and staff
        for nf, facility in enumerate(facilities_by_uids):
            facility_staff = facilities_staff_uids[nf]

            for u in facility:
                popdict[u]['snf_res'] = 1
                popdict[u]['snfid'] = nf

            for u in facility_staff:
                popdict[u]['snf_staff'] = 1
                popdict[u]['snfid'] = nf

            if use_two_group_reduction:
                popdict = create_reduced_contacts_with_group_types(popdict, facility, facility_staff, 'LTCF', average_degree=average_LTCF_degree, force_cross_edges=True)

            else:
                for u in facility:
                    popdict[u]['contacts']['LTCF'] = set(facility)
                    popdict[u]['contacts']['LTCF'] = popdict[u]['contacts']['LTCF'].union(set(facility_staff))
                    popdict[u]['contacts']['LTCF'].remove(u)

                for u in facility_staff:
                    popdict[u]['contacts']['LTCF'] = set(facility)
                    popdict[u]['contacts']['LTCF'] = popdict[u]['contacts']['LTCF'].union(set(facility_staff))
                    popdict[u]['contacts']['LTCF'].remove(u)

        # read in home contacts
        for nh, household in enumerate(homes_by_uids):
            for u in household:
                popdict[u]['contacts']['H'] = set(household)
                popdict[u]['contacts']['H'].remove(u)
                popdict[u]['hhid'] = nh

        # read in school contacts
        for ns, school in enumerate(schools_by_uids):
            teachers = teachers_by_uids[ns]
            for u in school:
                popdict[u]['contacts']['S'] = set(school)
                popdict[u]['contacts']['S'] = popdict[u]['contacts']['S'].union(set(teachers))
                popdict[u]['contacts']['S'].remove(u)
                popdict[u]['scid'] = ns
                popdict[u]['sc_student'] = 1

            for u in teachers:
                popdict[u]['contacts']['S'] = set(school)
                popdict[u]['contacts']['S'] = popdict[u]['contacts']['S'].union(set(teachers))
                popdict[u]['contacts']['S'].remove(u)
                popdict[u]['scid'] = ns
                popdict[u]['sc_teacher'] = 1

        # read in workplace contacts
        for nw, workplace in enumerate(workplaces_by_uids):
            for u in workplace:
                popdict[u]['contacts']['W'] = set(workplace)
                popdict[u]['contacts']['W'].remove(u)
                popdict[u]['wpid'] = nw
                if workplaces_by_industry_codes is not None:
                    popdict[u]['wpindcode'] = int(workplaces_by_industry_codes[nw])

    return popdict


def make_graphs(popdict, layers):
    """
    Make a dictionary of Networkx by layer.

    Args:
        popdict (dict) : dictionary of individuals with attributes, including their age, household ID, school ID, workplace ID, and the ids of their contacts by layer
        layers (list)  : list of contact layers

    Retuns:
        Dictionary of Networkx graphs, one for each layer of contacts.
    """
    G_dic = {}

    for i, layer in enumerate(layers):
        G = nx.Graph()
        for uid in popdict:
            contacts = popdict[uid]['contacts'][layer]
            for j in contacts:
                G.add_edge(uid, j)
        G_dic[layer] = G
    return G_dic


def write_edgelists(popdict, layers, G_dic=None, location=None, state_location=None, country_location=None):
    """
    Write edgelists for each layer of contacts.

    Args:
        popdict (dict)            : dictionary of individuals with attributes, including their age, household ID, school ID, workplace ID, and the ids of their contacts by layer
        layers (list)             : list of contact layers
        G_dic (dict)              : dictionary of Networkx graphs, one for each layer of contacts
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in

    Returns:
        None
    """
    n = len(popdict)
    layer_names = {'H': 'households', 'S': 'schools', 'W': 'workplaces'}
    if G_dic is None:
        G_dic = make_graphs(popdict, layers)
    for layer in G_dic:
        file_path = os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, 'contact_networks')
        file_path = os.path.join(file_path, location + '_' + str(n) + '_synthetic_' + layer_names[layer] + '_edgelist.dat')
        nx.write_edgelist(G_dic[layer], file_path, data=False)


def make_contacts(popdict=None, n_contacts_dic=None, location=None, state_location=None, country_location=None, sheet_name=None, options_args=None, activity_args=None, network_distr_args=None):
    '''
    Generates a list of contacts for everyone in the population. popdict is a
    dictionary with N keys (one for each person), with subkeys for age, sex, location,
    and potentially other factors. This function adds a new subkey, contacts, which
    is a list of contact IDs for each individual. If directed=False (default),
    if person A is a contact of person B, then person B is also a contact of person A.

    Example output (input is the same, minus the "contacts" field)::

        popdict = {
            '8acf08f0': {
                'age': 57.3,
                'sex': 0,
                'loc': (47.6062, 122.3321),
                'contacts': ['43da76b5']
                },
            '43da76b5': {
                'age': 55.3,
                'sex': 1,
                'loc': (47.2473, 122.6482),
                'contacts': ['8acf08f0', '2d2ad46f']
                },
        }

    Args:
        popdict (dict)            : dictionary, should have ages of individuals if not using cached microstructure data
        n_contacts_dic (dict)     : average number of contacts by setting
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in
        sheet_name (string)       : name of the sheet in the excel file with contact patterns
        options_args (dict)       : dictionary of flags to set different population and contact generating options
        activity_args (dict)      : dictionary of age bounds for participating in different activities like going to school or working, also student-teacher ratio
        network_distr_args (dict) : network distribution parameters dictionary for average_degree, network_type, and directionality, can also include powerlaw exponents,
                                    block sizes (re: SBMs), clustering distribution, or other properties needed to generate network structures. Checkout
                                    https://networkx.github.io/documentation/stable/reference/generators.html#module-networkx.generators for what's possible
                                    Default 'network_type' is 'poisson_degree' for Erdos-Renyi random graphs in large n limit.

    Returns:
        A dictionary of individuals with attributes, including their age and the ids of their contacts.

    '''
    ### Defaults ###
    if location             is None: location = 'seattle_metro'
    if state_location       is None: state_location = 'Washington'
    if country_location     is None: country_location = 'usa'
    if sheet_name           is None: sheet_name = 'United States of America'

    if n_contacts_dic       is None: n_contacts_dic = {'H': 4, 'S': 20, 'W': 20, 'C': 20}

    if network_distr_args   is None: network_distr_args = {'average_degree': 30, 'directed': False, 'network_type': 'poisson_degree'}  # general we should default to undirected because directionality doesn't make sense for infectious diseases
    if 'network_type' not in network_distr_args: network_distr_args['network_type'] = 'poisson_degree'
    if 'directed' not in network_distr_args: network_distr_args['directed'] = False
    if 'average_degree' not in network_distr_args: network_distr_args['average_degree'] = 30

    ### Rationale behind default activity_args parameters
    # college_age_max: 22: Because many people in the usa context finish tertiary school of some form (vocational, community college, university), but not all and this is a rough cutoff
    # student_teacher_ratio: 30: King County, WA records seem to indicate median value near that (many many 1 student classrooms skewing the average) - could vary and may need to be lowered to account for extra staff in schools
    # worker_age_min: 23: to keep ages for different activities clean
    # worker_age_max: 65: age at which people are able to retire in many places
    # activity_args might also include different n_contacts for college kids ....
    if activity_args        is None: activity_args = {'student_age_min': 4, 'student_age_max': 18, 'student_teacher_ratio': 30, 'worker_age_min': 23, 'worker_age_max': 65, 'college_age_min': 18, 'college_age_max': 23}

    options_keys = ['use_age', 'use_sex', 'use_loc', 'use_social_layers', 'use_activity_rates', 'use_microstructure', 'use_age_mixing', 'use_industry_code', 'use_long_term_care_facilities', 'use_two_group_reduction']
    if options_args is None: options_args = dict.fromkeys(options_keys, False)
    if options_args.get('average_LTCF_degree') is None: options_args['average_LTCF_degree'] = 20

    # fill in the other keys as False!
    for key in options_keys:
        if key not in options_args:
            options_args[key] = False

    # to call in pre-generated contact networks that exhibit real-world-like clustering and age-specific mixing
    if options_args['use_microstructure']:
        if 'Npop' not in network_distr_args: network_distr_args['Npop'] = 10000
        country_location = 'usa'
        if options_args['use_long_term_care_facilities']:
            popdict = make_contacts_with_facilities_from_microstructure(datadir, location, state_location, country_location, network_distr_args['Npop'], options_args['use_two_group_reduction'], options_args['average_LTCF_degree'])
        else:
            popdict = make_contacts_from_microstructure(datadir, location, state_location, country_location, network_distr_args['Npop'], options_args['use_industry_code'])

    # to generate contact networks that observe age-specific mixing but not clustering (for locations that haven't been vetted by the microstructure generation method in contact_networks.py or for which we don't have enough data to do that)
    else:
        # for locations with sex by age data - likely only for the US
        if options_args['use_age_mixing'] and options_args['use_sex']:
            if options_args['use_social_layers']:
                popdict = make_contacts_with_social_layers_and_sex(popdict, n_contacts_dic, location, state_location, country_location, sheet_name, activity_args, network_distr_args)
            else:
                popdict = make_contacts_without_social_layers_and_sex(popdict, n_contacts_dic, location, state_location, country_location, sheet_name, network_distr_args)

        # for locations without sex by age data (basically anywhere outside of the US)
        elif options_args['use_age_mixing'] and not options_args['use_sex']:
            if options_args['use_social_layers']:
                popdict = make_contacts_with_social_layers_152(popdict, n_contacts_dic, location, state_location, country_location, sheet_name, activity_args, network_distr_args)
            else:
                popdict = make_contacts_without_social_layers_152(popdict, n_contacts_dic, location, state_location, country_location, sheet_name, network_distr_args)

        else:
            # this makes the generic case with a default age and sex distribution : if you give the popdict with ages it'll connect people at random with different ages but not according to any age-mixing data.
            popdict = make_contacts_generic(popdict, network_distr_args)

    return popdict


@nb.njit((nb.int64[:], nb.int64))
def choose_contacts(a, size):
    ''' Numbafy np.random.choice(); about twice as fast '''
    close_contacts = np.random.choice(a, size=size, replace=False)
    return close_contacts


def trim_contacts(contacts, trimmed_size_dic=None, use_clusters=False, verbose=False):

    """
    Trim down contacts in school or work environments from everyone.

    Args:
        contacts (dict)         : dictionary of individuals with attributes, including their age and the ids of their contacts
        trimmed_size_dic (dict) : dictionary of threshold values for the number of contacts in school ('S') and work ('W') so that for individuals with more contacts than this, we select a smaller subset of contacts considerd close contacts
        use_clusters (bool)     : If True, trimmed down contact networks will preserve clustering so that an individual's close contacts in school or at work are also contacts of each other
        verbose (bool)          : If True, print average number of close contacts in school and at work

    Returns:
        A dictionary of individuals with attributes, including their age and the ids of their close contacts.
    """

    trimmed_size_dic = sc.mergedicts({'S': 20, 'W': 10}, trimmed_size_dic)

    keys = trimmed_size_dic.keys()

    if isinstance(list(contacts.keys())[0], str):
        string_uids = True
    else:
        string_uids = False

    if use_clusters:
        raise NotImplementedError("Clustered method not yet implemented.")

    else:

        if not string_uids:
            for n, uid in enumerate(contacts):
                for k in keys:
                    setting_contacts = np.array(list(contacts[uid]['contacts'][k]), dtype=np.int64)
                    if len(setting_contacts) > trimmed_size_dic[k]/2:
                        close_contacts = choose_contacts(setting_contacts, size=int(trimmed_size_dic[k]/2))
                        contacts[uid]['contacts'][k] = set(close_contacts)
        else:
            for n, uid in enumerate(contacts):
                for k in keys:
                    setting_contacts = list(contacts[uid]['contacts'][k])
                    if len(setting_contacts) > trimmed_size_dic[k]/2:
                        close_contacts = np.random.choice(setting_contacts, size=int(trimmed_size_dic[k]/2))
                        contacts[uid]['contacts'][k] = set(close_contacts)

        for n, uid in enumerate(contacts):
            for k in keys:
                for c in contacts[uid]['contacts'][k]:
                    contacts[c]['contacts'][k].add(uid)
    return contacts


def show_layers(popdict, show_ages=False, show_n=20):
    """
    Print out the contacts for individuals in the different possible social settings or layers.

    Args:
        popdict (dict)   : dictionary of individuals with attributes, including their age and the ids of their close contacts
        show_ages (bool) : If True, show the ages of contacts, else show their ids
        show_n (int)     : number of individuals to show contacts for

    Returns:
        None
    """

    uids = popdict.keys()
    uids = [uid for uid in uids]

    layers = popdict[uids[0]]['contacts'].keys()
    if show_ages:
        for n, uid in enumerate(uids):
            if n >= show_n:
                break
            print('person', uid, 'age', popdict[uid]['age'])
            for k in layers:
                contact_ages = [popdict[c]['age'] for c in popdict[uid]['contacts'][k]]
                print('layer', k, 'contact ages', sorted(contact_ages))
            print()

    else:
        for n, uid in enumerate(uids):
            if n >= show_n:
                break
            print('person', uid, 'age', popdict[uid]['age'])
            for k in layers:
                print('layer', k, 'contact ids', popdict[uid]['contacts'][k])
            print()
