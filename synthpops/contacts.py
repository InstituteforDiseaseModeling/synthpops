import sciris as sc
import numpy as np
import networkx as nx
from .base import *
from . import data_distributions as spdata
from . import sampling as spsamp
from .config import datadir
import os
import pandas as pd


def make_popdict(n=None, uids=None, ages=None, sexes=None, location=None, state_location=None, country_location=None, use_demography=False, id_len=6):
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

    min_people = 1000

    if location             is None: location = 'seattle_metro'
    if state_location is None: state_location = 'Washington'

    # A list of UIDs was supplied as the first argument
    if uids is not None:  # UIDs were supplied, use them
        n = len(uids)
    else:  # Not supplied, generate
        n = int(n)
        uids = []
        for i in range(n):
            uids.append(sc.uuid(length=id_len))

    # Check that there are enough people
    if n < min_people:
        print('Warning: with {n}<{min_people} people, contact matrices will be approximate')

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
                # raise NotImplementedError('Currently, only locations in the US are supported. Next version!')

    # you only have ages...
    elif ages is not None and sexes is None:
        if country_location == 'usa':
            if location is None: location, state_location = 'seattle_metro', 'Washington'
            gen_ages, gen_sexes = spsamp.get_usa_sex_n(datadir, ages, location=location, state_location=state_location, country_location=country_location)
        else:
            gen_ages = ages
            gen_sexes = list(np.random.binomial(1, p=0.5, size=n))
            # raise NotImplementedError('Currently, only locations in the US are supported.')

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
        popdict[uid] = {}
        popdict[uid]['age'] = ages[i]
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

    N = len(popdict)

    if network_type == 'poisson_degree':
        p = float(n_contacts)/N

        G = nx.erdos_renyi_graph(N, p, directed=directed)

    A = [a for a in G.adjacency()]

    for n, uid in enumerate(uids):
        # source_uid = uids[n]
        targets = [t for t in A[n][1].keys()]
        target_uids = [uids[target] for target in targets]
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

    uids_by_age_dic = get_uids_by_age_dic(popdict)
    age_bracket_distr = spdata.read_age_bracket_distr(datadir, location=location, state_location=state_location, country_location=country_location)
    age_brackets = spdata.get_census_age_brackets(datadir, state_location=state_location, country_location=country_location)
    num_agebrackets = len(age_brackets)
    age_by_brackets_dic = get_age_by_brackets_dic(age_brackets)

    age_mixing_matrix_dic = spdata.get_contact_matrix_dic(datadir, sheet_name=sheet_name)
    age_mixing_matrix_dic['M'] = combine_matrices(age_mixing_matrix_dic, n_contacts_dic, num_agebrackets)  # may need to normalize matrices before applying this method to K. Prem et al matrices because of the difference in what the raw matrices represent

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

    uids_by_age_dic = get_uids_by_age_dic(popdict)
    age_bracket_distr = spdata.read_age_bracket_distr(datadir, location=location, state_location=state_location, country_location=country_location)
    age_brackets = spdata.get_census_age_brackets(datadir, state_location=state_location, country_location=country_location)
    num_agebrackets = len(age_brackets)
    age_by_brackets_dic = get_age_by_brackets_dic(age_brackets)

    age_mixing_matrix_dic = spdata.get_contact_matrix_dic(datadir, sheet_name=sheet_name)
    age_mixing_matrix_dic['M'] = combine_matrices(age_mixing_matrix_dic, n_contacts_dic, num_agebrackets)  # may need to normalize matrices before applying this method to K. Prem et al matrices because of the difference in what the raw matrices represent

    n_contacts = network_distr_args['average_degree']
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

    # using a flat contact matrix
    uids_by_age_dic = spsamp.get_uids_by_age_dic(popdict)
    if country_location is None:
        raise NotImplementedError

    age_bracket_distr = spdata.read_age_bracket_distr(datadir, location=location, state_location=state_location, country_location=country_location)
    gender_fraction_by_age = spdata.read_gender_fraction_by_age_bracket(datadir, location=location, state_location=state_location, country_location=country_location)
    age_brackets = spdata.get_census_age_brackets(datadir, state_location=state_location, country_location=country_location)
    age_by_brackets_dic = get_age_by_brackets_dic(age_brackets)
    num_agebrackets = len(age_brackets)

    age_mixing_matrix_dic = spdata.get_contact_matrix_dic(datadir, sheet_name)
    age_mixing_matrix_dic['M'] = combine_matrices(age_mixing_matrix_dic, n_contacts_dic, num_agebrackets)  # may need to normalize matrices before applying this method to K. Prem et al matrices because of the difference in what the raw matrices represent

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
    uids_by_age_dic = get_uids_by_age_dic(popdict)
    if country_location is None:
        raise NotImplementedError

    age_bracket_distr = spdata.read_age_bracket_distr(datadir, location=location, state_location=state_location, country_location=country_location)
    gender_fraction_by_age = spdata.read_gender_fraction_by_age_bracket(datadir, location=location, state_location=state_location, country_location=country_location)
    age_brackets = spdata.get_census_age_brackets(datadir, state_location=state_location, country_location=country_location)
    age_by_brackets_dic = get_age_by_brackets_dic(age_brackets)
    num_agebrackets = len(age_brackets)

    age_mixing_matrix_dic = spdata.get_contact_matrix_dic(datadir, sheet_name)

    n_contacts = network_distr_args['average_degree']
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
<<<<<<< HEAD
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


def make_contacts_from_microstructure(datadir, location, state_location, country_location, n):
    """
    Make a popdict from synthetic household, school, and workplace files with uids.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in
        n (int)                   : number of people in the population

    Returns:
        A popdict of people with attributes, including their age and the ids of their contacts in the households ('H'),
        schools ('S'), and workplace ('W') layers where contacts are clustered and thus the network is composed of groups
        of people interacting with each other. For example, all household members are contacts of each other, and everyone
        at the same school is a contact of each other.
    """
    file_path = os.path.join(datadir, 'demographics', 'contact_matrices_152_countries', country_location, state_location, 'contact_networks')

    households_by_uid_path = os.path.join(file_path, location + '_' + str(n) + '_synthetic_households_with_uids.dat')
    age_by_uid_path = os.path.join(file_path, location + '_' + str(n) + '_age_by_uid.dat')

    workplaces_by_uid_path = os.path.join(file_path, location + '_' + str(n) + '_synthetic_workplaces_with_uids.dat')
    schools_by_uid_path = os.path.join(file_path, location + '_' + str(n) + '_synthetic_schools_with_uids.dat')

    df = pd.read_csv(age_by_uid_path, delimiter=' ', header=None)

    age_by_uid_dic = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
    uids = age_by_uid_dic.keys()

    # you have ages but not sexes so we'll just populate that for you at random
    popdict = {}
    for i, uid in enumerate(uids):
        popdict[uid] = {}
        popdict[uid]['age'] = int(age_by_uid_dic[uid])
        popdict[uid]['sex'] = np.random.binomial(1, p=0.5)
        popdict[uid]['loc'] = None
        popdict[uid]['contacts'] = {}
        for k in ['H', 'S', 'W', 'C']:
            popdict[uid]['contacts'][k] = set()

    fh = open(households_by_uid_path, 'r')
    for c, line in enumerate(fh):
        r = line.strip().split(' ')

        for uid in r:
            popdict[uid]['contacts']['H'] = set(r)
            popdict[uid]['contacts']['H'].remove(uid)
    fh.close()

    fs = open(schools_by_uid_path, 'r')
    for c, line in enumerate(fs):
        r = line.strip().split(' ')

        for uid in r:
            popdict[uid]['contacts']['S'] = set(r)
            popdict[uid]['contacts']['S'].remove(uid)
    fs.close()

    fw = open(workplaces_by_uid_path, 'r')
    for c, line in enumerate(fw):
        r = line.strip().split(' ')

        for uid in r:
            popdict[uid]['contacts']['W'] = set(r)
            popdict[uid]['contacts']['W'].remove(uid)
    fw.close()

    return popdict


def make_contacts_from_microstructure_objects(age_by_uid_dic, homes_by_uids, schools_by_uids, workplaces_by_uids):
    popdict = {}

    for uid in age_by_uid_dic:
        popdict[uid] = {}
        popdict[uid]['age'] = age_by_uid_dic[uid]
        popdict[uid]['sex'] = np.random.randint(2)
        popdict[uid]['loc'] = None
        popdict[uid]['contacts'] = {'H': set(), 'S': set(), 'W': set()}
        popdict[uid]['hhid'] = -1
        popdict[uid]['scid'] = -1
        popdict[uid]['wpid'] = -1

    for nh, household in enumerate(homes_by_uids):
        for uid in household:
            popdict[uid]['contacts']['H'] = set(household)
            popdict[uid]['contacts']['H'].remove(uid)
            popdict[uid]['hhid'] = nh

    for ns, school in enumerate(schools_by_uids):
        for uid in school:
            popdict[uid]['contacts']['S'] = set(school)
            popdict[uid]['contacts']['S'].remove(uid)
            popdict[uid]['scid'] = ns

    for nw, workplace in enumerate(workplaces_by_uids):
        for uid in workplace:
            popdict[uid]['contacts']['W'] = set(workplace)
            popdict[uid]['contacts']['W'].remove(uid)
            popdict[uid]['wpid'] = nw


    return popdict


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

    options_keys = ['use_age', 'use_sex', 'use_loc', 'use_social_layers', 'use_activity_rates', 'use_microstructure', 'use_age_mixing']
    if options_args         is None: options_args = dict.fromkeys(options_keys, False)

    # fill in the other keys as False!
    for key in options_keys:
        if key not in options_args:
            options_args[key] = False

    # to call in pre-generated contact networks that exhibit real-world-like clustering and age-specific mixing
    if options_args['use_microstructure']:
        if 'Npop' not in network_distr_args: network_distr_args['Npop'] = 10000
        country_location = 'usa'
        popdict = make_contacts_from_microstructure(datadir, location, state_location, country_location, network_distr_args['Npop'])

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

    if use_clusters:
        raise NotImplementedError("Clustered method not yet implemented.")

    else:
        for n, uid in enumerate(contacts):
            for k in keys:
                setting_contacts = contacts[uid]['contacts'][k]
                if len(setting_contacts) > trimmed_size_dic[k]/2:
                    close_contacts = np.random.choice(list(setting_contacts), size=int(trimmed_size_dic[k]/2))
                    contacts[uid]['contacts'][k] = set(close_contacts)

        for n, uid in enumerate(contacts):
            for k in keys:
                for c in contacts[uid]['contacts'][k]:
                    contacts[c]['contacts'][k].add(uid)

        test_sizes = True
        if test_sizes:

            for k in keys:
                sizes = []
                for n, uid in enumerate(contacts):
                    if len(contacts[uid]['contacts'][k]) > 0:
                        sizes.append(len(contacts[uid]['contacts'][k]))
                if verbose:
                    print(k, np.mean(sizes))

    return contacts


def show_layers(popdict, show_ages=False, show_n=20):
    """
    Print out the contacts for individuals in the different possible social settings or layers.

    Args:
        popdict (dict)   : dictionary of individuals with attributes, including their age and the ids of their close contacts
        show_ages (bool) : If True, show the ages of contacts, else show their ids

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
            print(uid, popdict[uid]['age'])
            for k in layers:
                contact_ages = [popdict[c]['age'] for c in popdict[uid]['contacts'][k]]
                print(k, sorted(contact_ages))

    else:
        for n, uid in enumerate(uids):
            if n >= show_n:
                break
            print(uid)
            for k in layers:
                print(k, popdict[uid]['contacts'][k])
