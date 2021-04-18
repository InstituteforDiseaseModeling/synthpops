"""
This module generates the household, school, and workplace contact networks.
"""

import sciris as sc
import numpy as np
import networkx as nx
from . import data_distributions as spdata
from . import schools as spsch
from .config import logger as log, checkmem


def make_contacts_from_microstructure_objects(age_by_uid_dic,
                                              homes_by_uids,
                                              schools_by_uids=None,
                                              teachers_by_uids=None,
                                              non_teaching_staff_uids=None,
                                              workplaces_by_uids=None,
                                              facilities_by_uids=None,
                                              facilities_staff_uids=None,
                                              use_two_group_reduction=False,
                                              average_LTCF_degree=20,
                                              with_school_types=False,
                                              school_mixing_type='random',
                                              average_class_size=20,
                                              inter_grade_mixing=0.1,
                                              average_student_teacher_ratio=20,
                                              average_teacher_teacher_degree=3,
                                              average_student_all_staff_ratio=15,
                                              average_additional_staff_degree=20,
                                              school_type_by_age=None,
                                              workplaces_by_industry_codes=None,
                                              max_contacts=None):
    """
    From microstructure objects (dictionary mapping ID to age, lists of lists in different settings, etc.), create a dictionary of individuals.
    Each key is the ID of an individual which maps to a dictionary for that individual with attributes such as their age, household ID (hhid),
    school ID (scid), workplace ID (wpid), workplace industry code (wpindcode) if available, and contacts in different layers.

    Args:
        age_by_uid_dic (dict)                             : dictionary mapping id to age for all individuals in the population
        homes_by_uids (list)                              : A list of lists where each sublist is a household and the IDs of the household members.
        schools_by_uids (list)                            : A list of lists, where each sublist represents a school and the ids of the students and teachers within it
        teachers_by_uids (list)                           : A list of lists, where each sublist represents a school and the ids of the teachers within it
        workplaces_by_uids (list)                         : A list of lists, where each sublist represents a workplace and the ids of the workers within it
        facilities_by_uids (list)                         : A list of lists, where each sublist represents a skilled nursing or long term care facility and the ids of the residents living within it
        facilities_staff_uids (list)                      : A list of lists, where each sublist represents a skilled nursing or long term care facility and the ids of the staff working within it
        non_teaching_staff_uids (list)                    : None or a list of lists, where each sublist represents a school and the ids of the non teaching staff within it
        use_two_group_reduction (bool)                    : If True, create long term care facilities with reduced contacts across both groups
        average_LTCF_degree (int)                         : default average degree in long term care facilities
        with_school_types (bool)                          : If True, creates explicit school types.
        school_mixing_type (str or dict)                  : The mixing type for schools, 'random', 'age_clustered', or 'age_and_class_clustered' if string, and a dictionary of these by school type otherwise. 'random' means random graphs for each school, 'age_clustered' means random graphs but with students mostly mixing within the age/grade (inter_grade_mixing controls mixing between grades), 'age_and_grade_clustered' means students cohorted into classes with their own teachers.
        average_class_size (float)                        : The average classroom size.
        inter_grade_mixing (float)                        : The average fraction of mixing between grades in the same school for clustered school mixing types.
        average_student_teacher_ratio (float)             : The average number of students per teacher.
        average_teacher_teacher_degree (float)            : The average number of contacts per teacher with other teachers.
        average_student_all_staff_ratio (float)           : The average number of students per staff members at school (including both teachers and non teachers).
        average_additional_staff_degree (float)           : The average number of contacts per additional non teaching staff in schools.
        school_type_by_age (dict)                         : A dictionary of probabilities for the school type likely for each age.
        workplaces_by_industry_codes (np.ndarray or None) : array with workplace industry code for each workplace
        trimmed_size_dic (dict)                           : If supplied, trim contacts on creation rather than post hoc.

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

        If with_school_types==False, completely random schools will be generated with respect to the average_class_size,
        but other parameters such as average_additional_staff_degree will not be used.
    """
    log.debug('make_contacts_from_microstructure_objects()')
    popdict = {}

    grade_age_mapping = {i: i+5 for i in range(13)}
    age_grade_mapping = {i+5: i for i in range(13)}
    age_grade_mapping[3] = 0
    age_grade_mapping[4] = 0

    # what are the school types by age
    school_type_by_age = sc.mergedicts(spdata.get_default_school_types_by_age_single(), school_type_by_age)
    school_types = list(set(school_type_by_age.values()))  # get the location specific school types whatever they may be

    # check school mixing type
    if isinstance(school_mixing_type, str):
        school_mixing_type_dic = dict.fromkeys(school_types, school_mixing_type)
    elif isinstance(school_mixing_type, dict):
        school_mixing_type_dic = sc.dcp(school_mixing_type)
        school_mixing_type_dic = sc.mergedicts(dict.fromkeys(school_types, 'random'), school_mixing_type_dic)  # if the dictionary given doesn't specify the mixing type for an expected school type, set the mixing type for that school type to random by default

    uids = age_by_uid_dic.keys()
    uids = [uid for uid in uids]

    popdict = {}

    # Handle trimming
    do_trim = max_contacts is not None
    max_contacts = sc.mergedicts({'W': 20}, max_contacts)
    trim_keys = max_contacts.keys()

    # Handle LTCF
    use_ltcf = facilities_by_uids is not None
    if use_ltcf:
        layer_keys = ['H', 'S', 'W', 'C', 'LTCF']
    else:
        layer_keys = ['H', 'S', 'W', 'C']

    log.debug('  starting...' + checkmem())

    # TODO: include age-based sex ratios
    sexes = np.random.randint(2, size=len(age_by_uid_dic))


    for u, uid in enumerate(age_by_uid_dic):
        popdict[uid] = {}
        popdict[uid]['age'] = int(age_by_uid_dic[uid])
        popdict[uid]['sex'] = sexes[u]
        popdict[uid]['loc'] = None
        popdict[uid]['contacts'] = {}
        if use_ltcf:
            popdict[uid]['snf_res'] = None
            popdict[uid]['snf_staff'] = None
        popdict[uid]['hhid'] = None
        popdict[uid]['scid'] = None
        popdict[uid]['sc_student'] = None
        popdict[uid]['sc_teacher'] = None
        popdict[uid]['sc_staff'] = None
        popdict[uid]['sc_type'] = None
        popdict[uid]['sc_mixing_type'] = None
        popdict[uid]['wpid'] = None
        popdict[uid]['wpindcode'] = None
        if use_ltcf:
            popdict[uid]['snfid'] = None
        for k in layer_keys:
            popdict[uid]['contacts'][k] = set()

    # read in facility residents and staff
    if use_ltcf:
        for nf, facility in enumerate(facilities_by_uids):
            facility_staff = facilities_staff_uids[nf]

            for u in facility:
                popdict[u]['snf_res'] = 1
                popdict[u]['snfid'] = nf

            for u in facility_staff:
                popdict[u]['snf_staff'] = 1
                popdict[u]['snfid'] = nf

            if use_two_group_reduction:
                popdict = create_reduced_contacts_with_group_types(popdict, facility, facility_staff, 'LTCF',
                                                                   average_degree=average_LTCF_degree,
                                                                   force_cross_edges=True)

            else:
                log.debug('...LTCFs ' + checkmem())
                for uid in facility:
                    popdict[uid]['contacts']['LTCF'] = set(facility)
                    popdict[uid]['contacts']['LTCF'] = popdict[uid]['contacts']['LTCF'].union(set(facility_staff))
                    popdict[uid]['contacts']['LTCF'].remove(uid)

                for uid in facility_staff:
                    popdict[uid]['contacts']['LTCF'] = set(facility)
                    popdict[uid]['contacts']['LTCF'] = popdict[uid]['contacts']['LTCF'].union(set(facility_staff))
                    popdict[uid]['contacts']['LTCF'].remove(uid)


    log.debug('...households ' + checkmem())
    for nh, household in enumerate(homes_by_uids):
        for uid in household:
            popdict[uid]['contacts']['H'] = set(household)
            popdict[uid]['contacts']['H'].remove(uid)
            popdict[uid]['hhid'] = nh


    log.debug('...students ' + checkmem())


    for ns, students in enumerate(schools_by_uids):
        teachers = teachers_by_uids[ns]
        if non_teaching_staff_uids is None:
            non_teaching_staff = []
        elif non_teaching_staff_uids == []:
            non_teaching_staff = []
        else:
            non_teaching_staff = non_teaching_staff_uids[ns]

        this_school_type = None
        this_school_mixing_type = None

        if with_school_types:
            student_ages = [age_by_uid_dic[i] for i in students]
            min_age = min(student_ages)
            this_school_type = school_type_by_age[min_age]
            this_school_mixing_type = school_mixing_type_dic[this_school_type]
            spsch.add_school_edges(popdict, students, student_ages, teachers, non_teaching_staff, age_by_uid_dic, grade_age_mapping, age_grade_mapping, average_class_size, inter_grade_mixing, average_student_teacher_ratio, average_teacher_teacher_degree, average_additional_staff_degree, this_school_mixing_type)
        else:
            school = students.copy() + teachers.copy() + non_teaching_staff.copy()
            school_edges = spsch.generate_random_contacts_across_school(school, average_class_size)
            spsch.add_contacts_from_edgelist(popdict, school_edges, 'S')

        for uid in students:
            popdict[uid]['scid'] = ns
            popdict[uid]['sc_student'] = 1
            popdict[uid]['sc_type'] = this_school_type
            popdict[uid]['sc_mixing_type'] = this_school_mixing_type

        for uid in teachers:
            popdict[uid]['scid'] = ns
            popdict[uid]['sc_teacher'] = 1
            popdict[uid]['sc_type'] = this_school_type
            popdict[uid]['sc_mixing_type'] = this_school_mixing_type

        for uid in non_teaching_staff:
            popdict[uid]['scid'] = ns
            popdict[uid]['sc_staff'] = 1
            popdict[uid]['sc_type'] = this_school_type
            popdict[uid]['sc_mixing_type'] = this_school_mixing_type


    log.debug('...workplaces ' + checkmem())
    if do_trim and 'W' in trim_keys:
        max_W_size = int(max_contacts['W'] // 2)  # Divide by 2 since bi-directional contacts get added in later

        # Loop over workplaces but only generate the requested contacts
        for nw, workplace in enumerate(workplaces_by_uids):
            for uid in workplace:
                uids = set(workplace)
                uids.remove(uid)
                if len(uids) > max_W_size:
                    uids = np.random.choice(list(uids), size=max_W_size, replace=False)
                popdict[uid]['contacts']['W'] = set(uids)
                popdict[uid]['wpid'] = nw
                if workplaces_by_industry_codes is not None:
                    popdict[uid]['wpindcode'] = int(workplaces_by_industry_codes[nw])

        # Add pairing contacts back in
        for uid in popdict.keys():
            for c in popdict[uid]['contacts']['W']:
                popdict[c]['contacts']['W'].add(uid)
    else:
        for nw, workplace in enumerate(workplaces_by_uids):
            for uid in workplace:
                popdict[uid]['contacts']['W'] = set(workplace)
                popdict[uid]['contacts']['W'].remove(uid)
                popdict[uid]['wpid'] = nw
                if workplaces_by_industry_codes is not None:
                    popdict[uid]['wpindcode'] = int(workplaces_by_industry_codes[nw])

    log.debug('...done ' + checkmem())
    return popdict


def create_reduced_contacts_with_group_types(popdict, group_1, group_2, setting, average_degree=20, p_matrix=None, force_cross_edges=True):
    """
    Create contacts between members of group 1 and group 2, fixing the average degree, and the
    probability of an edge between any two groups controlled by p_matrix if provided.
    Forces inter group edge for each individual in group 1 with force_cross_groups equal to True.
    This means not everyone in group 2 will have a contact with group 1.

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
                ncut = len(group_1_neighbors) - average_degree  # rough number to cut
                # ncut = spsamp.pt(ncut)  # sample from poisson that number
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


def get_contact_counts_by_layer(popdict,
                                layer='S'):
    """
    Method to count the number of contacts for individuals in the population
    based on their role in a layer and the role of their contacts. For example,
    in schools this method can distinguish the number of contacts between
    students, teachers, and non teaching staff in the population, as well as
    return the number of contacts between all individuals present in a school.
    In a population with a school layer and roles defined as students, teachers,
    and non teaching staff, this method will return the number of contacts or
    edges for sc_students, sc_teachers, and sc_staff to sc_student, sc_teacher,
    sc_staff, all_staff, all. all_staff is the combination of sc_teacher and
    sc_staff, and all is all kinds of people in schools.

    Args:
        popdict (dict)  : popdict of a Pop object, Dictionary keys are the IDs of individuals in the population and the values are a dictionary
        layer (str)     : name of the physial contact layer: H for households, S for schools, W for workplaces, C for community, etc.

    Returns:
        dict: A dictionary with keys = people_types (default to ['sc_student',
        'sc_teacher', 'sc_staff']) and each value is a dictionary which stores
        the list of counts for each type of contact: default to ['sc_student',
        'sc_teacher', 'sc_staff', 'all_staff', 'all'] for example:
        contact_counter['sc_teacher']['sc_teacher'] store the counts of each
        teacher's contacts or edges to other teachers.
    """
    layer = layer.upper()
    layer_keys = {"S": "scid",
                  "W": "wpid",
                  "H": "hhid",
                  "LTCF": "snfid"}

    if layer == 'S':
        people_types = ['sc_student', 'sc_teacher', 'sc_staff']
        contact_types = people_types + ['all_staff', 'all']
        contact_counter = {k: dict(zip(contact_types, ([] for _ in contact_types))) for k in
                           dict.fromkeys(people_types)}
        # index_switcher is a case-switch selector for the person selected by its type
        index_switcher = {
            'sc_student': contact_counter['sc_student'],
            'sc_teacher': contact_counter['sc_teacher'],
            'sc_staff': contact_counter['sc_staff']
        }
    elif layer in ["W", "H", "LTCF"]:
        people_types = [layer_keys[layer]]
        contact_types = ['all']
        contact_counter = {k: dict(zip(contact_types, ([] for _ in contact_types))) for k in
                           dict.fromkeys(people_types)}
        index_switcher = {
            layer_keys[layer]: contact_counter[layer_keys[layer]]
        }
    else:
        raise NotImplementedError(f"layer {layer} not supported.")

    for uid, person in popdict.items():
        if person[layer_keys[layer]] is not None:
            # count_switcher is a case-switch selector for contact counts by type
            count_switcher = {
                'sc_student': len([c for c in person["contacts"]["S"] if popdict[c]['sc_student']]),
                'sc_teacher': len([c for c in person["contacts"]["S"] if popdict[c]['sc_teacher']]),
                'sc_staff': len([c for c in person["contacts"]["S"] if popdict[c]['sc_staff']]),
                'all': len([c for c in person["contacts"][layer]])
            }
            for k1 in people_types:
                # if this person does not belong to a particular key, we don't need to store the counts under this key
                if person.get(k1) is not None:
                    # store sc_teacher, sc_student, sc_staff, all_staff and all below
                    if layer == "S":
                        for k2 in people_types:
                            index_switcher.get(k1)[k2].append(count_switcher.get(k2))
                        index_switcher.get(k1)["all_staff"].append(
                            count_switcher.get('sc_teacher') + count_switcher.get('sc_staff'))
                    # for other types, only all contacts are stored
                    index_switcher.get(k1)["all"].append(count_switcher.get('all'))

    return contact_counter


def filter_people(pop, ages=None, uids=None):
    """
    Helper function to filter people based on their uid and age.
    """
    uid_mask = Ellipsis
    if uids is not None:
        uid_mask = [True if i in uids else False for i in range(pop.n)]

    age_mask = Ellipsis
    if ages is not None:
        age_mask = [True if a in ages else False for a in pop.age_by_uid]

    mask = Ellipsis
    if uid_mask is Ellipsis:
        mask = np.array(age_mask)
    elif age_mask is Ellipsis:
        mask = np.array(uid_mask)
    elif uid_mask is not Ellipsis and age_mask is not Ellipsis:
        mask = np.multiply(uid_mask, age_mask)
        mask = np.array(mask)

    return np.arange(pop.n)[mask]


def count_layer_degree(pop, layers='H', ages=None, uids=None, uids_included=None):

    if uids_included is None:
        uids_included = filter_people(pop, ages, uids)

    layers = sc.tolist(layers)

    degree = [[] for i in range(pop.max_age)]
    for i in uids_included:
        a = pop.age_by_uid[i]
        nc = 0
        ca = []
        for layer in layers:
            nc += len(pop.popdict[i]['contacts'][layer])
            ca.extend([pop.age_by_uid[j] for j in pop.popdict[i]['contacts'][layer]])

        # print(i, a, nc, ca)
        degree[a].append(nc)

    # for a in ages:
        # print(a, degree[a])
    print(len(degree))
    return degree


def compute_layer_degree_statistics(pop, layers='H', ages=None, uids=None, uids_included=None, degree=None, alpha=0.05):

    if degree is None:
        degree = count_layer_degree(pop, layers, ages, uids, uids_included)

    # stats = [[] for a in range(pop.n)]
    stats = {}
    stats['mean'] = np.array([np.mean(degree[a]) for a in degree])




