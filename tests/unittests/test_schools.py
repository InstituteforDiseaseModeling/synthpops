import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sciris as sc
import synthpops as sp
from scipy import stats
from collections import Counter


def test_generate_random_contacts_across_school(n=300, average_class_size=20):

    all_school_uids = np.arange(n)

    edges = sp.schools.generate_random_contacts_across_school(all_school_uids, average_class_size)
    G = nx.Graph()
    G.add_edges_from(edges)
    degree = [G.degree(i) for i in G.nodes()]
    print(Counter(degree))

    p = average_class_size/n
    print(p)

    # sp.check_dist(actual=degree, expected=average_class_size, dist='poisson', check='dist', verbose=True)
    # sp.check_dist(actual=degree, expected=)
    # rvs = stats.binom(n, p)
    r = stats.binom.rvs(n, p, size=len(all_school_uids))
    print(Counter(r))
    sp.check_dist(actual=degree, expected=(n, p), dist='binom', check='dist', verbose=True)


def data_for_add_school_edges(**kwargs):
    """Setting up the data to debug the different school graph generation methods"""
    default_kwargs = sc.objdict(n=300, average_class_size=20, min_age=5, max_age=10,
                                average_student_teacher_ratio=20,
                                average_student_all_staff_ratio=15,
                                average_additional_staff_degree=20,
                                average_teacher_teacher_degree=3,
                                inter_grade_mixing = 0.10, school_mixing_type = 'random',
                                )
    default_kwargs.grade_age_mapping = {i: i+5 for i in range(13)}
    default_kwargs.age_grade_mapping = {i+5: i for i in range(13)}
    default_kwargs.age_grade_mapping[3] = 0
    default_kwargs.age_grade_mapping[4] = 0
    kwargs = sc.objdict(sc.mergedicts(default_kwargs, kwargs))

    n = kwargs.n
    average_class_size = kwargs.average_class_size
    age_range = np.arange(kwargs.min_age, kwargs.max_age + 1)
    average_student_teacher_ratio = kwargs.average_student_teacher_ratio
    average_student_all_staff_ratio = kwargs.average_student_all_staff_ratio

    syn_school_uids = list(np.arange(n))
    syn_school_ages = list(np.random.randint(kwargs.min_age, kwargs.max_age + 1, size=n))

    age_by_uid_dic = dict(zip(syn_school_uids, syn_school_ages))

    n_teachers = int(np.ceil(len(syn_school_uids)/average_student_teacher_ratio))
    teachers = list(np.arange(syn_school_uids[-1] + 1, syn_school_uids[-1] + 1 + n_teachers))
    teacher_ages = list(np.random.randint(25, 75, size=n_teachers))

    n_all_staff = int(np.ceil(len(syn_school_uids)/average_student_all_staff_ratio))

    n_non_teaching_staff = max(n_all_staff - n_teachers, 1)
    non_teaching_staff = list(np.arange(teachers[-1] + 1, teachers[-1] + 1 + n_non_teaching_staff))
    non_teaching_staff_ages = list(np.random.randint(20, 75, size=n_non_teaching_staff))

    for ni, i in enumerate(teachers):
        age_by_uid_dic[i] = teacher_ages[ni]
    for ni, i in enumerate(non_teaching_staff):
        age_by_uid_dic[i] = non_teaching_staff_ages[ni]

    popdict = {}
    for i in age_by_uid_dic:
        popdict[i] = {}
        popdict[i]['age'] = int(age_by_uid_dic[i])
        popdict[i]['contacts'] = {}
        popdict[i]['scid'] = None
        popdict[i]['sc_student'] = None
        popdict[i]['sc_teacher'] = None
        popdict[i]['sc_staff'] = None
        if i in syn_school_uids:
            popdict[i]['sc_student'] = 1
        elif i in teachers:
            popdict[i]['sc_teacher'] = 1
        elif i in non_teaching_staff:
            popdict[i]['sc_staff'] = 1

    kwargs.syn_school_uids = syn_school_uids
    kwargs.syn_school_ages = syn_school_ages
    kwargs.age_by_uid_dic = age_by_uid_dic
    kwargs.n_teachers = n_teachers
    kwargs.teachers = teachers
    kwargs.teacher_ages = teacher_ages
    kwargs.n_all_staff = n_all_staff
    kwargs.n_non_teaching_staff = n_non_teaching_staff
    kwargs.non_teaching_staff = non_teaching_staff
    kwargs.non_teaching_staff_ages = non_teaching_staff_ages
    kwargs.popdict = popdict

    return kwargs


def debug_add_school_edges_random(kwargs):
    """Debug random school mixing type"""

    school = sc.dcp(kwargs.syn_school_uids)
    school += kwargs.teachers

    sc.tic()
    edges0 = sp.schools.generate_random_contacts_across_school(school, kwargs.average_class_size)
    school_with_staff = sc.dcp(kwargs.syn_school_uids) + sc.dcp(kwargs.teachers) + sc.dcp(kwargs.non_teaching_staff)
    additional_staff_edges = sp.schools.generate_random_contacts_for_additional_school_members(school_with_staff, kwargs.non_teaching_staff, kwargs.average_additional_staff_degree)
    all_edges0 = []
    all_edges0.extend(edges0)
    all_edges0.extend(additional_staff_edges)
    G0 = nx.Graph()
    G0.add_edges_from(all_edges0)
    degree0 = [G0.degree(i) for i in G0.nodes()]
    print("original synthpops method")
    print(f"number of edges: {len(G0.edges())}, average degree: {np.mean(degree0)}")
    sc.toc()

    sc.tic()
    school += kwargs.non_teaching_staff
    edges = sp.schools.generate_random_contacts_across_school(school, kwargs.average_class_size)
    all_edges = []
    all_edges.extend(edges)

    G = nx.Graph()
    G.add_edges_from(all_edges)
    degree = [G.degree(i) for i in G.nodes()]
    print("erdos_renyi_graph method")
    print(f"number of edges: {len(G.edges())}, average degree: {np.mean(degree)}")
    sc.toc()
    """
    Redesign/solution: use fast_gnp_random_graph() if
    average_additional_staff_degree can be obsolete here:
    
    When a school is completely random, should there be different
    characteristics for different members? In this case, students, teachers, and
    non teaching staff should have the same average degree. 

    Faster only when p is small, but will reduce run time in most cases
    students, teachers, and staff have no special relationship in terms of
    connections so create fast random graph for them (still need to  map to
    person ids for the generated graph for each school which does not take in a
    list of ids).

    On the other hand, if we want the average degree of students to be different
    from the average degree of teachers to be different from the average degree
    of non teaching staff, then in this case the best approach would be to use
    an ERGM. An ERGM would produce a random graph where different members can
    fit to different  characteristics. It won't be able to produce networks with
    more structure than typically randomly chosen connections from a pool of
    candidates, but could useful here to controlling degree.
    """
    sc.tic()
    print("fast_gnp_random_graph method")
    G2 = nx.fast_gnp_random_graph(len(school), p=kwargs.average_class_size/len(school), seed=kwargs.seed)
    degree2 = [G2.degree(i) for i in G2.nodes()]
    print(f"number of edges: {len(G2.edges())}, average degree: {np.mean(degree2)}")
    sc.toc()
    print()

    fig, ax = plt.subplots(1, 1)
    res = stats.probplot(degree0, dist=stats.poisson, sparams=(kwargs.average_class_size), plot=ax)
    # print("original res: ", res)
    res = stats.probplot(degree, dist=stats.poisson, sparams=(kwargs.average_class_size, ), plot=ax)
    # print("erdos_renyi_graph res:", res)
    res = stats.probplot(degree2, dist=stats.poisson, sparams=(kwargs.average_class_size, ), plot=ax)
    # print("fast_gnp_random_graph res: ", res)
    # plt.show()

    return G2


def debug_add_school_edges_age_clustered(kwargs):
    """Debug age clustered mixing type"""
    sc.tic()
    edges0 = sp.schools.generate_random_classes_by_grade_in_school(kwargs.syn_school_uids, kwargs.syn_school_ages,
        kwargs.age_by_uid_dic, kwargs.grade_age_mapping, kwargs.age_grade_mapping,
        # kwargs.average_class_size - len(kwargs.non_teaching_staff)/len(kwargs.non_teaching_staff), 
        kwargs.average_class_size,
        kwargs.inter_grade_mixing)
    teacher_edges0 = sp.schools.generate_edges_for_teachers_in_random_classes(kwargs.syn_school_uids, kwargs.syn_school_ages,
        kwargs.teachers, kwargs.age_by_uid_dic, kwargs.average_student_teacher_ratio,
        kwargs.average_teacher_teacher_degree)
    school_with_staff = sc.dcp(kwargs.syn_school_uids) + sc.dcp(kwargs.teachers) + sc.dcp(kwargs.non_teaching_staff)
    additional_staff_edges = sp.schools.generate_random_contacts_for_additional_school_members(school_with_staff, kwargs.non_teaching_staff, kwargs.average_additional_staff_degree)

    all_edges0 = []
    all_edges0.extend(edges0)
    all_edges0.extend(teacher_edges0)
    all_edges0.extend(additional_staff_edges)
    G0 = nx.Graph()
    G0.add_edges_from(all_edges0)

    degree0 = [G0.degree(i) for i in G0.nodes()]
    print("original synthpops method")
    print(f"number of edges: {len(G0.edges())}, average degree: {np.mean(degree0)}")
    print(len(edges0), len(teacher_edges0), len(additional_staff_edges), len(all_edges0))
    sc.toc()


    # redesign
    edges = sp.schools.generate_random_classes_by_grade_in_school(kwargs.syn_school_uids, kwargs.syn_school_ages, 
        kwargs.age_by_uid_dic, kwargs.grade_age_mapping, kwargs.age_grade_mapping,
        kwargs.average_class_size, kwargs.inter_grade_mixing)
    teacher_edges = sp.schools.generate_edges_for_teachers_in_random_classes(kwargs.syn_school_uids, kwargs.syn_school_ages,
        kwargs.teachers, kwargs.age_by_uid_dic, kwargs.average_student_teacher_ratio,
        kwargs.average_teacher_teacher_degree)
    # additional_staff_edges = sp.schools.generate_random_contacts_for_additional_school_members(school_with_staff, kwargs.non_teaching_staff, kwargs.average_additional_staff_degree)
    
    # for e in additional_staff_edges:
    #     i, j = e
    #     # print(e)
    #     if j in kwargs.syn_school_uids:
    #         print('y', j)
    #     if j in kwargs.teachers:
    #         print('yt', j)
        # if i kwargs.non_teaching_staff:
            # 
    G = nx.Graph()
    G.add_edges_from(edges)
    G.add_edges_from(teacher_edges)
    print(len(edges), len(teacher_edges), len(G.edges()))



if __name__ == '__main__':
    sp.set_seed(10)

    kwargs = data_for_add_school_edges(n=500, average_class_size=10, 
                                 average_additional_staff_degree=10,
                                 average_teacher_teacher_degree=3)

    # when average_class_size and average_additional_staff_degree are different,
    # the poisson test no longer passes

    # debug_add_school_edges_random(kwargs)
    for d in range(4):
        kwargs.seed=d
        debug_add_school_edges_random(kwargs)


    debug_add_school_edges_age_clustered(kwargs)

