"""
Script to examine the degree by edge type and role in schools.
"""

import os
import sciris as sc
import synthpops as sp
import numpy as np
import networkx as nx
from collections import Counter
import matplotlib as mplt
import matplotlib.pyplot as plt

# regenerate = False
regenerate = True
do_save = False

pars = dict(
        n                               = 22500,
        rand_seed                       = 123,
        max_contacts                    = None,
        # generate                      = True,

        country_location                = 'usa',
        state_location                  = 'Washington',
        location                        = 'seattle_metro',

        with_industry_code              = 0,
        with_facilities                 = 1,
        with_non_teaching_staff         = 1, # NB: has no effect
        use_two_group_reduction         = 1,
        with_school_types               = 1,

        average_LTCF_degree             = 20,
        ltcf_staff_age_min              = 20,
        ltcf_staff_age_max              = 60,

        # school_mixing_type              = 'age_and_class_clustered',
        school_mixing_type              = 'age_clustered',
        # school_mixing_type              = 'random',
        average_class_size              = 20,
        inter_grade_mixing              = 0.5, # NB: has no effect
        teacher_age_min                 = 25,
        teacher_age_max                 = 75,
        staff_age_min                   = 20,
        staff_age_max                   = 75,

        average_student_teacher_ratio   = 20,
        average_teacher_teacher_degree  = 3,
        average_student_all_staff_ratio = 15,
        average_additional_staff_degree = 20,
)

# for now save the pop object to the data folder
datadir = sp.datadir
output_file = os.path.join(datadir, f"school_testing_{pars['n']}.pop")

if regenerate:
    pop = sp.make_population(**pars)

    # save pop object
    if do_save:
        sc.saveobj(output_file, pop)

else:
    try:
        pop = sc.loadobj(output_file)
    except:
        raise ValueError(f"{output_file} doesn't exist yet. Set regenerate to True and do_save to True to regenerate the population and save it to file. Then you should be able to load it.")


students = {}
teachers = {}
staff = {}

schools = {}
school_ages = {}

# grab everyone in schools
for i in range(pars['n']):
    person = pop[i]
    for k in person['contacts']:
        person['contacts'][k] = np.array(person['contacts'][k])

    pop[i] = person

    scid = person['scid']
    if scid is not None:
        schools.setdefault(scid, {})
        schools[scid]['sc_type'] = person['sc_type']
        schools[scid].setdefault('pids', [])
        schools[scid]['pids'].append(i)

        school_ages.setdefault(scid, {})
        school_ages[scid]['sc_type'] = person['sc_type']
        school_ages[scid].setdefault('ages', [])
        school_ages[scid]['ages'].append(person['age'])

        if person['sc_teacher']:
            teachers[i] = person

        elif person['sc_staff']:
            staff[i] = person

        elif person['sc_student']:
            students[i] = person


student_ids = set(students.keys())
teacher_ids = set(teachers.keys())
staff_ids = set(staff.keys())
all_school_pids = student_ids.union(teacher_ids).union(staff_ids)
n_students = len(student_ids)
n_teachers = len(teacher_ids)
n_staff = len(staff_ids)

for scid in sorted(schools.keys()):
    schools[scid]['pids'] = np.asarray(schools[scid]['pids'])
    schools[scid]['n_students'] = len(set(schools[scid]['pids']).intersection(student_ids))
    schools[scid]['n_teachers'] = len(set(schools[scid]['pids']).intersection(teacher_ids))
    schools[scid]['n_staff'] = len(set(schools[scid]['pids']).intersection(staff_ids))

# sc.tic()

school_roles = ['student', 'teacher', 'staff']
school_edges = [f'{r1}_{r2}' for r1 in school_roles for r2 in school_roles]
school_e1 = [f'e1_{s}' for s in school_edges]
school_e2 = [f'e2_{s}' for s in school_edges]

# store edges and ids by school
for scid in sorted(schools.keys()):

    schools[scid].setdefault('e1', [])
    schools[scid].setdefault('e2', [])
    for sr in school_e1:
        schools[scid].setdefault(sr, [])
    for sr in school_e2:
        schools[scid].setdefault(sr, [])

    for ni, i in enumerate(schools[scid]['pids']):

        contacts = pop[i]['contacts']['S']
        schools[scid]['e1'].extend([i] * len(contacts))
        schools[scid]['e2'].extend(contacts)

# sc.toc()


G = nx.Graph()
G1 = nx.Graph()  # edges between students
G2 = nx.Graph()  # edges between students and teachers
G3 = nx.Graph()  # edges between students and staff
G4 = nx.Graph()  # edges between teachers
G5 = nx.Graph()  # edges between teachers and staff
G6 = nx.Graph()  # edges between staff

graphs = [G, G1, G2, G3, G4, G5, G6]

edge_types = ['all'] + [f'{school_roles[i]}_{school_roles[j]}' for i in range(len(school_roles)) for j in range(i, len(school_roles))]

sc.tic()

e1 = {sr: [] for sr in edge_types}
e2 = {sr: [] for sr in edge_types}

# only care about grade schools at the moment so don't include universities (school_type: 'uv')
grade_school_types = ['pk', 'es', 'ms', 'hs']
grade_school_students = {}
grade_school_teachers = {}
grade_school_staff = {}
for ni, i in enumerate(students.keys()):
    if students[i]['sc_type'] in grade_school_types:
        grade_school_students[i] = students[i]

        contacts = students[i]['contacts']['S']
        e1['all'].extend([i] * len(contacts))
        e2['all'].extend(contacts)

        student_contacts = list(set(contacts).intersection(student_ids))
        teacher_contacts = list(set(contacts).intersection(teacher_ids))
        staff_contacts = list(set(contacts).intersection(staff_ids))

        e1['student_student'].extend([i] * len(student_contacts))
        e2['student_student'].extend(student_contacts)

        e1['student_teacher'].extend([i] * len(teacher_contacts))
        e2['student_teacher'].extend(teacher_contacts)

        e1['student_staff'].extend([i] * len(staff_contacts))
        e2['student_staff'].extend(staff_contacts)

for ni, i in enumerate(teachers.keys()):
    if teachers[i]['sc_type'] in grade_school_types:
        grade_school_teachers[i] = teachers[i]

        contacts = teachers[i]['contacts']['S']
        e1['all'].extend([i] * len(contacts))
        e2['all'].extend(contacts)

        student_contacts = list(set(contacts).intersection(student_ids))
        teacher_contacts = list(set(contacts).intersection(teacher_ids))
        staff_contacts = list(set(contacts).intersection(staff_ids))

        e1['student_teacher'].extend(student_contacts)
        e2['student_teacher'].extend([i] * len(student_contacts))

        e1['teacher_teacher'].extend([i] * len(teacher_contacts))
        e2['teacher_teacher'].extend(teacher_contacts)

        e1['teacher_staff'].extend([i] * len(staff_contacts))
        e2['teacher_staff'].extend(staff_contacts)

for ni, i in enumerate(staff.keys()):
    if staff[i]['sc_type'] in grade_school_types:
        grade_school_staff[i] = staff[i]

        contacts = staff[i]['contacts']['S']
        e1['all'].extend([i] * len(contacts))
        e2['all'].extend(contacts)

        student_contacts = list(set(contacts).intersection(student_ids))
        teacher_contacts = list(set(contacts).intersection(teacher_ids))
        staff_contacts = list(set(contacts).intersection(staff_ids))

        e1['student_staff'].extend(student_contacts)
        e2['student_staff'].extend([i] * len(student_contacts))

        e1['teacher_staff'].extend(teacher_contacts)
        e2['teacher_staff'].extend([i] * len(teacher_contacts))

        e1['staff_staff'].extend([i] * len(staff_contacts))
        e2['staff_staff'].extend(staff_contacts)

sc.toc()

grade_school_student_ids = set(grade_school_students.keys())
grade_school_teacher_ids = set(grade_school_teachers.keys())
grade_school_staff_ids = set(grade_school_staff.keys())
grade_school_pids = grade_school_student_ids.union(grade_school_teacher_ids).union(grade_school_staff_ids)

grade_school_student_teacher_ids = grade_school_student_ids.union(grade_school_teacher_ids)
grade_school_student_staff_ids = grade_school_student_ids.union(grade_school_staff_ids)
grade_school_teacher_staff_ids = grade_school_teacher_ids.union(grade_school_staff_ids)

# add edges to different graphs by edge type
G.add_edges_from(zip(e1['all'], e2['all']))
G1.add_edges_from(zip(e1['student_student'], e2['student_student']))
G2.add_edges_from(zip(e1['student_teacher'], e2['student_teacher']))
G3.add_edges_from(zip(e1['student_staff'], e2['student_staff']))
G4.add_edges_from(zip(e1['teacher_teacher'], e2['teacher_teacher']))
G5.add_edges_from(zip(e1['teacher_staff'], e2['teacher_staff']))
G6.add_edges_from(zip(e1['staff_staff'], e2['staff_staff']))

G.add_nodes_from(grade_school_pids)
G1.add_nodes_from(grade_school_student_ids)
G2.add_nodes_from(grade_school_student_teacher_ids)
G3.add_nodes_from(grade_school_student_staff_ids)
G4.add_nodes_from(grade_school_teacher_ids)
G5.add_nodes_from(grade_school_teacher_staff_ids)
G6.add_nodes_from(grade_school_staff_ids)


# measure age mixing
density_or_frequency = 'density'
setting_code = 'S'

contact_matrix = sp.calculate_contact_matrix(pop, density_or_frequency, setting_code)
student_contact_matrix = contact_matrix[3:20, 3:20]
normed_contact_matrix = student_contact_matrix.copy()
for i in range(len(normed_contact_matrix)):
    normed_contact_matrix[i, :] = student_contact_matrix[i, :] / np.sum(student_contact_matrix[i, :])

# print(normed_contact_matrix[3:20, 3:20])

# plot matrix
# fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# ax.imshow(normed_contact_matrix)

# sp.plot_contacts(pop, setting_code, density_or_frequency=density_or_frequency, aggregate_flag=False, logcolors_flag=False, cmap='cmr.freeze_r', do_show=True)

# print statements

print(f"number of all schools: {len(schools)}")
print('mean school degree from dictionary method', np.sum([len(schools[scid]['e1']) for scid in schools if schools[scid]['sc_type'] != 'uv'])/np.sum([len(schools[scid]['pids']) for scid in schools if schools[scid]['sc_type'] != 'uv']))


print("\nSchool stats")
print(f"Total students: {len(student_ids)}, grade school students: {len(grade_school_students)}")
print(f"Total teachers: {len(teacher_ids)}, grade school teachers: {len(grade_school_teachers)}")
print(f"Total non-teaching staff: {len(staff_ids)}, grade school non-teaching staff: {len(grade_school_staff)}")
print(f"Total staff: {len(teacher_ids) + len(staff_ids)}, grade school staff: {len(grade_school_teachers) + len(grade_school_staff)}")

print("\nSchool degree stats")
for ng, g in enumerate(graphs):
    k = np.array([g.degree(i) for i in g.nodes()])
    mask = k > 0  # non zero degrees
    print(f"mean degree of {edge_types[ng]} edges: {np.mean(k):.4f}, non-zero degree {np.mean(k[mask]):.4f}")

print()
print(f"Student degree: { (2 * len(G1.edges()) + len(G2.edges()) + len(G3.edges()))/len(grade_school_students)}")
print(f"Teacher degree: { (len(G2.edges()) + 2 * len(G4.edges()) + len(G5.edges()))/len(grade_school_teachers)}")
print(f"Staff degree: { (len(G3.edges()) + len(G5.edges()) + 2 * len(G6.edges()))/len(grade_school_staff)}")

