"""
This module generates school contacts by class and grade in flexible ways.
Contacts can be clustered into classes and also mixed across the grade and
across the school.

H. Guclu et. al (2016) shows that mixing across grades is low for public schools
in elementary and middle schools. Mixing across grades is however higher in high
schools.

Functions in this module are flexible to allow users to specify the inter-grade
mixing (for 'age_clustered' school_mixing_type), and to choose whether contacts
are clustered within a grade. Clustering contacts across different grades is not
supported because there is no data to suggest that this happens commonly.
"""
from collections import Counter
from itertools import combinations

import sciris as sc
import numpy as np
import networkx as nx
import logging

from . import data_distributions as spdata
from . import defaults

from . import base as spb
from . import sampling as spsamp
from . import contact_networks as spcnx
from .config import logger as log


__all__ = ['get_school_type_labels', 'count_enrollment_by_school_type',
           'get_generated_school_size_distributions', 'count_enrollment_by_age',
           'get_enrollment_rates_by_age',
           'School',
           'Classroom',
           ]


class School(spb.LayerGroup):
    """
    A class for individual schools and methods to operate on each.

    Args:
        kwargs (dict): data dictionary of the school
    """
    def __init__(self, scid=None, sc_type=None, school_mixing_type=None,
                 student_uids=np.array([], dtype=int), teacher_uids=np.array([], dtype=int),
                 non_teaching_staff_uids=np.array([], dtype=int), **kwargs):
        """
        Class constructor for an base empty setting group.

        Args:
            **scid (int)                         : id of the school
            **sc_type (str)                      : school type defined by grade/age ranges
            **school_mixing_type (str)           : the mixing type of the school, 'random', 'age_clustered', or 'age_and_class_clustered' if str. Else, None. See sp.schools.add_school_edges() for more information.
            **student_uids (np.array)            : ids of student members
            **teacher_uids (np.array)            : ids of teacher members
            **non_teaching_staff_uids (np.array) : ids of non_teaching_staff members
        """
        super().__init__(scid=scid, sc_type=sc_type, school_mixing_type=school_mixing_type,
                         student_uids=student_uids, teacher_uids=teacher_uids,
                         non_teaching_staff_uids=non_teaching_staff_uids, **kwargs)
        self.validate()

        return

    def validate(self):
        """
        Check that information supplied to make a school is valid and update
        to the correct type if necessary.
        """
        for key in ['student_uids', 'teacher_uids', 'non_teaching_staff_uids']:
            if key in self.keys():
                try:
                    self[key] = sc.promotetoarray(self[key], dtype=int)
                except:
                    errmsg = f"Could not convert school key {key} to an np.array() with type int. This key only takes arrays with int values."
                    raise TypeError(errmsg)

        for key in ['scid']:
            if key in self.keys():
                if not isinstance(self[key], (int)):
                    if self[key] is not None:
                        errmsg = f"Error: Expected type int or None for school key {key}. Instead the type of this value is {type(self[key])}."
                        raise TypeError(errmsg)

        for key in ['sc_type']:
            if key in self.keys():
                if not isinstance(self[key], str):
                    if self[key] is not None:
                        errmsg = f"Error: Expected type str or None school key {key}."
                        raise TypeError(errmsg)
        return

    @property
    def member_uids(self):
        """
        Return ids of all school members: students, teachers, and non teaching staff.

        Returns:
            np.ndarray : school member ids

        """
        return np.concatenate((self['student_uids'], self['teacher_uids'], self['non_teaching_staff_uids']))

    def member_ages(self, age_by_uid):
        """
        Return ages of all school members: students, teachers, and non teaching staff.

        Args:
            age_by_uid (np.ndarray) : mapping of age to uid

        Returns:
            np.ndarray: school member ages
        """
        return np.concatenate((self.student_ages(age_by_uid),
                               self.teacher_ages(age_by_uid),
                               self.non_teaching_staff_ages(age_by_uid)))

    def student_ages(self, age_by_uid):
        """
        Return student ages in the school.

        Args:
            age_by_uid (np.ndarray) : mapping of age to uid

        Returns:
            np.ndarray : student ages in school
        """
        return super().member_ages(age_by_uid, self['student_uids'])

    def teacher_ages(self, age_by_uid):
        """
        Return teacher ages in the school.

        Args:
            age_by_uid (np.ndarray) : mapping of age to uid

        Returns:
            np.ndarray : teacher ages in school
        """
        return super().member_ages(age_by_uid, self['teacher_uids'])

    def non_teaching_staff_ages(self, age_by_uid):
        """
        Return non-teaching staff ages in the school.

        Args:
            age_by_uid (np.ndarray) : mapping of age to uid

        Returns:
            np.ndarray : non-teaching staff ages in school
        """
        return super().member_ages(age_by_uid, self['non_teaching_staff_uids'])

    def __len__(self):
        """Return the length as the number of members in the school."""
        return len(self.member_uids)

    def get_classroom(self, clid):
        """
        Return the classroom indexed at clid if school_mixing_type is equal to
        'age_and_class_clustered'.

        Args:
            clid (int) : classroom id number

        Returns:
            sp.Classroom : the classroom indexed at clid
        """
        if self['school_mixing_type'] == 'age_and_class_clustered':
            if not isinstance(clid, int):
                raise TypeError("clid must be an int.")
            if len(self['classrooms']) <= clid:
                raise IndexError(f"Classroom id (clid): {clid} out of range.")
            return self['classrooms'][clid]
        else:
            return


class Classroom(spb.LayerGroup):
    """
    A class for individual classrooms and methods to operate on each.

    Args:
        kwargs (dict): data dictionary of the classroom
    """

    def __init__(self, clid=None, student_uids=np.array([], dtype=int), teacher_uids=np.array([], dtype=int), **kwargs):
        """
        Class constructor for an base empty setting group.

        Args:
            **clid (int)              : id of the classroom
            **student_uids (np.array) : ids of student members
            **teacher_uids (np.array) : ids of teacher members
        """
        super().__init__(clid=clid, student_uids=student_uids, teacher_uids=teacher_uids, **kwargs)

        self.validate()

        return

    def validate(self):
        """
        Check that information supplied to make a school is valid and update
        to the correct type if necessary.
        """
        for key in ['student_uids', 'teacher_uids']:
            if key in self.keys():
                try:
                    self[key] = sc.promotetoarray(self[key], dtype=int)
                except:
                    errmsg = f"Could not convert classroom key {key} to a np.array()"
                    raise TypeError(errmsg)

        for key in ['clid']:
            if key in self.keys():
                if not isinstance(self[key], int):
                    if self[key] is not None:
                        errmsg = f"Error: Expected type int or None for classroom key {key}."
                        raise TypeError(errmsg)
        return

    @property
    def member_uids(self):
        """
        Return ids of all classroom members: students and teachers.

        Returns:
            np.ndarray : classroom member ids
        """
        return np.concatenate((self['student_uids'], self['teacher_uids']))

    def member_ages(self, age_by_uid):
        """
        Return ages of all classroom members: students and teachers.

        Args:
            age_by_uid (np.ndarray) : mapping of age to uid

        Returns:
            np.ndarray : classroom member ages
        """
        return np.concatenate((self.student_ages(age_by_uid),
                               self.teacher_ages(age_by_uid)))

    def student_ages(self, age_by_uid):
        """
        Return student ages in the classroom.

        Args:
            age_by_uid (np.ndarray) : mapping of age to uid

        Returns:
            np.ndarray : student ages in classroom
        """
        return super().member_ages(age_by_uid, self['student_uids'])

    def teacher_ages(self, age_by_uid):
        """
        Return teacher ages in the classroom.

        Args:
            age_by_uid (np.ndarray) : mapping of age to uid

        Returns:
            np.ndarray : teacher ages in classroom
        """
        return super().member_ages(age_by_uid, self['teacher_uids'])

    def __len__(self):
        """Return the length as the number of members in the classroom."""
        return len(self.member_uids)


def get_school(pop, scid):
    """
    Return school with id: scid.

    Args:
        pop (sp.Pop) : population
        scid (int)   : school id number

    Returns:
        sp.School: A populated school.
    """
    if not isinstance(scid, int):
        raise TypeError(f"scid must be an int.")
    if len(pop.schools) <= scid:
        raise IndexError(f"School id (scid): {scid} out of range.")
    return pop.schools[scid]


def get_classroom(pop, scid, clid):
    """
    Return the classroom indexed at clid if school_mixing_type is equal to
    'age_and_class_clustered'.

    Args:
        pop (sp.Pop) : population
        scid (int)   : school id number

    Returns:
        sp.Classroom: A populated classroom.
    """
    school = get_school(pop, scid)
    return school.get_classroom(clid)


def add_school(pop, school):
    """
    Add a school to the list of schools.

    Args:
        pop (sp.Pop) : population
        school (sp.School) : school
    """
    if not isinstance(school, School):
        raise ValueError('school is not a sp.School')

    # ensure scid to match the index in the list
    if school['scid'] != len(pop.schools):
        school['scid'] = len(pop.schools)
    pop.schools.append(school)
    pop.n_schools = len(pop.schools)
    return


def add_classroom(school, classroom):
    """
    Add a classroom to the school.

    Args:
        school (sp.School)       : school
        classroom (sp.Classroom) : classroom
    """
    if not isinstance(school, School):
        raise ValueError('school is not a sp.School')

    if not isinstance(classroom, Classroom):
        raise ValueError('classroom is not a sp.Classroom')

    # ensure scid to match the index in the list
    if classroom['scid'] != len(school['classrooms']):
        school['scid'] = len(school['classrooms'])
    school['classrooms'].append(classroom)
    school['n_classrooms'] = len(school['classrooms'])
    return


def initialize_empty_schools(pop, n_schools=None):
    """
    Array of empty schools.

    Args:
        pop (sp.Pop)    : population
        n_schools (int) : the number of schools to initialize
    """
    if n_schools is not None and isinstance(n_schools, int):
        pop.n_schools = n_schools
    else:
        pop.n_schools = 0
    pop.schools = [School() for ns in range(pop.n_schools)]
    return


def initialize_empty_classrooms(school, n_classrooms=None):
    """
    Array of empty classrooms.

    Args:
        school (sp.School) : school
        n_classrooms (int) : the number of classrooms to initialize
    """
    if school['school_mixing_type'] == 'age_and_class_clustered':
        if n_classrooms is not None and isinstance(n_classrooms, int):
            school['n_classrooms'] = n_classrooms
        else:
            school['n_classrooms'] = 0
        school['classrooms'] = [Classroom() for nc in range(school['n_classrooms'])]

    return


def populate_schools(pop, student_lists, teacher_lists, non_teaching_staff_lists, age_by_uid, school_types=None, school_mixing_types=None):
    """
    Populate all of the schools. Store each school at the index corresponding to it's scid.

    Args:
        pop (sp.Pop)                    : population
        student_lists (list)            : list of lists where each sublist represents a school and contains the ids of the students
        teacher_lists (list)            : list of lists where each sublist represents a school and contains the ids of the teachers
        non_teaching_staff_lists (list) : list of lists where each sublist represents a school and contains the ids of the non teaching staff
        age_by_uid (dict)               : dictionary mapping each person's id to their age
        school_types (list)             : list of the school types
        school_mixing_types (list)      : list of the school mixing types
    """
    initialize_empty_schools(pop, len(student_lists))

    log.debug("Populating schools.")

    if school_types is None:
        school_types = [None for ns in range(len(student_lists))]

    if school_mixing_types is None:
        school_mixing_types = [None for ns in range(len(student_lists))]

    for ns in range(len(student_lists)):
        students = student_lists[ns]
        teachers = teacher_lists[ns]
        non_teaching_staff = non_teaching_staff_lists[ns]
        sc_type = school_types[ns]
        school_mixing_type = school_mixing_types[ns]

        kwargs = dict(scid=ns,
                      sc_type=sc_type,
                      school_mixing_type=school_mixing_type,
                      student_uids=students,
                      teacher_uids=teachers,
                      non_teaching_staff_uids=non_teaching_staff,
                      )
        school = School()
        school.set_layer_group(**kwargs)
        pop.schools[school['scid']] = sc.dcp(school)

    return


def populate_classrooms(school, student_lists, teacher_lists, age_by_uid):
    """
    Populate all of the classrooms in a school if
    school_mixing_type == 'age_and_class_clustered'. Store each school at the
    index corresponding to it's scid.

    Args:
        school (sp.School)   : school
        student_lists (list) : list of lists where each sublist represents a classroom and contains the ids of the students
        teacher_lists (list) : list of lists where each sublist represents a classroom and contains the ids of the teachers
        age_by_uid (dict)    : dictionary mapping each person's id to their age
    """
    if school['school_mixing_type'] == 'age_and_class_clustered':
        if len(school['classrooms']) < len(student_lists):
            log.debug(f"Reinitializing list of classrooms")
            initialize_empty_classrooms(school, len(student_lists))

        log.debug("Populating classrooms.")

        for nc in range(len(student_lists)):
            students = student_lists[nc]
            teachers = teacher_lists[nc]

            kwargs = dict(clid=nc,
                          student_uids=students,
                          teacher_uids=teachers,
                          )
            classroom = Classroom()
            classroom.set_layer_group(**kwargs)
            school['classrooms'][classroom['clid']] = sc.dcp(classroom)
    return


def get_school_type_labels():
    school_type_labels = {'pk': 'Pre-school', 'es': 'Elementary School',
                          'ms': 'Middle School', 'hs': 'High School',
                          'uv': 'University'}
    return school_type_labels


def get_uids_in_school(datadir, n, location, state_location, country_location, age_by_uid=None, homes_by_uids=None, folder_name=None, use_default=False):
    """
    Identify who in the population is attending school based on enrollment rates
    by age.

    Args:
        datadir (string)          : The file path to the data directory.
        n (int)                   : The number of people in the population.
        location (string)         : The name of the location.
        state_location (string)   : The name of the state the location is in.
        country_location (string) : The name of the country the location is in.
        age_by_uid (dict)         : A dictionary mapping ID to age for all individuals in the population.
        homes_by_uids (list)      : A list of lists where each sublist is a household and the IDs of the household members.
        folder_name (string)      : The name of the folder the location is in, e.g. 'contact_networks'
        use_default (bool)        : If True, try to first use the other parameters to find data specific to the location under study; otherwise, return default data drawing from default_location, default_state, default_country.

    Returns:
        A dictionary of students in schools mapping their ID to their age, a
        dictionary of students in school mapping age to the list of IDs with
        that age, and a dictionary mapping age to the number of students with
        that age.
    """
    uids_in_school = {}
    uids_in_school_by_age = {}
    ages_in_school_count = dict.fromkeys(np.arange(101), 0)

    rates = spdata.get_school_enrollment_rates(datadir, location=location, state_location=state_location, country_location=country_location, use_default=use_default)

    for a in np.arange(101):
        uids_in_school_by_age[a] = []

    # go through homes and make a list of uids going to school as students, this should preserve ordering of students by homes and so create schools with siblings going to the same school
    for home in homes_by_uids:
        for uid in home:

            a = age_by_uid[uid]
            if rates[a] > 0:
                b = np.random.binomial(1, rates[a])  # ask each person if they'll be a student - probably could be done in a faster, more aggregate way.
                if b:
                    uids_in_school[uid] = a
                    uids_in_school_by_age[a].append(uid)
                    ages_in_school_count[a] += 1

    return uids_in_school, uids_in_school_by_age, ages_in_school_count


def send_students_to_school_with_school_types(school_size_distr_by_type, school_size_brackets, uids_in_school, uids_in_school_by_age, ages_in_school_count, school_types_distr_by_age, school_type_age_ranges):
    """
    A method to send students to school together. This method uses the
    dictionaries school_types_distr_by_age, school_type_age_ranges, and
    school_size_distr_by_type to first determine the type of school based on the
    age of a sampled reference student. Then the school type is used to
    determine the age range of the school. After that, the size of the school is
    then sampled conditionally on the school type and then the rest of the
    students are chosen from the lists of students available in the dictionary
    uids_in_school_by_age. This method is not perfect and requires a strict
    definition of school type by age. For now, it is not able to model mixed
    school types such as schools with Kindergarten through Grade 8 (K-8), or
    Kindergarten through Grade 12. These mixed types of schools may be common in
    some settings and this feature may be added later.

    Args:
        school_size_distr_by_type (dict) : A dictionary of school size distributions binned by size groups or brackets for each school type.
        school_size_brackets (dict)      : A dictionary of school size brackets.
        uids_in_school (dict)            : A dictionary of students in school mapping ID to age.
        uids_in_school_by_age (dict)     : A dictionary of students in school mapping age to the list of IDs with that age.
        ages_in_school_count (dict)      : A dictionary mapping age to the number of students with that age.
        school_types_distr_by_age (dict) : A dictionary of the school type for each age.
        school_type_age_ranges (dict)    : A dictionary of the age range for each school type.

    Returns:
        Two lists of lists and third flat list, the first where each sublist is
        the ages of students in the same school, and the second is the same list
        but with the IDs of each student in place of their age. The third is a
        list of the school types for each school, where each school has a single
        string to represent it's school type.
    """

    student_age_lists = []
    student_uid_lists = []
    school_types = []

    sorted_size_brackets = sorted(school_size_brackets.keys())

    ages_in_school_distr = spb.norm_dic(ages_in_school_count)
    age_keys = list(ages_in_school_count.keys())

    while len(uids_in_school):

        new_student_ages = []
        new_student_uids = []

        aindex = age_keys[spsamp.fast_choice(ages_in_school_distr.values())]

        uid = uids_in_school_by_age[aindex][0]
        uids_in_school_by_age[aindex].remove(uid)
        uids_in_school.pop(uid, None)
        ages_in_school_count[aindex] -= 1
        ages_in_school_distr = spb.norm_dic(ages_in_school_count)

        new_student_ages.append(aindex)
        new_student_uids.append(uid)

        school_types_possible = sorted(school_types_distr_by_age[aindex].keys())
        prob = [school_types_distr_by_age[aindex][s] for s in school_types_possible]
        school_type = np.random.choice(school_types_possible, p=prob, size=1)[0]
        school_type_age_range = school_type_age_ranges[school_type]

        school_size_distr = school_size_distr_by_type[school_type]

        prob_by_sorted_size_brackets = [school_size_distr[b] for b in sorted_size_brackets]
        size_bracket = np.random.choice(sorted_size_brackets, p=prob_by_sorted_size_brackets)
        size = np.random.choice(school_size_brackets[size_bracket])
        size -= 1

        potential_student_ages = []
        for a in school_type_age_range:
            potential_student_ages.extend([a] * ages_in_school_count[a])

        if size >= len(potential_student_ages):
            size = len(potential_student_ages)
            school_age_count = {a: ages_in_school_count[a] for a in school_type_age_range}
            other_schools = [ns for ns in range(len(student_uid_lists)) if school_types[ns] == school_type]
            log.debug(f"other schools to merge with {other_schools} {school_type} {size} {school_size_brackets[0][0]}")

            # school is too small, try to merge it without another school of the same type
            if (size < school_size_brackets[0][0]) & (len(other_schools) > 0):
                log.debug(f'School size ({size + 1}) smaller than minimum school size {school_size_brackets[0][0]}. Will try now to merge with another school of the same type already made.')

                # another random school of the same type
                rns = other_schools[spsamp.fast_choice(np.ones(len(other_schools)))]

                for n, a in enumerate(school_type_age_range):
                    count = len(uids_in_school_by_age[a])
                    school_uids_in_age = uids_in_school_by_age[a]
                    new_student_ages.extend([a for i in range(count)])
                    new_student_uids.extend(school_uids_in_age)
                    ages_in_school_count[a] -= count

                # add to a previously generated school, add their ages and their uids, school type was already determined
                student_age_lists[rns].extend(new_student_ages)
                student_uid_lists[rns].extend(new_student_uids)

            else:
                log.debug(f'School size ({size + 1}) smaller than minimum school size {school_size_brackets[0][0]} but there are no other schools of the same type to merge with, so creating this one with however many students are available.')
                for n, a in enumerate(school_type_age_range):
                    count = len(uids_in_school_by_age[a])
                    school_uids_in_age = uids_in_school_by_age[a]
                    new_student_ages.extend([a for i in range(count)])
                    new_student_uids.extend(school_uids_in_age)
                    ages_in_school_count[a] -= count

                # add new school to lists although smaller than expected from school size distribution data
                student_age_lists.append(new_student_ages)
                student_uid_lists.append(new_student_uids)
                school_types.append(school_type)

        else:
            chosen = np.random.choice(potential_student_ages, size=size, replace=False)
            school_age_count = Counter(chosen)

            for n, a in enumerate(school_type_age_range):
                count = school_age_count[a]
                school_uids_in_age = uids_in_school_by_age[a][:count]
                uids_in_school_by_age[a] = uids_in_school_by_age[a][count:]
                new_student_ages += [a for i in range(count)]
                new_student_uids += school_uids_in_age
                ages_in_school_count[a] -= count

            # have created a new school and now adding the school with students to the lists for each data type (age, uid, and school type)
            student_age_lists.append(new_student_ages)
            student_uid_lists.append(new_student_uids)
            school_types.append(school_type)

        # having placed the students in the appropriate school, either a new one or an old one when sizes are too small, remove these students from those available to place in future schools
        for uid in new_student_uids:
            uids_in_school.pop(uid, None)
        ages_in_school_distr = spb.norm_dic(ages_in_school_count)

    return student_age_lists, student_uid_lists, school_types


# adding edges to the popdict, either from an edgelist or groups (groups are better when you have fully connected graphs - no need to enumerate for n*(n-1)/2 edges!)
def add_contacts_from_edgelist(popdict, edgelist, setting):
    """
    Add contacts to popdict from edges in an edgelist. Note that this simply
    adds to the contacts already in the layer and does not overwrite the
    contacts.

    Args:
        popdict (dict)  : dict of people
        edgelist (list) : list of edges
        setting (str)   : social setting layer

    Returns:
        Updated popdict.

    """
    for e in edgelist:
        i, j = e

        popdict[i]['contacts'][setting].add(j)
        popdict[j]['contacts'][setting].add(i)

    return popdict


def add_contacts_from_group(popdict, group, setting):
    """
    Add contacts to popdict from fully connected group. Note that this simply
    adds to the contacts already in the layer and does not overwrite the
    contacts.

    Args:
        popdict (dict) : dict of people
        group (list)   : list of people in group
        setting (str)  : social setting layer

    Returns:
        Updated popdict.

    """
    for i in group:
        popdict[i]['contacts'][setting] = popdict[i]['contacts'][setting].union(group)
        popdict[i]['contacts'][setting].remove(i)

    return popdict


def generate_random_contacts_for_additional_school_members(school_uids, additional_school_member_uids, average_additional_school_members_degree=20):
    """
    Generate random contacts for additional school members. This might be people
    like non teaching staff such as principals, administrative staff, cleaning
    staff, or school nurses.

    Args:
        school_uids (list)                               : list of uids of individuals already in the school
        additional_school_member_uids (list)             : list of uids of the additional school member who do not have contacts yet or for whom more contacts are needed
        average_additional_school_members_degree (float) : average degree for the additional school members

    Returns:
        List of edges for the additional school members in school.

    """
    edges = []
    all_school_uids = school_uids.copy() + additional_school_member_uids.copy()
    for uid in additional_school_member_uids:
        k = np.random.poisson(average_additional_school_members_degree)
        possible_neighbors = all_school_uids.copy()
        possible_neighbors.remove(uid)
        new_neighbours = np.random.choice(possible_neighbors, k)
        for j in new_neighbours:
            e = (uid, j)
            edges.append(e)
    return edges


def generate_random_classes_by_grade_in_school(student_uids, student_ages, age_by_uid, grade_age_mapping, age_grade_mapping, average_class_size=20, inter_grade_mixing=0.1):
    """
    Generate edges for contacts mostly within the same age/grade. Edges are
    randomly distributed so that clustering is roughly average_class_size/size
    of the grade. Inter grade mixing is done by rewiring edges, specifically
    swapping endpoints of pairs of randomly sampled edges.

    Args:
        student_uids (list)        : list of uids of students in the school
        student_ages (list)        : list of the ages of the students in the school
        age_by_uid (dict)          : dict mapping uid to age
        grade_age_mapping (dict)   : dict mapping grade to an age
        age_grade_mapping (dict)   : dict mapping age to a grade
        average_class_size (float) : average class size
        inter_grade_mixing (float) : percent of edges that rewired to create edges across grades in schools when school_mixing_type is 'age_clustered'

    Returns:
        List of edges between students in school.

    """
    # what are the ages in the school
    age_counter = Counter(student_ages)
    age_keys = sorted(age_counter.keys())
    age_keys_indices = {a: i for i, a in enumerate(age_keys)}

    # create a dictionary with the list of uids for each age/grade
    uids_in_school_by_age = {}
    for a in age_keys:
        uids_in_school_by_age[a] = []

    for uid in student_uids:
        a = age_by_uid[uid]
        uids_in_school_by_age[a].append(uid)

    age_groups_smaller_than_degree = False
    for a in uids_in_school_by_age:
        if average_class_size > len(uids_in_school_by_age[a]):
            age_groups_smaller_than_degree = True

    # create a graph of contacts in the school
    G = nx.Graph()

    for a in uids_in_school_by_age:

        # for Erdos Renyi graph of N nodes and average degree k, p is essentially the density of all possible edges --> p = # edges / # all possible edges. With average degree k, # of edges is roughly N * k / 2 and # of all possible edges is N * (N-1) / 2, which leads us to k = (N - 1) * p or, in Stirling's Approx. k = N * p, that is p = k / N
        Ga = spcnx.random_graph_model(uids_in_school_by_age[a], average_class_size)
        for e in Ga.edges():
            i, j = e

            # add each edge to the overall school graph
            G.add_edge(uids_in_school_by_age[a][i], uids_in_school_by_age[a][j])

    # make sure all students are in the graph by adding those without an edge yet
    missing_uids = set(student_uids) - set(G.nodes())
    G.add_nodes_from(missing_uids)

    # flag was turned on to indicate that the average degree is too low. How can we add more edges? do the following: create a second random graph across the entire school. Loop over everyone and grab edges as necessary. Loop again to remove edges if it's too many.
    if age_groups_smaller_than_degree:

        G = add_random_contacts_from_graph(G, average_class_size)

    # rewire some edges between people within the same grade/age to now being edges across grades/ages
    E = list(G.edges())
    np.random.shuffle(E)

    nE = int(len(E) / 2.)  # we'll loop over edges in pairs so only need to loop over half the length
    missed_rewiring = 0

    for n in range(nE):
        if np.random.binomial(1, p=inter_grade_mixing):

            i = 2 * n
            j = 2 * n + 1

            ei = E[i]
            ej = E[j]

            ei1, ei2 = ei
            ej1, ej2 = ej

            # try to switch from ei1-ei2, ej1-ej2 to ei1-ej2, ej1-ei2
            if ei1 != ej1 and ei2 != ej2 and ei1 != ej2 and ej1 != ei2:
                new_ei = (ei1, ej2)
                new_ej = (ei2, ej1)

            # instead try to switch from ei1-ei2, ej1-ej2 to ei1-ej1, ei2-ej2
            elif ei1 != ej2 and ei2 != ej1 and ei1 != ej1 and ej2 != ei2:
                new_ei = (ei1, ej1)
                new_ej = (ei2, ej2)

            else:
                missed_rewiring += 1
                continue

            G.remove_edges_from([ei, ej])
            G.add_edges_from([new_ei, new_ej])

    # calculate school age mixing and print some debugging statements
    if logging.getLevelName(log.level) == 'DEBUG': # pragma: no cover
        print(f"clustering within age/grade clustered school: {nx.transitivity(G)}")
        print(f"missed rewiring {missed_rewiring} edge pairs out of {nE} possible pairs.")
        ecount = np.zeros((len(age_keys), len(age_keys)))
        for e in G.edges():
            i, j = e

            age_i = age_by_uid[i]
            index_i = age_keys_indices[age_i]
            age_j = age_by_uid[j]
            index_j = age_keys_indices[age_j]

            ecount[index_i][index_j] += 1
            ecount[index_j][index_i] += 1

        print(f"within school age mixing matrix\n {ecount}")

    return list(G.edges())


def generate_clustered_classes_by_grade_in_school(student_uids, student_ages, age_by_uid, grade_age_mapping, age_grade_mapping, average_class_size=20, return_edges=False):
    """
    Generate edges for contacts mostly within the same age/grade. Edges are
    randomly distributed so that clustering is roughly average_class_size/size
    of the grade.

    The last classroom created may be much smaller than the average_class_size.

    Args:
        student_uids (list)        : list of uids of students in the school
        student_ages (list)        : list of the ages of the students in the school
        age_by_uid (dict)          : dict mapping uid to age
        grade_age_mapping (dict)   : dict mapping grade to an age
        age_grade_mapping (dict)   : dict mapping age to a grade
        average_class_size (float) : average class size
        return_edges (bool)        : If True, return edges, else return two groups of contacts - students and teachers for each class

    Returns:
        List of edges between students in school or groups of contacts.

    """
    # what are the ages in the school
    age_counter = Counter(student_ages)
    age_keys = sorted(age_counter.keys())
    age_keys_indices = {a: i for i, a in enumerate(age_keys)}

    # create a dictionary with the list of uids for each age/grade
    uids_in_school_by_age = {}
    for a in age_keys:
        uids_in_school_by_age[a] = []

    for uid in student_uids:
        a = age_by_uid[uid]
        uids_in_school_by_age[a].append(uid)

    G = nx.Graph()

    nodes_left = []
    groups = []

    for a in uids_in_school_by_age:
        nodes = sc.dcp(uids_in_school_by_age[a])
        np.random.shuffle(nodes)

        while len(nodes) > 0:
            cluster_size = np.random.poisson(average_class_size)

            if cluster_size > len(nodes):
                # gather the last group of nodes into a pool to choose from afterwards
                nodes_left += list(nodes)
                break

            group = nodes[:cluster_size]
            if cluster_size > 0:
                groups.append(group)
            nodes = nodes[cluster_size:]

    # shuffle the students left over to place into classrooms
    np.random.shuffle(nodes_left)

    while len(nodes_left) > 0:
        cluster_size = np.random.poisson(average_class_size)

        if cluster_size > len(nodes_left):
            cluster_size = len(nodes_left)
            break

        group = nodes_left[:cluster_size]
        if cluster_size > 0:
            groups.append(group)
        nodes_left = nodes_left[cluster_size:]

    # with some school sizes and parameter values you may not have made any classrooms yet
    if len(groups) == 0:
        groups.append(nodes_left[:cluster_size])
        nodes_left = nodes_left[cluster_size:]

    else:
        for i in nodes_left:
            ng = spsamp.fast_choice(np.ones(len(groups)))  # choose one of the other classes to add to
            groups[ng].append(i)

    if return_edges: # pragma: no cover
        for ng in range(len(groups)):
            group = groups[ng]
            Gn = nx.complete_graph(len(group))
            for e in Gn.edges():
                i, j = e
                node_i = group[i]
                node_j = group[j]
                G.add_edge(node_i, node_j)

    if logging.getLevelName(log.level) == 'DEBUG': # pragma: no cover

        if return_edges:
            ecount = np.zeros((len(age_keys), len(age_keys)))
            for e in G.edges():
                i, j = e

                age_i = age_by_uid[i]
                index_i = age_keys_indices[age_i]
                age_j = age_by_uid[j]
                index_j = age_keys_indices[age_j]

                ecount[index_i][index_j] += 1
                ecount[index_j][index_i] += 1

            print(f"within school age mixing matrix\n{ecount}")

    if return_edges:
        return list(G.edges())

    else:
        # if returning groups, much easier to add to population dictionaries and assign teachers to a single class
        return groups


def generate_edges_between_teachers(teacher_uids, average_teacher_teacher_degree):
    """
    Generate edges between teachers.

    Args:
        teachers (list)                      : a list of teachers
        average_teacher_teacher_degree (int) : average number of contacts with other teachers

    Return:
        List of edges between teachers.

    """
    edges = []
    if average_teacher_teacher_degree > len(teacher_uids):
        eiter = combinations(teacher_uids, 2)
        edges = [e for e in eiter]

    else:
        G = spcnx.random_graph_model(teacher_uids, average_teacher_teacher_degree)
        for e in G.edges():
            i, j = e
            teacher_i = teacher_uids[i]
            teacher_j = teacher_uids[j]
            e = (teacher_i, teacher_j)
            edges.append(e)

    return edges


def generate_edges_for_teachers_in_random_classes(student_uids, student_ages, teacher_uids, age_by_uid, average_student_teacher_ratio=20, average_teacher_teacher_degree=4):
    """
    Generate edges for teachers, including to both students and other teachers
    at the same school. Well mixed contacts within the same age/grade, some
    cross grade mixing. Teachers are clustered by grade mostly.

    Args:
        student_uids (list)                    : list of uids of students in the school
        student_ages (list)                    : list of the ages of the students in the school
        teacher_uids (list)                    : list of teachers in the school
        age_by_uid (dict)                      : dict mapping uid to age
        grade_age_mapping (dict)               : dict mapping grade to an age
        age_grade_mapping (dict)               : dict mapping age to a grade
        average_student_teacher_ratio (float)  : average number of students per teacher
        average_teacher_teacher_degree (float) : average number of contacts with other teachers

    Return:
        List of edges connected to teachers.

    """
    age_keys = list(set(student_ages))

    # create a dictionary with the list of uids for each age/grade
    uids_in_school_by_age = {}
    for a in age_keys:
        uids_in_school_by_age[a] = []

    for uid in student_uids:
        a = age_by_uid[uid]
        uids_in_school_by_age[a].append(uid)

    edges = []

    teachers_assigned = []
    available_teachers = sc.dcp(teacher_uids)
    for a in uids_in_school_by_age:

        n_teachers_needed = int(np.round(len(uids_in_school_by_age[a]) / average_student_teacher_ratio, 1))
        n_teachers_needed = max(1, n_teachers_needed)  # at least one teacher

        if n_teachers_needed > len(available_teachers) + len(teachers_assigned):
            n_teachers_needed = len(available_teachers) + len(teachers_assigned)
            selected_teachers = available_teachers + teachers_assigned

        elif n_teachers_needed > len(available_teachers):
            selected_teachers = available_teachers
            n_teachers_needed = n_teachers_needed - len(available_teachers)
            selected_teachers += list(np.random.choice(teachers_assigned, replace=False, size=n_teachers_needed))

        else:
            selected_teachers = np.random.choice(available_teachers, replace=False, size=n_teachers_needed)
            for t in selected_teachers:
                available_teachers.remove(t)
                teachers_assigned.append(t)

        # only adds one teacher per student
        for student in uids_in_school_by_age[a]:
            teacher = np.random.choice(selected_teachers)
            e = (student, teacher)
            edges.append(e)

    # some teachers left so add them as contacts to other students
    for teacher in available_teachers:

        n_students = max(1, np.random.poisson(average_student_teacher_ratio))

        if n_students > len(student_uids):
            n_students = len(student_uids)

        selected_students = np.random.choice(student_uids, replace=False, size=n_students)

        for student in selected_students:
            e = (student, teacher)
            edges.append(e)

        teachers_assigned.append(teacher)

    available_teachers = []

    teacher_teacher_edges = generate_edges_between_teachers(teachers_assigned, average_teacher_teacher_degree)
    edges += teacher_teacher_edges

    G = nx.Graph()
    G.add_edges_from(edges)

    for s in student_uids:
        log.debug(f"student {s}, age: {age_by_uid[s]}, has {G.degree(s)} contacts with teachers")
    for t in teachers_assigned:
        log.debug(f"teacher {t}, age: {age_by_uid[t]}, has {G.degree(t)} contacts with students")

    # not returning student-student contacts
    return edges


def generate_edges_for_teachers_in_clustered_classes(groups, teacher_uids, average_teacher_teacher_degree=4, return_edges=False):
    """
    Generate edges for teachers, including to both students and other teachers
    at the same school. Students and teachers are clustered into disjoint
    classes.

    Args:
        groups (list)                          : list of lists of students, clustered into groups mostly by grade
        teacher_uids (list)                    : list of teachers in the school
        average_teacher_teacher_degree (float) : average number of contacts with other teachers
        return_edges (bool)                    : If True, return edges, else return two groups of contacts - students and teachers for each class

    Return:
        List of edges connected to teachers.

    """
    edges = []
    teacher_groups = []
    np.random.shuffle(groups)  # shuffle the clustered groups of students / classes so that the classes aren't ordered from youngest to oldest

    available_teachers = sc.dcp(teacher_uids)

    # have exactly as many teachers as needed
    if len(groups) == len(available_teachers):
        for ng, t in enumerate(available_teachers):
            teacher_groups.append([t])
        available_teachers = []

    # you don't have enough teachers to cover the classes so break the extra groups up
    elif len(groups) > len(available_teachers):
        n_groups_to_break = len(groups) - len(available_teachers)

        # grab the last cluster and split it up and spread the students to the other groups
        for ngb in range(n_groups_to_break):
            group_to_break = groups[-1]

            for student in group_to_break:
                ng = np.random.randint(len(groups) - 1)  # find another class to join
                groups[ng].append(student)
            groups = groups[:-1]

        for ng, t in enumerate(available_teachers):
            teacher_groups.append([t])
        available_teachers = []

    elif len(groups) < len(available_teachers):
        for ng, group in enumerate(groups):

            # class size already determines that each class gets at least one teacher and make that a list - maybe we can add other teachers some other way
            teacher_groups.append([available_teachers[ng]])
        available_teachers = available_teachers[len(groups):]

        # spread extra teachers among the classes
        for t in available_teachers:
            ng = np.random.randint(len(groups))
            teacher_groups[ng].append(t)
        available_teachers = []

    # create edges between students and teachers
    for ng, group in enumerate(groups):
        for student in group:
            for teacher in teacher_groups[ng]:
                e = (student, teacher)
                edges.append(e)

    if return_edges:
        teacher_teacher_edges = []
        for ng, teacher_group in enumerate(teacher_groups):
            teacher_teacher_edges += generate_edges_between_teachers(teacher_group, average_teacher_teacher_degree)
        edges += teacher_teacher_edges
        # not returning student-student contacts
        return edges
    else:
        return groups, teacher_groups


def generate_random_contacts_across_school(all_school_uids, average_class_size):
    """
    Generate edges for contacts in a school where everyone mixes randomly.
    Assuming class and thus class size determines effective contacts.

    Args:
        all_school_uids (list)   : list of uids of individuals in the school
        average_class_size (int) : average class size or number of contacts in school

    Returns:
        List of edges between individuals in school.
    """
    edges = []
    G = spcnx.random_graph_model(all_school_uids, average_class_size)  # undirected graph
    for u, uid in enumerate(all_school_uids):
        es = [(uid, all_school_uids[v]) for v in G.neighbors(u)]
        edges.extend(es)

    return edges


def add_school_edges(popdict, student_uids, student_ages, teacher_uids, non_teaching_staff_uids, age_by_uid, grade_age_mapping, age_grade_mapping, average_class_size=20, inter_grade_mixing=0.1, average_student_teacher_ratio=20, average_teacher_teacher_degree=3, average_additional_staff_degree=20, school_mixing_type='random'):
    """
    Generate edges for teachers, including to both students and other teachers
    at the same school. When school_mixing_type is 'age_clustered' then
    inter_grade_mixing will rewire a fraction of the edges between students in
    the same age or grade to be edges with any other student in the school. When
    school_mixing_type is 'random' or 'age_and_class_clustered',
    inter_grade_mixing has no effect.

    Args:
        popdict (dict)                          : dictionary of people
        student_uids (list)                     : list of uids of students in the school
        student_ages (list)                     : list of the ages of the students in the school
        teacher_uids (list)                     : list of teachers in the school
        non_teaching_staff_uids (list)          : list of non teaching staff in the school
        age_by_uid (dict)                       : dict mapping uid to age
        grade_age_mapping (dict)                : dict mapping grade to an age
        age_grade_mapping (dict)                : dict mapping age to a grade
        average_class_size (float)              : average class size
        inter_grade_mixing (float)              : percent of edges that rewired to create edges across grades in schools when school_mixing_type is 'age_clustered'
        average_student_teacher_ratio (float)   : average number of students per teacher
        average_teacher_teacher_degree (float)  : average number of contacts with other teachers
        average_additional_staff_degree (float) : The average number of contacts per additional non teaching staff in schools.
        school_mixing_type(str)                 : 'random' for well mixed schools, 'age_clustered' for well mixed within the same grade and some intermixing with other grades, 'age_and_class_clustered' for disjoint classes in a school by age or grade

    Return:
        Updated popdict with edges generated in schools.

    Notes:
        average_teacher_teacher_degree will not be used in school_mixing_type == 'random' scenario.
    """
    # completely random contacts across the school, no guarantee of contact with a teacher, much like universities
    available_school_mixing_types = ['random', 'age_clustered', 'age_and_class_clustered']

    if school_mixing_type not in available_school_mixing_types:
        print(f"school_mixing_type: {school_mixing_type} 'does not exist. Please change this to one of: {available_school_mixing_types}")

    if school_mixing_type == 'random':
        school_uids = []
        school_uids.extend(student_uids)
        school_uids.extend(teacher_uids)
        edges = generate_random_contacts_across_school(school_uids, average_class_size)
        add_contacts_from_edgelist(popdict, edges, 'S')
        student_groups = [student_uids]
        teacher_groups = [teacher_uids]

    # random contacts across a grade in the school, most edges will across the same age group, much like middle schools or high schools, the inter_grade_mixing parameter is a tuning parameter, students get at least one teacher as a contact
    elif school_mixing_type == 'age_clustered':
        edges = generate_random_classes_by_grade_in_school(student_uids, student_ages, age_by_uid, grade_age_mapping, age_grade_mapping, average_class_size, inter_grade_mixing)

        teacher_edges = generate_edges_for_teachers_in_random_classes(student_uids, student_ages, teacher_uids, age_by_uid, average_student_teacher_ratio, average_teacher_teacher_degree)
        edges += teacher_edges

        add_contacts_from_edgelist(popdict, edges, 'S')
        student_groups = [student_uids]
        teacher_groups = [teacher_uids]

    # completely clustered into classes by age, one teacher per class at least
    elif school_mixing_type == 'age_and_class_clustered':

        student_groups = generate_clustered_classes_by_grade_in_school(student_uids, student_ages, age_by_uid, grade_age_mapping, age_grade_mapping, average_class_size=average_class_size, return_edges=False)
        student_groups_2 = sc.dcp(student_groups)
        student_groups, teacher_groups = generate_edges_for_teachers_in_clustered_classes(student_groups, teacher_uids, average_teacher_teacher_degree=average_teacher_teacher_degree)

        sum_diff = sum([len(group) for group in student_groups]) - sum([len(group) for group in student_groups_2])
        assert sum_diff == 0, f'Check failed. sum of the differences between student groups is not zero. Total school enrollment changed between the step of creating student groups and assigning teachers to each group. sum is {sum_diff}'

        for ng in range(len(student_groups)):
            student_group = student_groups[ng]
            teacher_group = teacher_groups[ng]
            group = student_group
            group += teacher_group

            add_contacts_from_group(popdict, group, 'S')

        log.debug(f"average_class_size, {average_class_size}, 'class_group sizes', {[len(group) for group in student_groups]}")

        # additional edges between teachers in different classes - makes distinct clusters connected - this may add edges again between teachers in the same class
        teacher_edges = generate_edges_between_teachers(teacher_uids, average_teacher_teacher_degree)
        add_contacts_from_edgelist(popdict, teacher_edges, 'S')

    all_school_uids = []
    all_school_uids.extend(student_uids)
    all_school_uids.extend(teacher_uids)
    additional_staff_edges = generate_random_contacts_for_additional_school_members(all_school_uids, non_teaching_staff_uids, average_additional_staff_degree)
    add_contacts_from_edgelist(popdict, additional_staff_edges, 'S')

    return popdict, student_groups, teacher_groups


def get_school_types_distr_by_age(school_type_age_ranges):
    """
    Return probabilities of school type for each age. For now assuming no
    overlapping of grades between school types.

    Return:
        A dictionary of default probabilities for the school type likely for
        each age.
    """
    school_types_distr_by_age = {}
    for a in range(101):
        school_types_distr_by_age[a] = dict.fromkeys(list(school_type_age_ranges.keys()), 0.)

    for k in school_type_age_ranges.keys():
        for a in school_type_age_ranges[k]:
            school_types_distr_by_age[a][k] = 1.

    return school_types_distr_by_age


def get_school_types_by_age_single(school_types_distr_by_age):
    """
    Return school type by age by assigning the school type with the highest
    probability.

    Return:
        A dictionary of default school type by age.

    """
    school_types_by_age_single = {}
    for a in range(101):
        values_to_keys = {school_types_distr_by_age[a][k]: k for k in school_types_distr_by_age[a]}
        max_v = max(values_to_keys.keys())
        max_k = values_to_keys[max_v]
        if max_v != 0:
            school_types_by_age_single[a] = max_k

    return school_types_by_age_single


def get_school_type_data(datadir, location, state_location, country_location, use_default=False):
    """
    Get location specific distributions on school type data if it's available for all the distributions of interest, otherwise return default data if use_default.

    Args:
        datadir (string)          : file path to the data directory
        location (string)         : name of the location
        state_location (string)   : name of the state the location is in
        country_location (string) : name of the country the location is in
        use_default (bool)        : if True, try to first use the other parameters to find data specific to the location under study, otherwise returns default data drawing from Seattle, Washington.

    Returns:
        3 dictionaries necessary to generate schools by the type of school (i.e. elementary, middle, high school, etc.).
    """
    school_size_distr_by_type = spdata.get_school_size_distr_by_type(datadir, location=location, state_location=state_location, country_location=country_location, use_default=use_default)
    school_size_brackets = spdata.get_school_size_brackets(datadir, location=location, state_location=state_location, country_location=country_location, use_default=use_default)  # for right now the size distribution for all school types will use the same brackets or bins
    school_type_age_ranges = spdata.get_school_type_age_ranges(datadir, location=location, state_location=state_location, country_location=country_location, use_default=use_default)

    #     if use_default:
    #         school_size_distr_by_type = spdata.get_default_school_size_distr_by_type()
    #         school_size_brackets = spdata.get_default_school_size_distr_brackets()
    #         school_type_age_ranges = spdata.get_default_school_type_age_ranges()
    #     else:
    #         raise ValueError(f"Data unavailable for the location specified. Please check input strings or set use_default to True to use default values.")

    return school_size_distr_by_type, school_size_brackets, school_type_age_ranges


def assign_teachers_to_schools(student_age_lists, student_uid_lists, employment_rates, workers_by_age_to_assign_count, potential_worker_uids, potential_worker_uids_by_age, potential_worker_ages_left_count, average_student_teacher_ratio=20, teacher_age_min=25, teacher_age_max=75):
    """
    Assign teachers to each school according to the average student-teacher
    ratio.

    Args:
        student_age_lists (list)                : list of lists where each sublist is a school with the ages of the students within
        student_uid_lists (list)                : list of lists where each sublist is a school with the ids of the students within
        employment_rates (dict)                 : employment rates by age
        workers_by_age_to_assign_count (dict)   : dictionary of the count of workers left to assign by age
        potential_worker_uids (dict)            : dictionary of potential workers mapping their id to their age
        potential_worker_uids_by_age (dict)     : dictionary mapping age to the list of worker ids with that age
        potential_worker_ages_left_count (dict) : dictionary of the count of potential workers left that can be assigned by age
        average_student_teacher_ratio (float)   : The average number of students per teacher
        teacher_age_min (int)                   : The minimum age for teachers
        teacher_age_max (int)                   : The maximum age for teachers

    Returns:
        List of lists of schools with the ages of individuals in each, lists of
        lists of schools with the ids of individuals in each, dictionary of
        potential workers mapping id to their age, dictionary mapping age to the
        list of potential workers of that age, dictionary with the count of
        workers left to assign for each age after teachers have been assigned.
    """

    log.debug('assign_teachers_to_schools()')
    # matrix method will already get some teachers into schools so student_teacher_ratio should be higher

    all_teachers = dict.fromkeys(np.arange(101), 0)

    teacher_age_lists = []
    teacher_uid_lists = []

    for n in range(len(student_age_lists)):
        student_ages = student_age_lists[n]
        student_uids = student_uid_lists[n]

        # size = len(school_uids)
        size = len(student_ages)
        nteachers = int(size / float(average_student_teacher_ratio))
        nteachers = max(1, nteachers)

        # log.debug(f"nteachers {nteachers}, student-teacher ratio, {(size / nteachers):.4f}")

        teacher_ages = []
        teacher_uids = []

        for nt in range(nteachers):

            a = spsamp.sample_from_range(workers_by_age_to_assign_count, teacher_age_min, teacher_age_max)
            uid = potential_worker_uids_by_age[a][0]
            teacher_ages.append(a)
            all_teachers[a] += 1

            potential_worker_uids_by_age[a].remove(uid)
            workers_by_age_to_assign_count[a] -= 1
            potential_worker_ages_left_count[a] -= 1
            potential_worker_uids.pop(uid, None)

            teacher_ages.append(a)
            teacher_uids.append(uid)

        teacher_age_lists.append(teacher_ages)
        teacher_uid_lists.append(teacher_uids)

        if logging.getLevelName(log.level) == 'DEBUG':
            print(f"nteachers {nteachers}, student-teacher ratio, {(size / nteachers):.4f}")
            print(f"school with teachers {sorted(student_uids)}")
            print(f"nkids: {(np.array(student_ages) <= 19).sum()}, n20=>: {(np.array(student_ages) > 19).sum()}")
            print(f"kid-adult ratio: {np.divide((np.array(student_ages) <= 19).sum() , (np.array(student_ages) > 19).sum())}")

    return teacher_age_lists, teacher_uid_lists, potential_worker_uids, potential_worker_uids_by_age, workers_by_age_to_assign_count


def assign_additional_staff_to_schools(student_uid_lists, teacher_uid_lists, workers_by_age_to_assign_count, potential_worker_uids, potential_worker_uids_by_age, potential_worker_ages_left_count, average_student_teacher_ratio=20, average_student_all_staff_ratio=15, staff_age_min=20, staff_age_max=75, with_non_teaching_staff=False):
    """
    Assign additional staff to each school according to the average student to
    all staff ratio.

    Args:
        student_uid_lists (list)                : list of lists where each sublist is a school with the ids of the students within
        teacher_uid_lists (list)                : list of lists where each sublist is a school with the ids of the teachers within
        workers_by_age_to_assign_count (dict)   : dictionary of the count of workers left to assign by age
        potential_worker_uids (dict)            : dictionary of potential workers mapping their id to their age
        potential_worker_uids_by_age (dict)     : dictionary mapping age to the list of worker ids with that age
        potential_worker_ages_left_count (dict) : dictionary of the count of potential workers left that can be assigned by age
        average_student_teacher_ratio (float)   : The average number of students per teacher.
        average_student_all_staff_ratio (float) : The average number of students per staff members at school (including both teachers and non teachers).
        staff_age_min (int)                     : The minimum age for non teaching staff.
        staff_age_max (int)                     : The maximum age for non teaching staff.
        with_non_teaching_staff (bool)          : If True, includes non teaching staff.

    Returns:
        List of lists of schools with the ids of non teaching staff for each
        school, dictionary of potential workers mapping id to their age,
        dictionary mapping age to the list of potential workers of that age,
        dictionary with the count of workers left to assign for each age after
        teachers have been assigned.
    """
    log.debug('assign_additional_staff_to_schools()')

    # with_non_teaching_staff is False so this method will not select anyone to be a non teaching staff member at schools - thus return empty lists for non_teaching_staff_uids
    if not with_non_teaching_staff:
        log.debug(f"with_non_teaching_staff: {with_non_teaching_staff}, so this method does not produce additional staff")

        non_teaching_staff_uid_lists = [[] for student_list in student_uid_lists]
        return non_teaching_staff_uid_lists, potential_worker_uids, potential_worker_uids_by_age, workers_by_age_to_assign_count

    if average_student_teacher_ratio < average_student_all_staff_ratio:
        errormsg = f"The ratio of students to all staff at school ({average_student_all_staff_ratio}) must be lower than or equal to the ratio students to teachers at school ({average_student_teacher_ratio}). All staff includes both teaching and non teaching staff, so if the student to all staff ratio is greater than the student to teacher ratio then this would expect there to be more teachers than all possible staff in a school."
        raise ValueError(errormsg)

    n_students_list = [len(student_list) for student_list in student_uid_lists]  # what is the number of students in each school
    n_teachers_list = [len(teacher_list) for teacher_list in teacher_uid_lists]  # what is the number of teachers in each school

    if average_student_all_staff_ratio == 0:
        raise ValueError(f"The ratio of students to all staff at school is {average_student_all_staff_ratio}. This would mean no students at the school. Try another value greater than 0 and less than the average_student_teacher_ratio: {average_student_teacher_ratio}.")

    else:
        n_all_staff_list = [max(1, int(i/average_student_all_staff_ratio)) for i in n_students_list]  # need at least one staff member
    n_non_teaching_staff_list = [n_all_staff_list[i] - n_teachers_list[i] for i in range(len(n_students_list))]

    min_n_non_teaching_staff = min(n_non_teaching_staff_list)

    # log.debug(f"list of number of students per school: {n_students_list}")
    # log.debug(f"list of number of teachers per school: {n_teachers_list}")
    # log.debug(f"list of number of all staff expected per school: {n_all_staff_list}")
    # log.debug(f"list of number of non teaching staff expected per school: {n_non_teaching_staff_list}")
    if min_n_non_teaching_staff <= 0:
        errormsg = f"At least one school expects only 1 non teaching staff member. Either check the average_student_teacher_ratio ({average_student_teacher_ratio}) and the average_student_all_staff_ratio ({average_student_all_staff_ratio}) if you do not expect this to be the case, or some of the generated schools may have too few staff members."
        log.debug(errormsg)

    n_non_teaching_staff_list = [i if i > 0 else 1 for i in n_non_teaching_staff_list]  # force one extra staff member beyond teachers

    non_teaching_staff_uid_lists = []

    for i in range(len(n_non_teaching_staff_list)):
        n_non_teaching_staff = n_non_teaching_staff_list[i]  # how many non teaching staff for the school
        non_teaching_staff_uids_in_this_school = []

        for j in range(n_non_teaching_staff):
            a = spsamp.sample_from_range(workers_by_age_to_assign_count, staff_age_min, staff_age_max)
            uid = potential_worker_uids_by_age[a][0]
            workers_by_age_to_assign_count[a] -= 1
            potential_worker_ages_left_count[a] -= 1
            potential_worker_uids.pop(uid, None)
            potential_worker_uids_by_age[a].remove(uid)

            non_teaching_staff_uids_in_this_school.append(uid)

        non_teaching_staff_uid_lists.append(non_teaching_staff_uids_in_this_school)

    return non_teaching_staff_uid_lists, potential_worker_uids, potential_worker_uids_by_age, workers_by_age_to_assign_count


def add_random_contacts_from_graph(G, average_degree):
    """
    Add additional edges at random to achieve the expected or desired average
    degree.

    Args:
        G (networkx Graph)   : networkx Graph object
        average_degree (int) : expected or desired average degree

    Returns:
        Updated networkx Graph object with additional edges added at random.

    """
    nodes = G.nodes()

    ordered_node_ids = {node: node_id for node_id, node in enumerate(nodes)}
    ids_to_ordered_nodes = {node_id: node for node_id, node in enumerate(nodes)}

    if len(nodes) == 0:
        return G

    p = average_degree / len(nodes)

    G2 = spcnx.random_graph_model(nodes, average_degree)

    for node in nodes:
        ordered_node_id = ordered_node_ids[node]

        extra_neighbors = list(G2.neighbors(ordered_node_id))
        extra_edges_needed = len(extra_neighbors) - G.degree(node)

        if extra_edges_needed > 0:
            extra_neighbors_to_add = np.random.choice(extra_neighbors, extra_edges_needed)
            for j in extra_neighbors_to_add:
                neighbor = ids_to_ordered_nodes[j]
                G.add_edge(node, neighbor)

    # in case you've added too many edges, let's remove a few - likely to not be hit
    for node in nodes:
        ordered_node_id = ordered_node_ids[node]
        extra_edges_to_remove = G.degree(node) - G2.degree(ordered_node_id)
        extra_edges_to_remove = int(extra_edges_to_remove / 2.)

        if extra_edges_to_remove > 0:
            extra_neighbors_to_remove = np.random.choice(extra_neighbors, extra_edges_to_remove)
            for j in extra_neighbors_to_remove:
                neighbor = ids_to_ordered_nodes[j]
                if G.has_edge(node, neighbor):
                    G.remove_edge(node, neighbor)

    return G


# %% Things added to enable not-by-type and random

def generate_school_sizes(school_size_distr_by_bracket, school_size_brackets, uids_in_school):
    """
    Given a number of students in school, generate a list of school sizes to
    place everyone in a school.

    Args:
        school_size_distr_by_bracket (dict) : The distribution of binned school sizes.
        school_size_brackets (dict)         : A dictionary of school size brackets.
        uids_in_school (dict)               : A dictionary of students in school mapping ID to age.

    Returns:
        A list of school sizes whose sum is the length of ``uids_in_school``.
    """
    ns = len(uids_in_school)
    sorted_brackets = sorted(school_size_brackets.keys())
    prob_by_sorted_brackets = [school_size_distr_by_bracket[b] for b in sorted_brackets]

    school_sizes = []

    while ns > 0:
        size_bracket = np.random.choice(sorted_brackets, p=prob_by_sorted_brackets)
        # size = np.random.choice(school_size_brackets[size_bracket])  # creates some schools that are much smaller than expected so use average instead
        size = int(np.mean(school_size_brackets[size_bracket]))  # use average school size to avoid schools with very small sizes
        ns -= size
        school_sizes.append(size)
    if ns < 0:
        school_sizes[-1] = school_sizes[-1] + ns
    np.random.shuffle(school_sizes)
    return school_sizes


def send_students_to_school(school_sizes, uids_in_school, uids_in_school_by_age, ages_in_school_count, age_brackets, age_by_brackets, contact_matrices): 
    """
    A method to send students to school together. Using the matrices to
    construct schools is not a perfect method so some things are more forced
    than the matrix method alone would create. This method models schools using
    matrices and so it does not create explicit school types.

    Args:
        school_sizes (list)          : A list of school sizes.
        uids_in_school (dict)        : A dictionary of students in school mapping ID to age.
        uids_in_school_by_age (dict) : A dictionary of students in school mapping age to the list of IDs with that age.
        ages_in_school_count (dict)  : A dictionary mapping age to the number of students with that age.
        age_brackets (dict)          : A dictionary mapping age bracket keys to age bracket range.
        age_by_brackets(dict)        : A dictionary mapping age to the age bracket range it falls within.
        contact_matrices (dict)      : A dictionary of age specific contact matrix for different physical contact settings.

    Returns:
        Two lists of lists and third flat list, the first where each sublist is
        the ages of students in the same school, and the second is the same list
        but with the IDs of each student in place of their age. The third is a
        list of the school types for each school, where each school has a single
        string to represent it's school type.
    """
    log.debug('send_students_to_school()')
    school_age_lists = []
    school_uid_lists = []
    school_types = []

    ages_in_school_distr = spb.norm_dic(ages_in_school_count)
    left_in_bracket = spb.get_aggregate_ages(ages_in_school_count, age_by_brackets)

    for n, size in enumerate(school_sizes):

        if len(uids_in_school) == 0:  # no more students left to send to school!
            break

        ages_in_school_distr = spb.norm_dic(ages_in_school_count)

        new_school = []
        new_school_uids = []

        aindex = spsamp.fast_choice(ages_in_school_distr.values())
        bindex = age_by_brackets[aindex]

        # reference students under 20 to prevent older adults from being reference students (otherwise we end up with schools with too many adults and kids mixing because the matrices represent the average of the patterns and not the bimodal mixing of adult students together at school and a small number of teachers at school with their students)
        if bindex >= 4:
            if np.random.binomial(1, p=0.7):

                aindex = spsamp.fast_choice(ages_in_school_distr.values())

        uid = uids_in_school_by_age[aindex][0]
        uids_in_school_by_age[aindex].remove(uid)
        uids_in_school.pop(uid, None)
        ages_in_school_count[aindex] -= 1
        ages_in_school_distr = spb.norm_dic(ages_in_school_count)

        new_school.append(aindex)
        new_school_uids.append(uid)

        log.debug(f"reference school age {aindex}, school size {size}, students left {len(uids_in_school)}, {left_in_bracket}")

        bindex = age_by_brackets[aindex]
        b_prob = contact_matrices['S'][bindex, :]

        left_in_bracket[bindex] -= 1

        # fewer students than school size so everyone else is in one school
        if len(uids_in_school) < size:
            for uid in uids_in_school:
                ai = uids_in_school[uid]
                new_school.append(int(ai))
                new_school_uids.append(uid)
                uids_in_school_by_age[ai].remove(uid)
                ages_in_school_count[ai] -= 1
                left_in_bracket[age_by_brackets[ai]] -= 1
            uids_in_school = {}

            log.debug(f"last school, size from distribution: {size}, size generated {len(new_school)}")

        else:
            bi_min = max(0, bindex-1)
            bi_max = bindex + 1

            for i in range(1, size):
                if len(uids_in_school) == 0:
                    break

                # no one left to send? should only choose other students from the mixing matrices, not teachers so don't create schools with
                if sum([left_in_bracket[bi] for bi in range(bi_min, bi_max+1)]) == 0:
                    break

                bi = spsamp.sample_single_arr(b_prob)

                while left_in_bracket[bi] == 0 or np.abs(bindex - bi) > 1:
                    bi = spsamp.sample_single_arr(b_prob)

                ai = spsamp.sample_from_range(ages_in_school_distr, age_brackets[bi][0], age_brackets[bi][-1])
                uid = uids_in_school_by_age[ai][0]  # grab the next student in line

                new_school.append(ai)
                new_school_uids.append(uid)

                uids_in_school_by_age[ai].remove(uid)
                uids_in_school.pop(uid, None)

                ages_in_school_count[ai] -= 1
                ages_in_school_distr = spb.norm_dic(ages_in_school_count)
                left_in_bracket[bi] -= 1

        school_age_lists.append(new_school)
        school_uid_lists.append(new_school_uids)
        school_types.append(None)
        new_school = np.array(new_school)
        kids = new_school <= 19

        if logging.getLevelName(log.level) == 'DEBUG':
            print(f"new school size {len(new_school)}, ages: {sorted(new_school)}, nkids: {kids.sum()}, n20=>: {len(new_school) - kids.sum()}, kid-adult ratio: {np.divide(kids.sum() , (len(new_school) - kids.sum()) )}")

    log.debug(f"people in school {np.sum([len(school) for school in school_age_lists])}, left to send: {len(uids_in_school)}")

    return school_age_lists, school_uid_lists, school_types


def count_enrollment_by_age(popdict):
    """
    Get enrollment count by age for students in the popdict.

    Args:
        popdict (dict): population dictionary

    Returns:
        dict: Dictionary of the count of enrolled students by age in popdict.
    """
    enrollment_count_by_age = dict.fromkeys(np.arange(0, defaults.settings.max_age), 0)
    for i, person in popdict.items():
        if person['scid'] is not None and person['sc_student']:
            enrollment_count_by_age[person['age']] += 1

    return enrollment_count_by_age


def get_enrollment_rates_by_age(enrollment_count_by_age, age_count):
    """
    Get enrollment rates by age.

    Args:
        enrollment_count_by_age (dict) : dictionary of the count of enrolled students
        age_count (dict)               : dictionary of the age count

    Returns:
        dict: Dictionary of the enrollment rates by age.
    """
    return {a: enrollment_count_by_age[a] / age_count[a] if age_count[a] > 0 else 0 for a in sorted(age_count.keys())}


def count_enrollment_by_school_type(popdict, **kwargs):
    """
    Get enrollment sizes by school types in popdict.

    Args:
        popdict (dict)             : population dictionary
        **with_school_types (bool) : If True, return enrollment by school types as defined in the popdict. Otherwise, combine all enrollment sizes for a school type of None.
        **keys_to_exclude (list)   : school types to exclude

    Returns:
        dict: Dictionary of generated enrollment sizes by school type.
    """
    kwargs = sc.objdict(sc.mergedicts(dict(with_school_types=False, keys_to_exclude=[]), kwargs))
    schools = dict()
    enrollment_by_school_type = dict()
    for i, person in popdict.items():
        if person['scid'] is not None and person['sc_student']:
            schools.setdefault(person['scid'], dict())
            schools[person['scid']]['sc_type'] = person['sc_type']
            schools[person['scid']].setdefault('enrolled', 0)
            schools[person['scid']]['enrolled'] += 1

    for i, school_i in schools.items():
        enrollment_by_school_type.setdefault(school_i['sc_type'], [])
        enrollment_by_school_type[school_i['sc_type']].append(school_i['enrolled'])

    if not kwargs.with_school_types:
        sc_types = set(enrollment_by_school_type.keys())
        if None not in sc_types:
            enrollment_by_school_type[None] = []
            for sc_type in set(sc_types.difference(set(kwargs.keys_to_exclude))):
                enrollment_by_school_type[None].extend(enrollment_by_school_type[sc_type])
                enrollment_by_school_type.pop(sc_type, None)

    return enrollment_by_school_type


def get_generated_school_size_distributions(enrollment_by_school_type, bins):
    """
    Get school size distributions by type.

    Args:
        enrollment_by_school_type (dict) : generated enrollment sizes by school types
        bins (list)                      : school size bins

    Returns:
        dict: Dictionary of generated school size distribution by school type.
    """
    generated_school_size_dist = dict()
    for sc_type in enrollment_by_school_type:
        sizes = enrollment_by_school_type[sc_type]
        hist, bins = np.histogram(sizes, bins=bins, density=0)
        if sum(sizes) > 0:
            generated_school_size_dist[sc_type] = {i: hist[i] / sum(hist) for i in range(len(hist))}
        else:
            generated_school_size_dist[sc_type] = {i: hist[i] for i in range(len(hist))}

    return generated_school_size_dist
