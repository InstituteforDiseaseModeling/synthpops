"""
Test school mixing patterns for age_and_class_clustered Schools. The assumption
is that one or more teachers are assigned to one class only. Students must be
assigned to only one class (class is composed of the same group of students).
Note that AgeClassClusteredClass and AgeClassClusteredSchool classes are for
test use only.

"""
import copy
import itertools
import networkx as nx
import sciris as sc
import synthpops as sp
import matplotlib.pyplot as plt

pars = dict(
    rand_seed               = 1,
    max_contacts            = None,
    country_location        = 'usa',
    state_location          = 'Washington',
    location                = 'seattle_metro',
    use_default             = True,
    with_non_teaching_staff = 1,
    with_school_types       = 1,
    school_mixing_type      = {'pk': 'age_and_class_clustered',
                               'es': 'age_and_class_clustered',
                               'ms': 'age_and_class_clustered',
                               'hs': 'random',
                               'uv': 'random'},
    average_class_size      = 30,
)


def test_age_and_class_clustered(do_show=False):
    """
    Test case for age_and_class_clustered type. A population based on pars is
    created and the classes formed are examined against the assumptions.

    Returns:
        None
    """
    pop = sp.Pop(n=15000, **pars)
    # create AgeClassClusteredSchool class objects from population
    # and check if there is overlapping for teachers/students
    schools = form_classes(pop.popdict, ['pk', 'es', 'ms'])
    check_class_overlapping(schools)
    check_classes_disjoint(schools, do_show=do_show)


def form_classes(popdict, school_types):
    """
    Args:
        popdict (dict)      : popdict of a Pop object
        school_types (list) : a list of school type

    Returns:
        An AgeClassClusteredSchool class object
    """

    # construct a list of schools which are objects of AgeClassClusteredSchool class
    schools = []

    # loop over population, form classes first by the following logic:
    # if a teacher is found, add them to the class
    # if a student is found, check their school contacts to find the teachers and add both teacher/student to the class
    for uid, person in popdict.items():

        if person["scid"] is not None and person["sc_type"] in school_types:

            # check if school exists by scid and return a list of schools
            school_scid = [s for s in schools if person["scid"] == s.scid]

            # throw an error if multiple schools returned as each school should have unique scid
            assert len(school_scid) <= 1, f"there should only be one school with {person['scid']}"

            # if school does not exist, create one object of AgeClassClusteredSchool with scid and sc_type
            # and add it to the list of schools
            if len(school_scid) == 0:
                schoolp = AgeClassClusteredSchool(scid=person["scid"], sc_type=person["sc_type"])
                schools.append(copy.deepcopy(schoolp))
            else:
                # get the first element of school (should have only one school returned)
                schoolp = school_scid[0]

            if person["sc_teacher"] is not None:

                # check all classes in the school with matching scid to see if this teacher was assigned
                classp = None
                for c in schoolp.classrooms:
                    if uid in c.teachers:
                        classp = c
                        break

                # if teacher was not assigned, form a new class and add this class to the school
                if classp is None:
                    classp = AgeClassClusteredClass()
                    classp.add_teacher(uid)
                    schoolp.classrooms.add(copy.deepcopy(classp))

            if person["sc_student"] is not None:
                # find the teachers from the student's contacts and check if they already teach a class
                classp = None
                teachersp = [sc for sc in person["contacts"]["S"] if popdict[sc]["sc_teacher"] is not None]

                # check all classes to see if teachers' id is in there
                classfound = [c for c in schoolp.classrooms if set(teachersp).intersection(c.teachers)]

                # if no such class is found, form the new class with all the teachers in contacts
                # and add the student to the class
                if len(classfound) == 0:
                    classp = AgeClassClusteredClass()
                    classp.add_teacher(teachersp)
                    classp.add_student(uid)
                    schoolp.classrooms.add(copy.deepcopy(classp))
                else:
                    classp = classfound[0]
                    # if more than 2 classes are found, merge the class
                    if len(classfound) > 1:
                        classp = schoolp.merge_classes(classfound)

                    # add student to the class where his teacher contact was assigned
                    # also add all the teachers of this student to the same class
                    classp.add_student(uid)
                    classp.add_teacher(teachersp)

    # print school info
    for s in schools:
        s.print_school()

    return sc.dcp(schools)


def check_class_overlapping(schools):
    """
    Verify that there is no overlapping in students and teachers.

    Args:
        schools (list): list of AgeClassClusteredSchool class objects

    Returns:
        None
    """
    setlist_teachers = []
    setlist_students = []

    # get teachers and students from each class to a list of items,
    # if there is overlapping an item should appear more than once
    for s in schools:
        for c in s.classrooms:
            setlist_teachers = setlist_teachers + list(c.teachers)
            setlist_students = setlist_students + list(c.students)

    # check if any overlapping exists in teachers/students
    dup_teacher = set([i for i in setlist_teachers if setlist_teachers.count(i) > 1])
    dup_student = set([i for i in setlist_students if setlist_students.count(i) > 1])

    assert len(dup_teacher) == 0, f"overlapped teachers: {dup_teacher}"
    assert len(dup_student) == 0, f"overlapped student: {dup_student}"


def check_classes_disjoint(schools, do_show=False):
    """
    Check if schools are formed by disjoint classes component.

    Args:
        schools (list) : list of AgeClassClusteredSchool class objects
        do_show (bool) : If True, draw the network of classes. Default to False.

    Returns:
        None
    """
    gschools = nx.Graph()
    for s in schools:
        # the connected component of the graph of each school should be each individual class
        graph_component = [g for g in nx.connected_components(s.to_graph())]
        assert len(graph_component) == len(s.classrooms), \
            f"school {str(s.scid)} has {str(len(s.classrooms))} classes but {str(len(graph_component))} components"

        # For union operation, graphs must be disjoint, otherwise an exception is raised.
        gschools = nx.union(gschools, s.to_graph())
    if do_show:
        plt.plot()
        nx.draw(gschools, with_labels=False)
        plt.show()


class AgeClassClusteredSchool:
    """
    This class is used by test only
    """
    def __init__(self, scid, sc_type=None):
        """
        class constructor

        Args:
            scid (int)    : school id
            sc_type (str) : school type, default to None
        """
        self.scid = scid
        self.sc_type = sc_type
        self.classrooms = set()

    def print_school(self):
        """
        Method to print the school info

        Returns:
            None
        """
        print(f"\n------{self.sc_type} SCID:{self.scid}-------")
        print(f"\n------{len(self.classrooms)} classes-------")
        for c in self.classrooms:
            c.print_class()

    def remove_classes_by_id(self, cid):
        """
        Remove the class from school by the class id.

        Args:
            cid (uuid): class id to identify class to be removed

        Returns:
            None
        """
        target = [c for c in self.classrooms if c.id == cid]
        for t in target:
            self.classrooms.remove(t)

    def merge_classes(self, c):
        """
        Merge a list of classes to a new class and update the school.

        Args:
            c (list): list of AgeClassClusteredClass objects to be merged

        Returns:
            A single merged AgeClassClusteredClass object.
        """
        newc = AgeClassClusteredClass()
        for i in c:
            newc.teachers.update(i.teachers)
            newc.students.update(i.students)
            self.remove_classes_by_id(i.id)
        newclass = copy.deepcopy(newc)
        self.classrooms.add(newclass)

        return newclass

    def to_graph(self):
        """
        Generate NetworkX graph representation of the school.

        Returns:
            An undirected NetworkX graph
        """
        gs = nx.Graph()
        for c in self.classrooms:
            gs.add_nodes_from(c.to_graph())
            gs.add_edges_from(c.to_graph().edges)

        return gs


class AgeClassClusteredClass:
    """
        This class is used by test only.
    """
    def __init__(self):
        """
        Constructor: create an empty class.
        """
        self.id = sc.uuid()
        self.teachers = set()
        self.students = set()

    def add_teacher(self, pid):
        """
        Add teacher / teachers to class.

        Args:
            pid (int): a teacher's id or a list of teachers' id

        Returns:
            None
        """
        self.teachers.update(sc.promotetolist(pid))

    def add_student(self, pid):
        """
        Add student / students to class.

        Args:
            pid (int): a student's id or a list of students' id

        Returns:
            None
        """
        self.students.update(sc.promotetolist(pid))

    def print_class(self):
        """
        Print method of class.

        Returns:
            None
        """
        print(f"\n\t------Class:{self.id}-------")
        print(f"\n\t------{len(self.teachers)} teachers and {len(self.students)} students-------")
        print(f"\tteachers:{self.teachers}")
        print(f"\tstudents:{self.students}")

    def to_graph(self):
        """
        Generate NetworkX graph representation of the class.

        Returns:
            An undirected NetworkX graph.
        """
        g = nx.Graph()
        for t in self.teachers:
            g.add_nodes_from([(t, {"position": "teacher"})])
        for s in self.students:
            g.add_nodes_from([(s, {"position": "student"})])
        g.add_edges_from(list(itertools.combinations(list(self.teachers)+list(self.students), 2)))

        return g
