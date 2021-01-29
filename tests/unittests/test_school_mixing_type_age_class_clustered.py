"""
Test school mixing patterns for age_and_class_clusetred Schools
The assumption is that one or more teachers are assigned to one class only
Students must be assigned to only one class (class is composed of the same group of students)
"""
import copy
import uuid
import synthpops as sp
pars = dict(
    n=2e4,
    rand_seed=1,
    max_contacts=None,
    country_location='usa',
    state_location='Washington',
    location='seattle_metro',
    use_default=True,
    with_non_teaching_staff=1,
    with_school_types=1,
    school_mixing_type={'pk': 'age_and_class_clustered',
                        'es': 'age_and_class_clustered',
                        'ms': 'age_and_class_clustered',
                        'hs': 'random',
                        'uv': 'random'},
    average_class_size=30,
)


def test_age_and_class_clustered():
    """
    Test case for age_and_class_clustered type
    a population based on pars is created and the classes formed are examined against the assumptions
    Returns:
        None
    """
    pop = sp.generate_synthetic_population(**pars)
    # create AgeClassClusetredSchool class objects from population
    # and check if there is overlapping for teachers/students
    schools = form_classes(pop, ['pk', 'es', 'ms'])
    check_class_overlapping(schools)


def form_classes(pop, school_types):
    """
    Args:
        pop: population
        school_types: a list of school type

    Returns:
        An AgeClassClusetredSchool class object
    """

    schools = []
    # loop over population, form classes first by the following logic:
    # if a teacher is found, add him to the class
    # if a student is found, check his school contacts to find the teachers and add both teacher/student to the class
    for k, p in pop.items():
        if p["scid"] is not None and p["sc_type"] in school_types:
            # check if school exists by scid
            schoolp = [s for s in schools if p["scid"] == s.id]
            assert len(schoolp) <= 1, f"there should only be one school with {p['scid']}"
            # if school does not exist, create one with scid and sc_type
            if len(schoolp) == 0:
                schoolp = AgeClassClusetredSchool(sc_id=p["scid"], sc_type=p["sc_type"])
                schools.append(copy.deepcopy(schoolp))
            else:
                # get the first element of school (should have only one school returned)
                schoolp = schoolp[0]
            if p["sc_teacher"] is not None:
                # check all classes in the school with matching scid to see if this teacher was assigned
                classp = None
                for c in schoolp.get_all_classes():
                    if k in c.get_teacher():
                        classp = c
                        break
                # if teacher was not assigned, form a new class and add this class to the school
                if classp is None:
                    classp = AgeClassClusetredClass()
                    classp.add_teacher(k)
                    schoolp.add_class(copy.deepcopy(classp))
            if p["sc_student"] is not None:
                # find the teachers from the student's contacts and check if they already teach a class
                classp = None
                teachersp = [sc for sc in p["contacts"]["S"] if pop[sc]["sc_teacher"] is not None]
                for c in schoolp.get_all_classes():
                    # check all classes to see if teachers' id is in there
                    if set(teachersp).issubset(c.get_teacher()):
                        classp = c
                        break
                # if no such class is found, form the new class with all the teachers in contacts
                # and add the student to the class
                if classp is None:
                    classp = AgeClassClusetredClass()
                    classp.add_teacher(teachersp)
                    classp.add_student(k)
                    schoolp.add_class(copy.deepcopy(classp))
                else:
                    # add student to the class where his teacher contact was assigned
                    classp.add_student(k)
    # print school info
    for s in schools:
        s.print_school()
    return copy.deepcopy(schools)


def check_class_overlapping(schools):
    """
    Verify that there is no overlapping in students and teachers
    Args:
        schools: list of AgeClassClusetredSchool class objects

    Returns:
        None
    """
    setlist_teachers = []
    setlist_students = []
    # get teachers and students from each class to a list of set items
    for s in schools:
        for c in s.get_all_classes():
            setlist_teachers.append(c.get_teacher())
            setlist_students.append(c.get_student())
    # check if any overlapping exists in teachers/students
    dup_teacher = set.intersection(*setlist_teachers)
    dup_student = set.intersection(*setlist_students)
    assert len(dup_teacher) == 0, f"overlapped teachers: {dup_teacher}"
    assert len(dup_student) == 0, f"overlapped student: {dup_student}"


class AgeClassClusetredSchool:

    def __init__(self, sc_id=None, sc_type=None):
        """
        class constructor

        Args:
            sc_id: school id, if not provided a random uuid will be assigned
            sc_type: school type, default to None
        """
        self.id = uuid.uuid4() if sc_id is None else sc_id
        self.sc_type = sc_type
        self.classrooms = set()

    def add_class(self, newclass):
        """
        add a new class to school

        Args:
            newclass: An AgeClassClusetredClass class object

        Returns:
            None
        """
        self.classrooms.add(newclass)

    def get_all_classes(self):
        """
        get all the classes in the school

        Returns:
            A set of AgeClassClusetredClass class objects
        """
        return self.classrooms

    def print_school(self):
        """
        method to print the school info

        Returns:
            None
        """
        print(f"\n------{self.get_type()}:{self.id}-------")
        print(f"\n------{len(self.get_all_classes())} classes-------")
        for c in self.get_all_classes():
            c.print_class()

    def get_type(self):
        """
        Get full name of school type

        Returns:
            A human-readable name for school type
        """
        st = {
            "ms": "middle school",
            "pk": "preschool",
            "es": "elementary school",
            "hs": "high school",
            "uv": "university"}
        return st.get(self.sc_type)


class AgeClassClusetredClass:

    def __init__(self):
        """
        Constructor: create an empty class
        """
        self.id = uuid.uuid4()
        self.teachers = set()
        self.students = set()

    def add_teacher(self, pid):
        """
        Add teacher / teachers to class

        Args:
            pid: a teacher's id or a list of teachers' id
        Returns:
            None
        """
        if type(pid) is list:
            self.teachers.update(pid)
        else:
            self.teachers.add(pid)

    def add_student(self, pid):
        """
        Add student / students to class

        Args:
            pid: a student's id or a list of students' id

        Returns:
            None
        """
        if type(pid) is list:
            self.students.update(pid)
        else:
            self.students.add(pid)

    def get_teacher(self):
        """
        get teachers in the school

        Returns:
            A set of teachers' ids
        """
        return self.teachers

    def get_student(self):
        """
        get students in the school

        Returns:
            A set of students' ids
        """
        return self.students

    def print_class(self):
        """
        print method of class

        Returns:
            None
        """
        print(f"\n\t------Class:{self.id}-------")
        print(f"\n\t------{len(self.get_teacher())} teachers and {len(self.get_student())} students-------")
        print(f"\tteachers:{self.get_teacher()}")
        print(f"\tstudents:{self.get_student()}")
