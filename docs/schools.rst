====================
School contact layer
====================

The :term:`school contact layer` represents all of the pairwise connections between
people in schools, including both students and teachers. Schools are special in that:

-   Enrollment rates by age determine the probability of individual being a student given their age.
-   Staff members such as teachers are chosen from individuals determined to be in the adult labor force.
-   The current methods in |SP| treat student and worker status as mutually exclusive. Many young adults
    may be both students and workers, part time or full time in either status. The ability to select
    individuals to participate in both activities will be introduced in a later version of the model.

.. TBD make sure this gets updated when the functionality is added

Data needed
===========

The following data is required for schools:

#.  **School size distribution**::

        school_size , percent
        0-50        , 0.2
        51-100      , 0.1
        101-300     , 0.3

#.  **Enrollment by age** specifying the percentage of people of each age attending school.
    See :py:func:`~synthpops.contact_networks.get_school_enrollment_rates`, but note that this mainly
    implements parsing a Seattle-specific data file to produce the following data structure, which could
    equivalently be read directly from a file::

        age , percent
        0   , 0
        1   , 0
        2   , 0
        3   , 0.529
        4   , 0.529
        5   , 0.95
        6   , 0.95
        7   , 0.95
        8   , 0.95
        9   , 0.95
        10  , 0.987
        11  , 0.987
        12  , 0.987
        13  , 0.987

#.  **School contact matrix** specifying the number/weight of contacts by age bin. This is similar to the
    household contact matrix. For example::

                0-10        , 10-20       , 20-30
        0-10    0.659867911 , 0.503965302 , 0.214772978
        10-20   0.314776879 , 0.895460015 , 0.412465791
        20-30   0.132821425 , 0.405073038 , 1.433888594

#.  **Employment rates by age**, which is used when determining who is in the labor force, and thus
    which adults are available to be chosen as teachers::

        Age , Percent
        16  , 0.496
        17  , 0.496
        18  , 0.496
        19  , 0.496
        20  , 0.838
        21  , 0.838
        22  , 0.838

#.  **Student teacher ratio**, which is the average ratio for the location. Methods to use a
    distribution or vary the ratio for different types of schools may come in later developments of
    the model::

        student_teacher_ratio=30

Typically, contact matrices describing age-specific mixing patterns in schools include the
interactions between students and their teachers. These patterns describe multiple types of
schools, from possibly preschools to universities.

Workflow
========


Use these |SP| functions to implement the school contact layer as follows:

#.  :py:func:`~synthpops.contact_networks.get_uids_in_school` uses the enrollment rates to determine
    which people attend school. This then provides the number of students needing to be assigned to schools.
#.  :py:func:`~synthpops.contact_networks.generate_school_sizes` generates schools according to the
    school size distribution until there are enough places for every student to be assigned a school.
#.  :py:func:`~synthpops.contact_networks.send_students_to_school` assigns specific students to
    specific schools.

    -   This function is similar to households in that a reference student is selected, and then the contact
        matrix is used to fill the remaining spots in the school.

    -   Some particulars in this function deal with ensuring a teacher/adult is less likely to be selected
        as a reference person, and restricting the age range of sampled people relative to the reference
        person so that a primary school age reference person will result in the rest of the school being
        populated with other primary school age children

#.  :py:func:`~synthpops.contact_networks.get_uids_potential_workers` selects teachers by first
    getting a pool of working age people that are not students.
#.  :py:func:`~synthpops.contact_networks.get_workers_by_age_to_assign` further filters this population
    by employment rates resulting in a collection of people that need to be assigned workplaces.
#.  In :py:func:`~synthpops.contact_networks.assign_teachers_to_work`, for each school, work out how
    many teachers are needed according to the number of students and the student-teacher ratio, and
    sample those teachers from the pool of adult workers. A minimum and maximum age for teachers can
    be provided to select teachers from a specified range of ages (this can be used to account for
    the additional years of education needed to become a teacher in many places).

.. _Mossong et al. 2008: https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.0050074
.. _Fumanelli et al. 2012: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002673
.. _Prem et al. 2017: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005697
.. _Mistry et al. 2020: https://arxiv.org/abs/2003.01214