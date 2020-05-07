=======================
Workplace contact layer
=======================

The :term:`workplace contact layer` represents all of the pairwise connections between people in
workplaces, except for teachers working in schools. After some workers are assigned to the
:term:`school contact layer` as teachers, all remaining workers are assigned to workplaces.
Workplaces are special in that there is little/no age structure so workers of all ages may be
present in every workplace.

Again, note that work and school are currently exclusive, because the people attending schools are
removed from the list of eligible workers. This doesn't necessarily need to be the case though. In
fact, we know that in any countries and cultures around the world, people take on multiple roles as
both students and workers, either part-time or full-time in one or both activities.


Data required
=============

The following data are required for generating the workplace contact layer:

#.  **Workplace size distribution** - again, this gets normalized so can be specified as absolute
    counts or as normalized values::

        work_size_bracket , size_count
        1-4               , 2947
        5-9               , 992
        10-19             , 639
        20-49             , 430
        50-99             , 140
        100-249           , 83
        250-499           , 26
        500-999           , 13
        1000-1999         , 12

#.  **Work contact matrix** specifying the number/weight of contacts by age bin. This is similar to
    the household contact matrix. For example::

                20-30       , 30-40       , 40-50
        20-30   0.659867911 , 0.503965302 , 0.214772978
        30-40   0.314776879 , 0.895460015 , 0.412465791
        40-50   0.132821425 , 0.405073038 , 1.433888594

Workflow
========

1.  :py:func:`~synthpops.contact_networks.generate_workplace_sizes` generates workplace sizes according
    to the workplace size distribution until the number of workers is reached.
2.  :py:func:`~synthpops.contact_networks.assign_rest_of_workers` populates workplaces just like for
    households and schools: randomly selecting the age of a reference person, and then sampling the
    rest of the workplace using the contact matrix.

.. _Mossong et al. 2008: https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.0050074
.. _Fumanelli et al. 2012: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002673
.. _Prem et al. 2017: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005697
.. _Mistry et al. 2020: https://arxiv.org/abs/2003.01214