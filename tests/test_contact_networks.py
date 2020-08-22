"""
Test generation of a synthetic population with microstructure, reading from file, and using sp.make_population to do both.
"""

import synthpops as sp


if __name__ == '__main__':

    datadir = sp.datadir

    location = 'seattle_metro'
    state_location = 'Washington'
    country_location = 'usa'
    sheet_name = 'United States of America'

    n = 10000
    n = int(n)
    verbose = False
    plot = False
    write = True
    return_popdict = True
    with_school_types = True
    average_class_size = 20
    inter_grade_mixing = 0.1
    average_student_teacher_ratio = 20
    average_teacher_teacher_degree = 3
    average_student_all_staff_ratio = 10
    average_additional_staff_degree = 20
    # school_mixing_type = 'random'
    # school_mixing_type = 'clustered'
    school_mixing_type = {'pk': 'clustered', 'es': 'random', 'ms': 'clustered', 'hs': 'random', 'uv': 'random'}

    # population = sp.generate_synthetic_population(n, datadir, location=location,
    #                                               state_location=state_location,
    #                                               country_location=country_location,
    #                                               sheet_name=sheet_name,
    #                                               with_school_types=with_school_types,
    #                                               school_mixing_type=school_mixing_type,
    #                                               average_class_size=average_class_size,
    #                                               inter_grade_mixing=inter_grade_mixing,
    #                                               average_student_teacher_ratio=average_student_teacher_ratio,
    #                                               average_teacher_teacher_degree=average_teacher_teacher_degree,
    #                                               average_student_all_staff_ratio=average_student_all_staff_ratio,
    #                                               average_additional_staff_degree=average_additional_staff_degree,
    #                                               verbose=verbose,
    #                                               plot=plot,
    #                                               write=write,
    #                                               return_popdict=return_popdict)

    population = sp.make_contacts_from_microstructure(datadir,
                                                      location,
                                                      state_location,
                                                      country_location,
                                                      n,
                                                      with_non_teaching_staff=True,
                                                      with_school_types=False,
                                                      school_mixing_type=school_mixing_type,
                                                      average_class_size=average_class_size,
                                                      inter_grade_mixing=inter_grade_mixing,
                                                      average_student_teacher_ratio=average_student_teacher_ratio,
                                                      average_teacher_teacher_degree=average_teacher_teacher_degree,
                                                      average_student_all_staff_ratio=average_student_all_staff_ratio,
                                                      average_additional_staff_degree=average_additional_staff_degree,
                                                      )

    sp.show_layers(population)

    # sp.generate_synthetic_population(n, datadir, location='seattle_metro', state_location='Washington', country_location='usa', sheet_name='United States of America', 
    #                               school_enrollment_counts_available=False, with_school_types=False, school_mixing_type='random', average_class_size=20, inter_grade_mixing=0.1, 
    #                               average_student_teacher_ratio=20, average_teacher_teacher_degree=3, teacher_age_min=25, teacher_age_max=75, 
    #                               average_student_all_staff_ratio=15, average_additional_staff_degree=20, staff_age_min=20, staff_age_max=75, 
    #                               verbose=False, plot=False, write=False, return_popdict=False, use_default=False)

    # population = sp.make_population(n,
    #                                 generate=True,
    #                                 with_school_types=True,
    #                                 school_mixing_type=school_mixing_type)

    for i in range(3000):
        person = population[i]
        if person['sc_staff']:
          print(i, person['sc_staff'])
#         ha = [population[c]['age'] for c in person['contacts']['H']]
#         # print(i, person['age'], ha, person['contacts']['H'])
#         if len(person['contacts']['S']) > 0:
#           sa = [population[c]['age'] for c in person['contacts']['S']]
#           # print(i, person['age'], 'S', sa, person['contacts']['S'])
#           print(i, person['age'], person['scid'], 'S', sa)
