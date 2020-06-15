"""
Test generation of a synthetic population with microstructure.
"""

import synthpops as sp


if __name__ == '__main__':

    datadir = sp.datadir

    location = 'seattle_metro'
    state_location = 'Washington'
    country_location = 'usa'
    sheet_name = 'United States of America'

    n = 0.5e3
    n = int(n)
    verbose = False
    plot = False
    write = False
    return_popdict = True
    with_school_types = True
    average_class_size = 20
    inter_grade_mixing = 0.1
    average_student_teacher_ratio = 20
    average_teacher_teacher_degree = 3
    # school_mixing_type = 'random'
    # school_mixing_type = 'clustered'
    school_mixing_type = {'pk': 'clustered', 'es': 'random', 'ms': 'clustered', 'hs': 'random', 'uv': 'random'}

    population = sp.generate_synthetic_population(n, datadir, location=location,
                                                  state_location=state_location,
                                                  country_location=country_location,
                                                  sheet_name=sheet_name,
                                                  with_school_types=with_school_types,
                                                  average_class_size=average_class_size,
                                                  inter_grade_mixing=inter_grade_mixing,
                                                  average_student_teacher_ratio=average_student_teacher_ratio,
                                                  average_teacher_teacher_degree=average_teacher_teacher_degree,
                                                  school_mixing_type=school_mixing_type,
                                                  verbose=verbose,
                                                  plot=plot,
                                                  write=write,
                                                  return_popdict=return_popdict)

    population = sp.make_population(n,
                                    generate=True,
                                    with_school_types=True,
                                    school_mixing_type=school_mixing_type)

#     for i in range(50):
#         person = population[i]
#         ha = [population[c]['age'] for c in person['contacts']['H']]
#         # print(i, person['age'], ha, person['contacts']['H'])
#         if len(person['contacts']['S']) > 0:
#           sa = [population[c]['age'] for c in person['contacts']['S']]
#           # print(i, person['age'], 'S', sa, person['contacts']['S'])
#           print(i, person['age'], person['scid'], 'S', sa)
