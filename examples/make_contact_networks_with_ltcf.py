import sciris as sc
import synthpops as sp

plot = False
verbose = False
# write = True
write = False
return_popdict = True
use_default = False

datadir = sp.datadir
country_location = 'usa'
state_location = 'Washington'
location = 'seattle_metro'
sheet_name = 'United States of America'

use_two_group_reduction = True
average_LTCF_degree = 20
ltcf_staff_age_min = 20
ltcf_staff_age_max = 60


with_school_types = True
average_class_size = 20
inter_grade_mixing = 0.1
average_student_teacher_ratio = 20
average_teacher_teacher_degree = 3
teacher_age_min = 25
teacher_age_max = 75

with_non_teaching_staff = True
# if with_non_teaching_staff is False, but generate is True, then average_all_staff_ratio should be average_student_teacher_ratio or 0
average_student_all_staff_ratio = 11
average_additional_staff_degree = 20
staff_age_min = 20
staff_age_max = 75

school_mixing_type = 'random'  # randomly mixing across the entire school
school_mixing_type = 'age_clustered'  # age_clustered means mixing across your own age/grade but randomly so students are not cohorted into classrooms but also don't mix much with other ages
school_mixing_type = 'age_and_class_clustered'  # age_and_class_clustered means mixing strictly with your own class. Each class gets at least 1 teacher. Students don't mix with students from other classes.

school_mixing_type = {'pk': 'age_and_class_clustered', 'es': 'random', 'ms': 'age_and_class_clustered', 'hs': 'random', 'uv': 'random'}
school_mixing_type = {'pk': 'age_clustered', 'es': 'random', 'ms': 'age_clustered', 'hs': 'random', 'uv': 'random'}
school_mixing_type = {'pk': 'random', 'es': 'random', 'ms': 'random', 'hs': 'random', 'uv': 'random'}

rand_seed = 1


n = 1000
n = int(n)

# # # generate and write to file
population = sp.generate_microstructure_with_facilities(datadir,
                                                        location=location,
                                                        state_location=state_location,
                                                        country_location=country_location,
                                                        n=n,
                                                        use_two_group_reduction=use_two_group_reduction,
                                                        average_LTCF_degree=average_LTCF_degree,
                                                        ltcf_staff_age_min=ltcf_staff_age_min,
                                                        ltcf_staff_age_max=ltcf_staff_age_max,
                                                        with_school_types=with_school_types,
                                                        school_mixing_type=school_mixing_type,
                                                        average_class_size=average_class_size,
                                                        inter_grade_mixing=inter_grade_mixing,
                                                        average_student_teacher_ratio=average_student_teacher_ratio,
                                                        average_teacher_teacher_degree=average_teacher_teacher_degree,
                                                        teacher_age_min=teacher_age_min,
                                                        teacher_age_max=teacher_age_max,
                                                        average_student_all_staff_ratio=average_student_all_staff_ratio,
                                                        average_additional_staff_degree=average_additional_staff_degree,
                                                        staff_age_min=staff_age_min,
                                                        staff_age_max=staff_age_max,
                                                        write=write,
                                                        plot=plot,
                                                        return_popdict=return_popdict,
                                                        use_default=use_default)

# # # read in from file
population = sp.make_contacts_with_facilities_from_microstructure(datadir,
                                                                  location=location,
                                                                  state_location=state_location,
                                                                  country_location=country_location,
                                                                  n=n,
                                                                  use_two_group_reduction=use_two_group_reduction,
                                                                  average_LTCF_degree=average_LTCF_degree,
                                                                  with_school_types=with_school_types,
                                                                  school_mixing_type=school_mixing_type,
                                                                  average_class_size=average_class_size,
                                                                  inter_grade_mixing=inter_grade_mixing,
                                                                  average_student_teacher_ratio=average_student_teacher_ratio,
                                                                  average_teacher_teacher_degree=average_teacher_teacher_degree,
                                                                  average_student_all_staff_ratio=average_student_all_staff_ratio,
                                                                  average_additional_staff_degree=average_additional_staff_degree)

# # generate on the fly
sc.tic()
population = sp.make_population(n=n,
                                generate=True,
                                with_facilities=True,
                                use_two_group_reduction=use_two_group_reduction,
                                average_LTCF_degree=average_LTCF_degree,
                                ltcf_staff_age_min=ltcf_staff_age_min,
                                ltcf_staff_age_max=ltcf_staff_age_max,
                                with_school_types=with_school_types,
                                school_mixing_type=school_mixing_type,
                                average_class_size=average_class_size,
                                inter_grade_mixing=inter_grade_mixing,
                                average_student_teacher_ratio=average_student_teacher_ratio,
                                average_teacher_teacher_degree=average_teacher_teacher_degree,
                                teacher_age_min=teacher_age_min,
                                teacher_age_max=teacher_age_max,
                                with_non_teaching_staff=with_non_teaching_staff,
                                average_student_all_staff_ratio=average_student_all_staff_ratio,
                                average_additional_staff_degree=average_additional_staff_degree,
                                staff_age_min=staff_age_min,
                                staff_age_max=staff_age_max,
                                rand_seed=rand_seed)
sc.toc()

sp.check_all_residents_are_connected_to_staff(population)


check = True
check = False

# check students in ES, MS, HS
# check teachers and staff in ES, MS, HS
# check the edges in these schools

if check:
    schools = {'es': {'students': 0, 'teachers': 0, 'staff': 0, 'ns': 0},
               'ms': {'students': 0, 'teachers': 0, 'staff': 0, 'ns': 0},
               'hs': {'students': 0, 'teachers': 0, 'staff': 0, 'ns': 0},
               }

    print('counting schools')

    n_school_edges = 0
    for i in population:
        person = population[i]

        if person['scid'] is not None:
            if person['sc_type'] in ['es', 'ms', 'hs']:
                if person['sc_student']:
                    schools[person['sc_type']]['students'] += 1
                elif person['sc_teacher']:
                    schools[person['sc_type']]['teachers'] += 1
                elif person['sc_staff']:
                    schools[person['sc_type']]['staff'] += 1
            n_school_edges += len(person['contacts']['S'])

        # print(i, person['scid'], person['sc_student'], person['sc_teacher'], person['sc_staff'])
    print('edges in schools', n_school_edges)
    print(schools)
