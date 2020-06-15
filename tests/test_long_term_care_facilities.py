import sciris as sc
import synthpops as sp

do_plot = False
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

with_school_types = True
average_class_size = 20
inter_grade_mixing = 0.1
average_student_teacher_ratio = 20
average_teacher_teacher_degree = 3
# school_mixing_type = 'random'
# school_mixing_type = 'clustered'
school_mixing_type = {'pk': 'clustered', 'es': 'random', 'ms': 'clustered', 'hs': 'random', 'uv': 'random'}


rand_seed = 1


n = 10e3
n = int(n)

# # generate and write to file
popdict = sp.generate_microstructure_with_facilities(datadir,
                                                     location=location,
                                                     state_location=state_location,
                                                     country_location=country_location,
                                                     gen_pop_size=n,
                                                     write=write,
                                                     do_plot=do_plot,
                                                     return_popdict=return_popdict,
                                                     with_school_types=with_school_types,
                                                     average_class_size=average_class_size,
                                                     inter_grade_mixing=inter_grade_mixing,
                                                     average_student_teacher_ratio=average_student_teacher_ratio,
                                                     average_teacher_teacher_degree=average_teacher_teacher_degree,
                                                     school_mixing_type=school_mixing_type)

# # read in from file
popdict = sp.make_contacts_with_facilities_from_microstructure(datadir,
                                                               location=location,
                                                               state_location=state_location,
                                                               country_location=country_location,
                                                               n=n)

# # generate on the fly
sc.tic()
popdict = sp.make_population(n=n,
                             generate=False,
                             with_facilities=True,
                             with_school_types=with_school_types,
                             school_mixing_type=school_mixing_type,
                             rand_seed=rand_seed)
sc.toc()

sp.check_all_residents_are_connected_to_staff(popdict)
