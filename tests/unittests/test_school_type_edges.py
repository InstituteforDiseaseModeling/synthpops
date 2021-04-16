import synthpops as sp
import numpy as np

do_plot = False
# write = True
write = False
return_popdict = True
use_default = False

# datadir = sp.datadir
datadir = sp.settings_config.datadir
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
school_mixing_type = {'pk': 'age_clustered', 'es': 'random', 'ms': 'age_clustered', 'hs': 'random', 'uv': 'random'}


rand_seed = 1

n = 15e3

# popdict = sp.make_population(n=n,
#                              generate=True,
#                              with_facilities=True,
#                              with_school_types=with_school_types,
#                              school_mixing_type=school_mixing_type,
#                              rand_seed=rand_seed)


# popdict = sp.make_population(n=n,
#                              generate=True,
#                              with_facilities=True,
#                              with_school_types=with_school_types,
#                              school_mixing_type='random',
#                              rand_seed=rand_seed)


# popdict = sp.make_population(n=n,
#                              generate=True,
#                              with_facilities=True,
#                              with_school_types=with_school_types,
#                              school_mixing_type='age_clustered',
#                              rand_seed=rand_seed)

popdict = sp.make_population(n=n,
                             generate=True,
                             with_facilities=True,
                             with_school_types=with_school_types,
                             school_mixing_type='age_and_class_clustered',
                             rand_seed=rand_seed)

