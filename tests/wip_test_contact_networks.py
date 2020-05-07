""" Testing ground for creating a neighbourhoood like population in the Seattle metro area. """

import synthpops as sp
import sciris as sc
import numpy as np
import pandas as pd
import functools
import math
import os, sys
from copy import deepcopy


import matplotlib as mplt
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.ticker import LogLocator, LogFormatter
import matplotlib.font_manager as font_manager
import os, sys
from copy import deepcopy
from collections import Counter
import cmocean
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Set user-specific configurations
username = os.path.split(os.path.expanduser('~'))[-1]
fontdirdict = {
    'dmistry': '/home/dmistry/Dropbox (IDM)/GoogleFonts',
    'cliffk': '/home/cliffk/idm/covid-19/GoogleFonts',
}
if username not in fontdirdict:
    fontdirdict[username] = os.path.expanduser(os.path.expanduser('~'),'Dropbox','GoogleFonts')

try:
    fontpath = fontdirdict[username]
    font_style = 'Roboto_Condensed'
    fontstyle_path = os.path.join(fontpath,font_style,font_style.replace('_','') + '-Light.ttf')
    prop = font_manager.FontProperties(fname = fontstyle_path)
    mplt.rcParams['font.family'] = prop.get_name()
except:
    print("You don't have access to the nice fonts folder mate.")

cmap = mplt.cm.get_cmap(cmocean.cm.deep_r)
cmap2 = mplt.cm.get_cmap(cmocean.cm.curl_r)
cmap3 = mplt.cm.get_cmap(cmocean.cm.matter)


datadir = sp.datadir

state_location = 'Washington'
location = 'seattle_metro'
country_location = 'usa'
sheet_name = 'United States of America'

age_brackets = sp.get_census_age_brackets(datadir,state_location,country_location)
age_by_brackets_dic = sp.get_age_by_brackets_dic(age_brackets)

num_agebrackets = 18
contact_matrix_dic = sp.get_contact_matrix_dic(datadir,sheet_name)


household_size_distr = sp.get_household_size_distr(datadir,location,state_location,country_location)
print(household_size_distr)



# Nhomes = 100000
# Nhomes = 10000
Nhomes = 6666

create_homes = True
# create_homes = False
if create_homes:
    household_size_distr = sp.get_household_size_distr(datadir,location,state_location,country_location)
    print('size distr',household_size_distr)

    Nhomes_to_sample_smooth = 100000
    hh_sizes = sp.generate_household_sizes(Nhomes_to_sample_smooth,household_size_distr)
    totalpop = sp.get_totalpopsize_from_household_sizes(hh_sizes)
    # hh_sizes = sp.generate_household_sizes(Nhomes,household_size_distr)

    syn_ages,syn_sexes = sp.get_usa_age_sex_n(datadir,location,state_location,country_location,totalpop)
    syn_age_count = Counter(syn_ages)
    syn_age_distr = sp.norm_dic(Counter(syn_ages))

    N = Nhomes
    hh_sizes = sp.generate_household_sizes_from_fixed_pop_size(N,household_size_distr)
    totalpop = sp.get_totalpopsize_from_household_sizes(hh_sizes)

    print(totalpop,'pop')

    hha_df = sp.get_household_head_age_by_size_df(datadir,state_location,country_location)
    hha_brackets = sp.get_head_age_brackets(datadir,country_location=country_location)
    hha_by_size = sp.get_head_age_by_size_distr(datadir,country_location=country_location)

    homes_dic,homes = sp.generate_all_households(N,hh_sizes,hha_by_size,hha_brackets,age_brackets,age_by_brackets_dic,contact_matrix_dic,deepcopy(syn_age_distr))
    homes_by_uids, age_by_uid_dic = sp.assign_uids_by_homes(homes)
    new_ages_count = Counter(age_by_uid_dic.values())

    fig = plt.figure(figsize = (6,4))
    ax = fig.add_subplot(111)

    x = np.arange(100)
    y_exp = np.zeros(100)
    y_sim = np.zeros(100)

    for a in range(100):
        expected = int(syn_age_distr[a] * totalpop)
        y_exp[a] = expected
        y_sim[a] = new_ages_count[a]

    ax.plot(x,y_exp,color = cmap(0.2), label = 'Expected')
    ax.plot(x,y_sim,color = cmap3(0.6), label = 'Simulated')
    leg = ax.legend(fontsize = 18)
    leg.draw_frame(False)
    ax.set_xlim(left = 0, right = 100)

    # fig.savefig('synthetic_age_comparison.pdf',format = 'pdf')
    # plt.show()

    sp.write_homes_by_age_and_uid(datadir,location,state_location,country_location,homes_by_uids,age_by_uid_dic)


age_by_uid_dic = sp.read_in_age_by_uid(datadir,location,state_location,country_location,Nhomes)

level = 'county'
sc_df = sp.get_school_enrollment_rates_df(datadir,location,state_location,level)
rates_by_age = sp.get_school_enrollment_rates(datadir,location,state_location,level)

school_sizes_count = sp.get_school_sizes_by_bracket(datadir,location,state_location)

# print(school_sizes_count)

uids_by_age_dic = sp.get_ids_by_age_dic(age_by_uid_dic)
# for a in sorted(uids_by_age_dic):
    # print(a, len(uids_by_age_dic[a]))

create_work_and_school = True
# create_work_and_school = False
if create_work_and_school:

    uids_in_school,uids_in_school_by_age,ages_in_school_count = sp.get_uids_in_school(datadir,location,state_location,country_location,'county',Nhomes)

    # print(uids_in_school_by_age)
    # print(uids_in_school)

    gen_school_sizes = sp.generate_school_sizes(school_sizes_count,uids_in_school)
    # print(gen_school_sizes)
    # print(gen_school_sizes)
    # print(age_brackets)
    # print(age_by_brackets_dic[34])

    gen_schools,gen_school_uids = sp.send_students_to_school(gen_school_sizes,uids_in_school,uids_in_school_by_age,ages_in_school_count,age_brackets,age_by_brackets_dic,contact_matrix_dic)
    # print('gen',gen_schools)
    for gn,g in enumerate(gen_schools):
        print(gn,'school','\n',sorted(g))
    # for s in range(5):
        # print(Counter(gen_schools[s]))
        # print(gen_schools[s])

    emp_rates = sp.get_employment_rates(datadir,location,state_location,country_location)
    # print(emp_rates)
    potential_worker_uids,potential_worker_uids_by_age,potential_worker_ages_left_count = sp.get_uids_potential_workers(uids_in_school,uids_in_school_by_age,age_by_uid_dic)
    workers_by_age_to_assign_count = sp.get_workers_by_age_to_assign(emp_rates,potential_worker_ages_left_count,uids_by_age_dic)
    # print(len(potential_worker_uids))
    gen_schools,gen_school_uids,potential_worker_uids,potential_worker_uids_by_age,workers_by_age_to_assign_count = sp.assign_teachers_to_work(gen_schools,gen_school_uids,emp_rates,workers_by_age_to_assign_count,potential_worker_uids,potential_worker_uids_by_age,potential_worker_ages_left_count)
    # print(len(potential_worker_uids))

    # for a in potential_worker_uids_by_age:
        # print(a, len(potential_worker_uids_by_age[a]))

    workplace_size_brackets = sp.get_workplace_size_brackets(datadir,country_location)
    workplace_size_count = sp.get_workplace_sizes(datadir,country_location)

    workplace_sizes = sp.generate_workplace_sizes(workplace_size_count,workplace_size_brackets,workers_by_age_to_assign_count)
    print('work')
    # print(workplace_sizes)
    # print(workplace_size_count)
    # print(workplace_size_brackets)

    gen_workplaces,gen_workplace_uids,potential_worker_uids,potential_worker_uids_by_age,workers_by_age_to_assign_count = sp.assign_rest_of_workers(workplace_sizes,potential_worker_uids,potential_worker_uids_by_age,workers_by_age_to_assign_count,age_brackets,age_by_brackets_dic,contact_matrix_dic)
    # print(age_by_brackets_dic[75])
    # print(workers_by_age_to_assign_count)

    # print(np.sum([workers_by_age_to_assign_count[a] for a in workers_by_age_to_assign_count]))
    # for a in workers_by_age_to_assign_count:
        # print(a,workers_by_age_to_assign_count[a])

# create_work_and_school = True
create_work_and_school = False
if create_work_and_school:
    sp.write_schools_by_age_and_uid(datadir,location,state_location,country_location,Nhomes,gen_school_uids,age_by_uid_dic)
    sp.write_workplaces_by_age_and_uid(datadir,location,state_location,country_location,Nhomes,gen_workplace_uids,age_by_uid_dic)

# read_pop = True
read_pop = False
if read_pop:
    popdict = sp.make_contacts_from_microstructure(datadir,location,state_location,country_location,Nhomes)

    uids = popdict.keys()
    uids = [uid for uid in uids]
# for i in range(20):
    # print(popdict[uids[i]])

scdf = sp.get_school_sizes_df(datadir,location,state_location)
# print(sorted(scdf.TotalEnrollment.values))
school_sizes = sp.get_school_sizes_by_bracket(datadir,location,state_location)
# print(school_sizes)