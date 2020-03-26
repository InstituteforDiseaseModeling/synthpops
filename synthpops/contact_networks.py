import sciris as sc
import numpy as np
import networkx as nx
import pandas as pd
from collections import Counter
import os
from . import synthpops as sp
from .config import datadir

from copy import deepcopy



def generate_household_sizes(Nhomes,hh_size_distr):
    max_size = max(hh_size_distr.keys())
    hh_sizes = np.random.multinomial(Nhomes,[hh_size_distr[s] for s in range(1,max_size+1)], size = 1)[0]
    return hh_sizes


def trim_households(N_extra,hh_size_distr):
    ss = np.sum([hh_size_distr[s] * s for s in hh_size_distr])
    print(ss)
    f = N_extra / np.round(ss,16)
    print(f,'f')
    hh_sizes_trim = np.zeros(len(hh_size_distr))
    for s in hh_size_distr:
        print(s,hh_size_distr[s], hh_size_distr[s] * s)
        hh_sizes_trim[s-1] = int(hh_size_distr[s] * f)
    print(np.sum([hh_sizes_trim[s-1] * s for s in hh_size_distr]))

    N_gen = np.sum([hh_sizes_trim[s-1] * s for s in hh_size_distr])
    s_range = np.arange(1,max(hh_size_distr) + 1)
    p = [hh_size_distr[s] for s in hh_size_distr]

    while (N_gen < N_extra):
        ns = np.random.choice(s_range,p = p)
        print(ns,N_gen,N_extra)
        N_gen += ns

        hh_sizes_trim[ns-1] +=1

    print(hh_sizes_trim)

    last_house_size = int(N_gen - N_extra)

    print(N_gen,'N_gen',last_house_size)
    print(hh_sizes_trim)
    if last_house_size > 0:
        hh_sizes_trim[last_house_size-1] -= 1
    elif last_house_size < 0:
        hh_sizes_trim[-last_house_size-1] +=1
    else:
        pass
    print(hh_sizes_trim)

    return hh_sizes_trim



def generate_household_sizes_from_fixed_pop_size(N,hh_size_distr):
    ss = np.sum([hh_size_distr[s] * s for s in hh_size_distr])
    print(ss)
    f = N / np.round(ss,1)
    hh_sizes = np.zeros(len(hh_size_distr))

    for s in hh_size_distr:
        print(s,hh_size_distr[s], hh_size_distr[s] * s)
        hh_sizes[s-1] = int(hh_size_distr[s] * f)
    # print(hh_sizes)
    print(np.sum([hh_sizes[s-1] * s for s in hh_size_distr]))
    N_gen = np.sum([hh_sizes[s-1] * s for s in hh_size_distr], dtype= int)

    trim_hh = trim_households(N_gen - N, hh_size_distr)

    new_hh_sizes = hh_sizes - trim_hh
    print('sum',np.sum([new_hh_sizes[s-1] * s for s in hh_size_distr]))

    new_hh_sizes = new_hh_sizes.astype(int)

    return new_hh_sizes

def generate_school_sizes(school_sizes_by_bracket,uids_in_school):
    n = len(uids_in_school)

    size_distr = sp.norm_dic(school_sizes_by_bracket)
    ss = np.sum([size_distr[s] * s for s in size_distr])

    f = n/ss
    sc = {}
    for s in size_distr:
        sc[s] = int(f * size_distr[s])

    school_sizes = []
    for s in sc:
        school_sizes += [int(s)] * sc[s]
    np.random.shuffle(school_sizes)
    return school_sizes




def get_totalpopsize_from_household_sizes(hh_sizes):
    return np.sum([hh_sizes[s] * (s+1) for s in range(len(hh_sizes))])


def generate_household_head_age_by_size(hha_by_size_counts,hha_brackets,hh_size,single_year_age_distr):

    distr = hha_by_size_counts[hh_size-1,:]
    b = sp.sample_single(distr)
    hha = sp.sample_from_range(single_year_age_distr,hha_brackets[b][0],hha_brackets[b][-1])

    return hha


def generate_living_alone(hh_sizes,hha_by_size_counts,hha_brackets,single_year_age_distr):
    size = 1
    print(hh_sizes)
    homes = np.zeros((hh_sizes[size-1],1))

    for h in range(hh_sizes[size-1]):
        hha = generate_household_head_age_by_size(hha_by_size_counts,hha_brackets,size,single_year_age_distr)
        homes[h][0] = hha

    return homes


def generate_larger_households(size,hh_sizes,hha_by_size_counts,hha_brackets,age_brackets,age_by_brackets_dic,contact_matrix_dic,single_year_age_distr):

    p_coin = dict.fromkeys(np.arange(2,len(hh_sizes)+1), 0.4)
    ya_coin = 0.4

    homes = np.zeros((hh_sizes[size-1],size))

    for h in range(hh_sizes[size-1]):

        hha = generate_household_head_age_by_size(hha_by_size_counts,hha_brackets,size,single_year_age_distr)
        # if hha >= 65 and hha <= 85:
            # if np.random.binomial(1,p_coin[size]):

                # while (hha >= 65 and hha <=85):
                    # hha = sp.sample_from_range(single_year_age_distr,30,50)
                    # hha = generate_household_head_age_by_size(hha_by_size_counts,hha_brackets,size,single_year_age_distr)
                    # print(hha)
        homes[h][0] = hha

        b = age_by_brackets_dic[hha]
        b_prob = contact_matrix_dic['H'][b,:]

        for n in range(1,size):
            bi = sp.sample_single(b_prob)
            ai = sp.sample_from_range(single_year_age_distr,age_brackets[bi][0],age_brackets[bi][-1])
            if ai == 100:
                ai -= 1

            if ai >= 40:
                if np.random.binomial(1,p_coin[size]):
                    while (ai > 40):
                        bi = sp.sample_single(b_prob)
                        ai = sp.sample_from_range(single_year_age_distr,age_brackets[bi][0],age_brackets[bi][-1])

            elif ai >= 18 and ai <= 22:
                if np.random.binomial(1,ya_coin):
                    while (ai >= 18):
                        bi = sp.sample_single(b_prob)
                        ai = sp.sample_from_range(single_year_age_distr,age_brackets[bi][0],age_brackets[bi][-1])

            # ai = sp.resample_age(single_year_age_distr,ai)
            homes[h][n] = ai
        # print(size, homes[h,:])

    return homes


def generate_all_households(hh_sizes,hha_by_size_counts,hha_brackets,age_brackets,age_by_brackets_dic,contact_matrix_dic,single_year_age_distr):

    homes_dic = {}
    homes_dic[1] = generate_living_alone(hh_sizes,hha_by_size_counts,hha_brackets,single_year_age_distr)

    for s in range(2,8):
        homes_dic[s] = generate_larger_households(s,hh_sizes,hha_by_size_counts,hha_brackets,age_brackets,age_by_brackets_dic,contact_matrix_dic,single_year_age_distr)
    
    print('inside', np.sum([hh_sizes[s-1] * s for s in np.arange(1,8)]))
    print('n homes', np.sum(hh_sizes))

    homes = []
    for s in homes_dic:
        homes += list(homes_dic[s])

    print(homes[0])
    print(len(homes))
    nhomes = len(homes)
    print(np.sum([ len(homes[n]) for n in range(nhomes)]))
    np.random.shuffle(homes)
    return homes_dic,homes


def assign_uids_by_homes(homes,id_len=16):

    setting_codes = ['H','S','W','R']

    age_by_uid_dic = {}
    homes_by_uids = []

    for h,home in enumerate(homes):

        home_ids = []
        for a in home:
            uid = sc.uuid(length=id_len)
            age_by_uid_dic[uid] = a
            home_ids.append(uid)
            # print(len(age_by_uid_dic), len(home_ids),len(home))
        # if h % 50 == 0:
            # print(h,home_ids, [age_by_uid_dic[ uid ] for uid in home_ids])
        homes_by_uids.append(home_ids)


    return homes_by_uids,age_by_uid_dic


def write_homes_by_age_and_uid(datadir,location,state_location,country_location,homes_by_uids,age_by_uid_dic):
    file_path = os.path.join(datadir,'demographics',country_location,state_location)
    os.makedirs(file_path,exist_ok=True)

    # households_by_age_path = os.path.join(file_path,location + '_' + str(len(homes_by_uids)) + '_synthetic_households_with_ages.dat')
    # households_by_uid_path = os.path.join(file_path,location + '_' + str(len(homes_by_uids)) + '_synthetic_households_with_uids.dat')
    # age_by_uid_path = os.path.join(file_path,location + '_' + str(len(homes_by_uids)) + '_age_by_uid.dat')


    households_by_age_path = os.path.join(file_path,location + '_' + str(len(age_by_uid_dic)) + '_synthetic_households_with_ages.dat')
    households_by_uid_path = os.path.join(file_path,location + '_' + str(len(age_by_uid_dic)) + '_synthetic_households_with_uids.dat')
    age_by_uid_path = os.path.join(file_path,location + '_' + str(len(age_by_uid_dic)) + '_age_by_uid.dat')

    fh_age = open(households_by_age_path,'w')
    fh_uid = open(households_by_uid_path,'w')
    f_age_uid = open(age_by_uid_path,'w')

    for n,ids in enumerate(homes_by_uids):

        home = homes_by_uids[n]

        for uid in home:

            fh_age.write( str(age_by_uid_dic[uid]) + ' ' )
            fh_uid.write( uid + ' ')
            f_age_uid.write( uid + ' ' + str(age_by_uid_dic[uid]) + '\n')
        fh_age.write('\n')
        fh_uid.write('\n')
    fh_age.close()
    fh_uid.close()
    f_age_uid.close()


# def read_in_age_by_uid(datadir,location,state_location,country_location,Nhomes):
def read_in_age_by_uid(datadir,location,state_location,country_location,N):

    file_path = os.path.join(datadir,'demographics',country_location,state_location)
    age_by_uid_path = os.path.join(file_path,location + '_' + str(N) + '_age_by_uid.dat')

    # age_by_uid_path = os.path.join(file_path,location + '_' + str(Nhomes) + '_age_by_uid.dat')
    df = pd.read_csv(age_by_uid_path,header = None, delimiter = ' ')
    return dict( zip(df.iloc[:,0].values, df.iloc[:,1].values) )


def get_school_enrollment_rates_df(datadir,location,level):
    if 'synthpops' in datadir:
        file_path = datadir.replace('synthpops','COVID-19 (1)')
    file_path = os.path.join(file_path,'seattle_network','schools',level + '_school_enrollment_by_age','ACSST5Y2018.S1401_data_with_overlays_2020-03-06T233142.csv')
    df = pd.read_csv(file_path)
    if location == 'seattle_metro':
        d = df[df['NAME'].isin(['King County, Washington','Geographic Area Name'])]
    return d


def get_school_enrollment_rates(datadir,location,level):
    df = get_school_enrollment_rates_df(datadir,location,level)
    skip_labels = ['Error','public','private','(X)']
    include_labels = ['Percent']

    labels = df.iloc[0,:]

    age_brackets = {}
    rates = {}
    rates_by_age = dict.fromkeys(np.arange(100),0)

    for n,label in enumerate(labels):

        if 'Percent' in label and 'enrolled in school' in label and 'Error' not in label and 'public' not in label and 'private' not in label and 'X' not in label and 'year olds' in label:

            age_bracket = label.replace(' year olds enrolled in school','')
            age_bracket = age_bracket.split('!!')
            age_bracket = age_bracket[-1]
            if 'to' in age_bracket:
                age_bracket = age_bracket.split(' to ')
            elif ' and ' in age_bracket:
                age_bracket = age_bracket = age_bracket.split(' and ')
            sa = int(age_bracket[0])
            ea = int(age_bracket[1])

            rates[len(age_brackets)] = float(df.iloc[1,n])
            age_brackets[len(age_brackets)] = np.arange(sa,ea+1)
            for a in np.arange(sa,ea+1):
                rates_by_age[a] = float(df.iloc[1,n])/100.

    return rates_by_age


def get_school_sizes_df(datadir,location):
    if 'synthpops' in datadir:
        file_path = datadir.replace('synthpops','COVID-19 (1)')
    file_path = os.path.join(file_path,'seattle_network','schools','Total_Enrollment_fiftypercentschools2017.dat')
    df = pd.read_csv(file_path)
    return df


def get_school_size_brackets(datadir,location):
    if 'synthpops' in datadir:
        file_path = datadir.replace('synthpops','COVID-19 (1)')
    file_path = os.path.join(file_path,'seattle_network','schools','school_size_brackets.dat')
    return sp.get_age_brackets_from_df(file_path)


def get_school_sizes_by_bracket(datadir,location):
    df = get_school_sizes_df(datadir,location)
    sizes = df.iloc[:,0].values
    size_count = Counter(sizes)

    size_brackets = get_school_size_brackets(datadir,location)
    size_by_bracket_dic = sp.get_age_by_brackets_dic(size_brackets)

    bracket_count = dict.fromkeys(np.arange(len(size_brackets)), 0)

    for s in size_count:
        bracket_count[ size_by_bracket_dic[s] ] += size_count[s]

    count_by_mean = {}

    for b in bracket_count:
        size = int(np.mean(size_brackets[b]))
        count_by_mean[size] = bracket_count[b]

    return count_by_mean


def generate_school_sizes(school_sizes_by_bracket,uids_in_school):
    n = len(uids_in_school)

    size_distr = sp.norm_dic(school_sizes_by_bracket)
    ss = np.sum([size_distr[s] * s for s in size_distr])

    f = n/ss
    sc = {}
    for s in size_distr:
        sc[s] = int(f * size_distr[s])

    school_sizes = []
    for s in sc:
        school_sizes += [int(s)] * sc[s]
    np.random.shuffle(school_sizes)
    return school_sizes


def get_uids_in_school(datadir,location,state_location,country_location,level,Nhomes):

    uids_in_school = {}
    uids_in_school_by_age = {}
    ages_in_school_count = dict.fromkeys(np.arange(100),0)

    rates = get_school_enrollment_rates(datadir,location,level)

    for a in np.arange(100):
        uids_in_school_by_age[a] = []

    age_by_uid_dic = read_in_age_by_uid(datadir,location,state_location,country_location,Nhomes)

    for uid in age_by_uid_dic:

        a = age_by_uid_dic[uid]
        if a < 35:
            b = np.random.binomial(1,rates[a])
            if b:
                uids_in_school[uid] = a
                uids_in_school_by_age[a].append(uid)

                ages_in_school_count[a] += 1

    return uids_in_school,uids_in_school_by_age,ages_in_school_count


def send_students_to_school(school_sizes,uids_in_school,uids_in_school_by_age,ages_in_school_count,age_brackets,age_by_brackets_dic,contact_matrix_dic):
    syn_schools = []
    syn_school_uids = []
    age_range = np.arange(100)

    ages_in_school_distr = sp.norm_dic(ages_in_school_count)

    for n,size in enumerate(school_sizes):

        ages_in_school_distr = sp.norm_dic(ages_in_school_count)

        new_school = []
        new_school_uids = []

        achoice = np.random.multinomial(1, [ages_in_school_distr[a] for a in ages_in_school_distr])
        aindex = np.where(achoice)[0][0]

        uid = uids_in_school_by_age[aindex][0]
        uids_in_school_by_age[aindex].remove(uid)
        ages_in_school_count[aindex] -= 1
        ages_in_school_distr = sp.norm_dic(ages_in_school_count)

        new_school.append(aindex)
        new_school_uids.append(uid)

        bindex = age_by_brackets_dic[aindex]
        b_prob = contact_matrix_dic['S'][bindex,:]

        for i in range(1,size):
            bi = sp.sample_single(b_prob)
            # while bi >= 7:
                # bi = sp.sample_single(b_prob)
            a_in_school_prob_sum = np.sum([ages_in_school_distr[a] for a in age_brackets[bi] if a < 100])

            while a_in_school_prob_sum == 0 or bi >= 7:
                bi = sp.sample_single(b_prob)
                a_in_school_prob_sum = np.sum([ages_in_school_distr[a] for a in age_brackets[bi] if a < 100])

            ai = sp.sample_from_range(ages_in_school_distr,age_brackets[bi][0], age_brackets[bi][-1])
            uid = uids_in_school_by_age[ai][0]

            new_school.append(ai)
            new_school_uids.append(uid)

            uids_in_school_by_age[ai].remove(uid)
            ages_in_school_count[ai] -= 1
            ages_in_school_distr = sp.norm_dic(ages_in_school_count)

        syn_schools.append(new_school)
        syn_school_uids.append(new_school_uids)

    print('left',len(uids_in_school_by_age))
    return syn_schools,syn_school_uids


def get_uids_potential_workers(uids_in_school,uids_in_school_by_age,age_by_uid_dic):
    potential_worker_uids = deepcopy(age_by_uid_dic)
    potential_worker_uids_by_age = {}
    potential_worker_ages_left_count = dict.fromkeys(np.arange(100),0)

    for a in range(100):
        if a >= 16:
            potential_worker_uids_by_age[a] = []

    for uid in uids_in_school:
        potential_worker_uids.pop(uid,None)

    for uid in potential_worker_uids:
        ai = potential_worker_uids[uid]
        if ai >= 16:
            potential_worker_uids_by_age[ai].append(uid)

            potential_worker_ages_left_count[ai] += 1

    # shuffle workers around!
    for ai in potential_worker_uids_by_age:
        np.random.shuffle(potential_worker_uids_by_age[ai])

    return potential_worker_uids,potential_worker_uids_by_age,potential_worker_ages_left_count


def get_employment_rates(datadir,location,state_location,country_location):
    if 'synthpops' in datadir:
        file_path = datadir.replace('synthpops','COVID-19 (1)')
    file_path = os.path.join(file_path,'demographics',country_location,state_location,'census','employment',location + '_employment_pct_by_age.csv')
    df = pd.read_csv(file_path)
    dic = dict(zip(df.Age,df.Percent))
    for a in range(75,100):
        dic[a] = 0.
    return dic


def get_workplace_size_brackets(datadir,country_location):
    if 'synthpops' in datadir:
        file_path = datadir.replace('synthpops','COVID-19 (1)')
    file_path = os.path.join(file_path,'demographics',country_location,'usa_work_size_brackets.dat')
    return sp.get_age_brackets_from_df(file_path)


def get_workplace_sizes(datadir,country_location):
    if 'synthpops' in datadir:
        file_path = datadir.replace('synthpops','COVID-19 (1)')
    file_path = os.path.join(file_path,'demographics',country_location,'usa_work_size_count.dat')
    df = pd.read_csv(file_path)
    return dict(zip(df.work_size_bracket,df.size_count))


def generate_workplace_sizes(workplace_sizes_by_bracket,workplace_size_brackets,workers_by_age_to_assign_count):
    n = np.sum([ workers_by_age_to_assign_count[a] for a in workers_by_age_to_assign_count ])

    size_distr = {}
    for b in workplace_size_brackets:
        size = int(np.mean(workplace_size_brackets[b]) + 0.5)
        size_distr[size] = workplace_sizes_by_bracket[b]

    size_distr = sp.norm_dic(size_distr)
    ss = np.sum([size_distr[s] * s for s in size_distr])

    f = n/ss
    sc = {}
    for s in size_distr:
        sc[s] = int(f * size_distr[s])

    workplace_sizes = []
    for s in sc:
        workplace_sizes += [int(s)] * sc[s]
    np.random.shuffle(workplace_sizes)
    return workplace_sizes


def get_workers_by_age_to_assign(employment_rates,potential_worker_ages_left_count,uids_by_age_dic):

    workers_by_age_to_assign_count = dict.fromkeys(np.arange(100),0)
    for a in potential_worker_ages_left_count:
        if a in employment_rates:
            c = int(employment_rates[a] * len(uids_by_age_dic[a]))
            workers_by_age_to_assign_count[a] = c

    return workers_by_age_to_assign_count


def assign_teachers_to_work(syn_schools,syn_school_uids,employment_rates,workers_by_age_to_assign_count,potential_worker_uids,potential_worker_uids_by_age,potential_worker_ages_left_count,student_teacher_ratio=80):

    teacher_age_min = 26
    teacher_age_max = 75

    for n in range(len(syn_schools)):
        school = syn_schools[n]
        school_uids = syn_school_uids[n]

        size = len(school)
        nteachers = int(size/float(student_teacher_ratio))
        teachers = []
        teacher_uids = []

        for nt in range(nteachers):

            a = sp.sample_from_range(workers_by_age_to_assign_count,teacher_age_min,teacher_age_max)
            uid = potential_worker_uids_by_age[a][0]
            teachers.append(a)

            potential_worker_uids_by_age[a].remove(uid)
            workers_by_age_to_assign_count[a] -= 1
            potential_worker_ages_left_count[a] -= 1
            potential_worker_uids.pop(uid,None)

            school.append(a)
            school_uids.append(uid)

        syn_schools[n] = school
        syn_school_uids[n] = school_uids

    return syn_schools,syn_school_uids,potential_worker_uids,potential_worker_uids_by_age,workers_by_age_to_assign_count


def assign_rest_of_workers(workplace_sizes,potential_worker_uids,potential_worker_uids_by_age,workers_by_age_to_assign_count,age_brackets,age_by_brackets_dic,contact_matrix_dic):

    syn_workplaces = []
    syn_workplace_uids = []
    age_range = np.arange(100)

    for n,size in enumerate(workplace_sizes):

        workers_by_age_to_assign_distr = sp.norm_dic(workers_by_age_to_assign_count)

        new_work = []
        new_work_uids = []

        achoice = np.random.multinomial(1, [workers_by_age_to_assign_distr[a] for a in age_range])
        aindex = np.where(achoice)[0][0]

        while len(potential_worker_uids_by_age[aindex]) == 0:
            achoice = np.random.multinomial(1, [workers_by_age_to_assign_distr[a] for a in age_range])
            aindex = np.where(achoice)[0][0]

        uid = potential_worker_uids_by_age[aindex][0]
        potential_worker_uids_by_age[aindex].remove(uid)
        workers_by_age_to_assign_count[aindex] -= 1

        potential_worker_uids.pop(uid,None)

        workers_by_age_to_assign_distr = sp.norm_dic(workers_by_age_to_assign_count)

        new_work.append(aindex)
        new_work_uids.append(uid)

        bindex = age_by_brackets_dic[aindex]
        b_prob = contact_matrix_dic['W'][bindex,:]

        for i in range(1,size):
            bi = sp.sample_single(b_prob)
            a_in_work_prob = np.sum([workers_by_age_to_assign_distr[a] for a in age_brackets])

            while bi >= 15:
                bi = sp.sample_single(b_prob)

            while a_in_work_prob == 0:
                bi = sp.sample_single(b_prob)
                a_in_work_prob = np.sum([workers_by_age_to_assign_distr[a] for a in age_brackets])

            ai = sp.sample_from_range(workers_by_age_to_assign_distr,age_brackets[bi][0],age_brackets[bi][-1])
            # while len(potential_worker_uids_by_age[ai]) == 0:
                # ai = sp.sample_from_range(workers_by_age_to_assign_distr,age_brackets[bi][0],age_brackets[bi][-1])
            if len(potential_worker_uids_by_age[ai]) > 0:
                # continue

                uid = potential_worker_uids_by_age[ai][0]

                new_work.append(ai)
                new_work_uids.append(uid)

                potential_worker_uids_by_age[ai].remove(uid)
                potential_worker_uids.pop(uid,None)
                workers_by_age_to_assign_count[ai] -= 1
                workers_by_age_to_assign_distr = sp.norm_dic(workers_by_age_to_assign_count)

        syn_workplaces.append(new_work)
        syn_workplace_uids.append(new_work_uids)

    return syn_workplaces,syn_workplace_uids,potential_worker_uids,potential_worker_uids_by_age,workers_by_age_to_assign_count


def write_schools_by_age_and_uid(datadir,location,state_location,country_location,Nhomes,schools_by_uids,age_by_uid_dic):

    file_path = os.path.join(datadir,'demographics',country_location,state_location)
    os.makedirs(file_path,exist_ok=True)
    schools_by_age_path = os.path.join(file_path,location + '_' + str(Nhomes) + '_synthetic_schools_with_ages.dat')
    schools_by_uid_path = os.path.join(file_path,location + '_' + str(Nhomes) + '_synthetic_schools_with_uids.dat')

    fh_age = open(schools_by_age_path,'w')
    fh_uid = open(schools_by_uid_path,'w')

    for n,ids in enumerate(schools_by_uids):

        school = schools_by_uids[n]

        for uid in schools_by_uids[n]:

            fh_age.write( str(age_by_uid_dic[uid]) + ' ' )
            fh_uid.write( uid + ' ')
        fh_age.write('\n')
        fh_uid.write('\n')
    fh_age.close()
    fh_uid.close()


def write_workplaces_by_age_and_uid(datadir,location,state_location,country_location,Nhomes,workplaces_by_uids,age_by_uid_dic):
    print(datadir)
    file_path = os.path.join(datadir,'demographics',country_location,state_location)
    os.makedirs(file_path,exist_ok=True)
    workplaces_by_age_path = os.path.join(file_path,location + '_' + str(Nhomes) + '_synthetic_workplaces_with_ages.dat')
    workplaces_by_uid_path = os.path.join(file_path,location + '_' + str(Nhomes) + '_synthetic_workplaces_with_uids.dat')

    fh_age = open(workplaces_by_age_path,'w')
    fh_uid = open(workplaces_by_uid_path,'w')

    for n,ids in enumerate(workplaces_by_uids):

        work = workplaces_by_uids[n]

        for uid in work:

            fh_age.write( str(age_by_uid_dic[uid]) + ' ' )
            fh_uid.write( uid + ' ')
        fh_age.write('\n')
        fh_uid.write('\n')
    fh_age.close()
    fh_uid.close()


def make_contacts_from_microstructure(datadir,location,state_location,country_location,Nhomes):
    file_path = os.path.join(datadir,'demographics',country_location,state_location)

    households_by_uid_path = os.path.join(file_path,location + '_' + str(Nhomes) + '_synthetic_households_with_uids.dat')
    age_by_uid_path = os.path.join(file_path,location + '_' + str(Nhomes) + '_age_by_uid.dat')

    workplaces_by_uid_path = os.path.join(file_path,location + '_' + str(Nhomes) + '_synthetic_workplaces_with_uids.dat')
    schools_by_uid_path = os.path.join(file_path,location + '_' + str(Nhomes) + '_synthetic_schools_with_uids.dat')

    df = pd.read_csv(age_by_uid_path, delimiter = ' ',header = None)

    age_by_uid_dic = dict(zip( df.iloc[:,0], df.iloc[:,1]))
    uids = age_by_uid_dic.keys()

    # you have both ages and sexes so we'll just populate that for you...
    popdict = {}
    for i,uid in enumerate(uids):
        popdict[uid] = {}
        popdict[uid]['age'] = age_by_uid_dic[uid]
        popdict[uid]['sex'] = np.random.binomial(1,p=0.5)
        popdict[uid]['loc'] = None
        popdict[uid]['contacts'] = {}
        for k in ['H','S','W','R']:
            popdict[uid]['contacts'][k] = set()

    fh = open(households_by_uid_path,'r')
    for c,line in enumerate(fh):
        r = line.strip().split(' ')
        for uid in r:

            for juid in r:
                if uid != juid:
                    popdict[uid]['contacts']['H'].add(juid)
    fh.close()

    fs = open(schools_by_uid_path,'r')
    for c,line in enumerate(fs):
        r = line.strip().split(' ')
        for uid in r:
            for juid in r:
                if uid != juid:
                    popdict[uid]['contacts']['S'].add(juid)
    fs.close()

    fw = open(workplaces_by_uid_path,'r')
    for c,line in enumerate(fw):
        r = line.strip().split(' ')
        for uid in r:
            for juid in r:
                if uid != juid:
                    popdict[uid]['contacts']['W'].add(juid)
    fw.close()

    return popdict


def generate_synthetic_population(n,datadir,location='seattle_metro',state_location='Washington',country_location='usa',use_bayesian=False):

    age_brackets = sp.get_census_age_brackets(datadir,country_location,use_bayesian)
    age_by_brackets_dic = sp.get_age_by_brackets_dic(age_brackets)

    if use_bayesian: num_agebrackets = 16
    else: num_agebrackets = 18

    contact_matrix_dic = sp.get_contact_matrix_dic(datadir,state_location,num_agebrackets)

    hh_size_distr = sp.get_household_size_distr(datadir,location,state_location,country_location,use_bayesian)

    if n < 3000:
        raise NotImplementedError
    n = int(n)


