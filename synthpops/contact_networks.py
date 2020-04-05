import sciris as sc
import numpy as np
import networkx as nx
import pandas as pd
from collections import Counter
import os
from . import synthpops as sp
from .config import datadir

from copy import deepcopy
import matplotlib as mplt
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import cmocean

# Set user-specific configurations
username = os.path.split(os.path.expanduser('~'))[-1]
fontdirdict = {
    'dmistry': '/home/dmistry/Dropbox (IDM)/GoogleFonts',
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
    pass

def generate_household_sizes(Nhomes,hh_size_distr):
    max_size = max(hh_size_distr.keys())
    hh_sizes = np.random.multinomial(Nhomes,[hh_size_distr[s] for s in range(1,max_size+1)], size = 1)[0]
    return hh_sizes


def trim_households(N_extra,hh_size_distr):
    ss = np.sum([hh_size_distr[s] * s for s in hh_size_distr])
    f = N_extra / np.round(ss,16)
    hh_sizes_trim = np.zeros(len(hh_size_distr))
    for s in hh_size_distr:
        hh_sizes_trim[s-1] = int(hh_size_distr[s] * f)

    N_gen = np.sum([hh_sizes_trim[s-1] * s for s in hh_size_distr])
    s_range = np.arange(1,max(hh_size_distr) + 1)
    p = [hh_size_distr[s] for s in hh_size_distr]

    while (N_gen < N_extra):
        ns = np.random.choice(s_range,p = p)
        N_gen += ns

        hh_sizes_trim[ns-1] +=1


    last_house_size = int(N_gen - N_extra)

    if last_house_size > 0:
        hh_sizes_trim[last_house_size-1] -= 1
    elif last_house_size < 0:
        hh_sizes_trim[-last_house_size-1] +=1
    else:
        pass
    # print('trim',hh_sizes_trim,[hh_sizes_trim[s] * (s+1) for s in range(len(hh_sizes_trim))], np.sum([hh_sizes_trim[s] * (s+1) for s in range(len(hh_sizes_trim))]))

    return hh_sizes_trim


def generate_household_sizes_from_fixed_pop_size(N,hh_size_distr):
    ss = np.sum([hh_size_distr[s] * s for s in hh_size_distr])
    f = N / np.round(ss,1)
    hh_sizes = np.zeros(len(hh_size_distr))

    for s in hh_size_distr:
        hh_sizes[s-1] = int(hh_size_distr[s] * f)
    N_gen = np.sum([hh_sizes[s-1] * s for s in hh_size_distr], dtype= int)

    trim_hh = trim_households(N_gen - N, hh_size_distr)
    new_hh_sizes = hh_sizes - trim_hh
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

    ya_coin = 0.15 # produce far too few young adults without this

    homes = np.zeros((hh_sizes[size-1],size))

    for h in range(hh_sizes[size-1]):

        hha = generate_household_head_age_by_size(hha_by_size_counts,hha_brackets,size,single_year_age_distr)
    
        homes[h][0] = hha

        b = age_by_brackets_dic[hha]
        b_prob = contact_matrix_dic['H'][b,:]

        for n in range(1,size):
            bi = sp.sample_single(b_prob)
            ai = sp.sample_from_range(single_year_age_distr,age_brackets[bi][0],age_brackets[bi][-1])

            if ai > 5 and ai <= 20:
                if np.random.binomial(1,ya_coin):
                    ai = sp.sample_from_range(single_year_age_distr,25,30)

            ai = sp.resample_age(single_year_age_distr,ai)

            homes[h][n] = ai

    return homes


def generate_all_households(N,hh_sizes,hha_by_size_counts,hha_brackets,age_brackets,age_by_brackets_dic,contact_matrix_dic,single_year_age_distr):

    homes_dic = {}
    homes_dic[1] = generate_living_alone(hh_sizes,hha_by_size_counts,hha_brackets,single_year_age_distr)
    # remove living alone from the distribution to choose from!
    for h in homes_dic[1]:
        single_year_age_distr[h[0]] -= 1.0/N

    for s in range(2,8):
        homes_dic[s] = generate_larger_households(s,hh_sizes,hha_by_size_counts,hha_brackets,age_brackets,age_by_brackets_dic,contact_matrix_dic,single_year_age_distr)

    homes = []
    for s in homes_dic:
        homes += list(homes_dic[s])

    nhomes = len(homes)
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

        homes_by_uids.append(home_ids)

    return homes_by_uids,age_by_uid_dic


def write_homes_by_age_and_uid(datadir,location,state_location,country_location,homes_by_uids,age_by_uid_dic):
    
    file_path = os.path.join(datadir,'demographics','contact_matrices_152_countries',state_location)
    os.makedirs(file_path,exist_ok=True)

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


def read_in_age_by_uid(datadir,location,state_location,country_location,N):

    file_path = os.path.join(datadir,'demographics','contact_matrices_152_countries',state_location)
    age_by_uid_path = os.path.join(file_path,location + '_' + str(N) + '_age_by_uid.dat')
    df = pd.read_csv(age_by_uid_path,header = None, delimiter = ' ')
    return dict( zip(df.iloc[:,0].values, df.iloc[:,1].values) )


def get_school_enrollment_rates_df(datadir,location,state_location,level):
    # if 'synthpops' in datadir:
        # file_path = datadir.replace('synthpops','COVID-19 (1)')
    file_path = os.path.join(datadir,'demographics','contact_matrices_152_countries',state_location,location,'schools',level + '_school_enrollment_by_age','ACSST5Y2018.S1401_data_with_overlays_2020-03-06T233142.csv')
    df = pd.read_csv(file_path)
    if location == 'seattle_metro':
        d = df[df['NAME'].isin(['King County, Washington','Geographic Area Name'])]
    return d


def get_school_enrollment_rates(datadir,location,state_location,level):
    df = get_school_enrollment_rates_df(datadir,location,state_location,level)
    skip_labels = ['Error','public','private','(X)']
    include_labels = ['Percent']

    labels = df.iloc[0,:]

    age_brackets = {}
    rates = {}
    rates_by_age = dict.fromkeys(np.arange(101),0)

    for n,label in enumerate(labels):

        # if any(l in label for l in skip_labels):
            # continue

        # if any(l in label for l in include_labels):
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

            rates[len(age_brackets)] = float(df.iloc[1,n])/100.
            age_brackets[len(age_brackets)] = np.arange(sa,ea+1)
            for a in np.arange(sa,ea+1):
                rates_by_age[a] = float(df.iloc[1,n])/100.

        if label == 'Estimate!!Percent!!Population enrolled in college or graduate school!!Population 35 years and over!!35 years and over enrolled in school':

            sa,ea = 35,50 # 50 is a guess because the label is 35 and over
            rates[len(age_brackets)] = float(df.iloc[1,n])/100.
            age_brackets[len(age_brackets)] = np.arange(sa,ea+1)
            for a in np.arange(sa,ea+1):
                rates_by_age[a] = float(df.iloc[1,n])/100.

    return rates_by_age


def get_school_sizes_df(datadir,location,state_location):
    # if 'synthpops' in datadir:
        # file_path = datadir.replace('synthpops','COVID-19 (1)')
    file_path = os.path.join(datadir,'demographics','contact_matrices_152_countries',state_location,location,'schools','Total_Enrollment_fiftypercentschools2017.dat')
    df = pd.read_csv(file_path)
    return df


def get_school_size_brackets(datadir,location,state_location):
    # if 'synthpops' in datadir:
        # file_path = datadir.replace('synthpops','COVID-19 (1)')
    file_path = os.path.join(datadir,'demographics','contact_matrices_152_countries',state_location,location,'schools','school_size_brackets.dat')
    return sp.get_age_brackets_from_df(file_path)


def get_school_sizes_by_bracket(datadir,location,state_location):
    df = get_school_sizes_df(datadir,location,state_location)
    sizes = df.iloc[:,0].values
    size_count = Counter(sizes)

    size_brackets = get_school_size_brackets(datadir,location,state_location)
    size_by_bracket_dic = sp.get_age_by_brackets_dic(size_brackets) # not actually ages just a useful function for mapping ints back to their bracket or grouping key

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
    school_sizes = []

    s_range = sorted(size_distr.keys())
    p = [size_distr[s] for s in s_range]
    
    while n > 0:
        s = np.random.choice(s_range, p = p)
        n -= s
        school_sizes.append(s)

    np.random.shuffle(school_sizes)
    return school_sizes


def get_uids_in_school(datadir,location,state_location,country_location,level,N,age_by_uid_dic=None):

    uids_in_school = {}
    uids_in_school_by_age = {}
    ages_in_school_count = dict.fromkeys(np.arange(101),0)

    rates = get_school_enrollment_rates(datadir,location,state_location,level)

    for a in np.arange(101):
        uids_in_school_by_age[a] = []

    if age_by_uid_dic is None:
        age_by_uid_dic = read_in_age_by_uid(datadir,location,state_location,country_location,N)

    for uid in age_by_uid_dic:

        a = age_by_uid_dic[uid]
        if a <= 50: # US census records for school enrollment
            b = np.random.binomial(1,rates[a]) # ask each person if they'll be a student - probably could be done in a faster, more aggregate way.
            if b:
                uids_in_school[uid] = a
                uids_in_school_by_age[a].append(uid)
                ages_in_school_count[a] += 1

    return uids_in_school,uids_in_school_by_age,ages_in_school_count


def send_students_to_school(school_sizes,uids_in_school,uids_in_school_by_age,ages_in_school_count,age_brackets,age_by_brackets_dic,contact_matrix_dic,verbose=False):
    """
    A method to send 'students' to school together. Using the matrices to construct schools is not a perfect method so some things are more forced than the matrix method alone would create.git 
    """

    syn_schools = []
    syn_school_uids = []
    age_range = np.arange(101)

    ages_in_school_distr = sp.norm_dic(ages_in_school_count)
    total_school_count = len(uids_in_school)
    left_in_bracket = sp.get_aggregate_ages(ages_in_school_count,age_by_brackets_dic)

    for n,size in enumerate(school_sizes):

        if len(uids_in_school) == 0: # no more students left to send to school!
            break

        ages_in_school_distr = sp.norm_dic(ages_in_school_count)

        new_school = []
        new_school_ages_in_school_countuids = []
        new_school_uids = []

        achoice = np.random.multinomial(1, [ages_in_school_distr[a] for a in ages_in_school_distr])
        aindex = np.where(achoice)[0][0]
        bindex = age_by_brackets_dic[aindex]

        # reference students under 20 to prevent older adults from being reference students (otherwise we end up with schools with too many adults and kids mixing because the matrices represent the average of the patterns and not the bimodal mixing of adult students together at school and a small number of teachers at school with their students)
        if bindex >= 4:
            if np.random.binomial(1,p=0.8):
                achoice = np.random.multinomial(1, [ages_in_school_distr[a] for a in ages_in_school_distr])
                aindex = np.where(achoice)[0][0]

        uid = uids_in_school_by_age[aindex][0]
        uids_in_school_by_age[aindex].remove(uid)
        uids_in_school.pop(uid,None)
        ages_in_school_count[aindex] -= 1
        ages_in_school_distr = sp.norm_dic(ages_in_school_count)

        new_school.append(aindex)
        new_school_uids.append(uid)

        if verbose:
            print('reference school age',aindex,'school size',size,'students left',len(uids_in_school),left_in_bracket)

        bindex = age_by_brackets_dic[aindex]
        b_prob = contact_matrix_dic['S'][bindex,:]

        left_in_bracket[bindex] -= 1

        # fewer students than school size so everyone else is in one school
        if len(uids_in_school) < size:
            for uid in uids_in_school:
                ai = uids_in_school[uid]
                new_school.append(int(ai))
                new_school_uids.append(uid)
                uids_in_school_by_age[ai].remove(uid)
                ages_in_school_count[ai] -= 1
                left_in_bracket[age_by_brackets_dic[ai]] -= 1
            uids_in_school = {}
            if verbose:
                print('last school','size from distribution',size,'size generated',len(new_school))

        else:
            bi_min = max(0,bindex-1)
            bi_max = bindex + 1

            for i in range(1,size):
                if len(uids_in_school) == 0:
                    break

                # no one left to send? should only choose other students from the mixing matrices, not teachers so don't create schools with 
                if np.sum([left_in_bracket[bi] for bi in np.arange(bi_min,bi_max+1)]) == 0:
                    break

                # a_in_school_prob_sum = np.sum([ages_in_school_distr[a] for bi in age_brackets for a in age_brackets[bi]])
                # if a_in_school_prob_sum == 0:
                    # break

                bi = sp.sample_single(b_prob)
                a_in_school_b_prob_sum = np.sum([ages_in_school_distr[a] for a in age_brackets[bi]])

                if bi >= 4:
                    if np.random.binomial(1,p=0):
                        bi = sp.sample_single(b_prob)
                        a_in_school_b_prob_sum = np.sum([ages_in_school_distr[a] for a in age_brackets[bi]])

                while left_in_bracket[bi] == 0 or np.abs(bindex - bi) > 1:
                # while left_in_bracket[bi] == 0:
                    
                    bi = sp.sample_single(b_prob)
                    a_in_school_b_prob_sum = np.sum([ages_in_school_distr[a] for a in age_brackets[bi]])

                ai = sp.sample_from_range(ages_in_school_distr,age_brackets[bi][0], age_brackets[bi][-1])
                uid = uids_in_school_by_age[ai][0] # grab the next student in line

                new_school.append(ai)
                new_school_uids.append(uid)

                uids_in_school_by_age[ai].remove(uid)
                uids_in_school.pop(uid,None)

                ages_in_school_count[ai] -= 1
                ages_in_school_distr = sp.norm_dic(ages_in_school_count)
                left_in_bracket[bi] -= 1

        syn_schools.append(new_school)
        syn_school_uids.append(new_school_uids)
        new_school = np.array(new_school)
        kids = new_school <=19
        new_school_age_counter = Counter(new_school)
        if verbose:
            print('new school ages',len(new_school),sorted(new_school),'nkids',kids.sum(),'n20+',len(new_school)-kids.sum(),'kid-adult ratio', kids.sum()/(len(new_school)-kids.sum()) )

    print('people in school', np.sum([ len(school) for school in syn_schools ]),'left to send',len(uids_in_school))
    return syn_schools,syn_school_uids


def get_uids_potential_workers(uids_in_school,uids_in_school_by_age,age_by_uid_dic):
    potential_worker_uids = deepcopy(age_by_uid_dic)
    potential_worker_uids_by_age = {}
    potential_worker_ages_left_count = dict.fromkeys(np.arange(101),0)

    for a in range(101):
        if a >= 16: # US Census employment records start at 16
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

    file_path = os.path.join(datadir,'demographics','contact_matrices_152_countries',country_location,state_location,'employment',location + '_employment_pct_by_age.csv')
    df = pd.read_csv(file_path)
    dic = dict(zip(df.Age,df.Percent))
    # for a in range(75,81):
        # dic[a] = dic[a]/2.
    # Census records give the last group as 75+ but very unlikely over 80 are working so reduce this likelihood
    for a in range(81,100):
        dic[a] = dic[a]/10.
    return dic


def get_workplace_size_brackets(datadir,country_location):

    file_path = os.path.join(datadir,'demographics','contact_matrices_152_countries',country_location,'usa_work_size_brackets.dat')
    return sp.get_age_brackets_from_df(file_path)


def get_workplace_sizes(datadir,country_location):

    file_path = os.path.join(datadir,'demographics','contact_matrices_152_countries',country_location,'usa_work_size_count.dat')
    df = pd.read_csv(file_path)
    return dict(zip(df.work_size_bracket,df.size_count))


def generate_workplace_sizes(workplace_sizes_by_bracket,workplace_size_brackets,workers_by_age_to_assign_count):

    nw = np.sum([ workers_by_age_to_assign_count[a] for a in workers_by_age_to_assign_count ])

    size_distr = {}
    for b in workplace_size_brackets:
        size = int(np.mean(workplace_size_brackets[b]) + 0.5)
        size_distr[size] = workplace_sizes_by_bracket[b]

    size_distr = sp.norm_dic(size_distr)
    ss = np.sum([size_distr[s] * s for s in size_distr])

    f = nw/ss
    sc = {}
    for s in size_distr:
        sc[s] = int(f * size_distr[s])

    workplace_sizes = []
    for s in sc:
        workplace_sizes += [int(s)] * sc[s]
    np.random.shuffle(workplace_sizes)
    return workplace_sizes


def get_workers_by_age_to_assign(employment_rates,potential_worker_ages_left_count,uids_by_age_dic):

    workers_by_age_to_assign_count = dict.fromkeys(np.arange(101),0)
    for a in potential_worker_ages_left_count:
        if a in employment_rates:
            try:
                c = int(employment_rates[a] * len(uids_by_age_dic[a]))
            except:
                c = 0
            workers_by_age_to_assign_count[a] = c

    return workers_by_age_to_assign_count


def assign_teachers_to_work(syn_schools,syn_school_uids,employment_rates,workers_by_age_to_assign_count,potential_worker_uids,potential_worker_uids_by_age,potential_worker_ages_left_count,student_teacher_ratio=30,verbose=False):
    # matrix method will already get some teachers into schools so student_teacher_ratio should be higher
    teacher_age_min = 26 # US teachers need at least undergrad to teach and typically some additional time to gain certification from a teaching program - it should take a few years after college
    teacher_age_max = 75 # use max age from employment records.

    for n in range(len(syn_schools)):
        school = syn_schools[n]
        school_uids = syn_school_uids[n]

        size = len(school)
        nteachers = int(size/float(student_teacher_ratio))
        nteachers = max(1,nteachers)
        if verbose:
            print('nteachers',nteachers,'student-teacher ratio',size/nteachers)
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
        if verbose:
            print('school with teachers',sorted(school))
            print('nkids', (np.array(school)<=19).sum(),'n20+', (np.array(school) > 19).sum() )
            print('kid-adult ratio' ,(np.array(school)<=19).sum()/ (np.array(school) > 19).sum() )

    return syn_schools,syn_school_uids,potential_worker_uids,potential_worker_uids_by_age,workers_by_age_to_assign_count


def assign_rest_of_workers(workplace_sizes,potential_worker_uids,potential_worker_uids_by_age,workers_by_age_to_assign_count,age_brackets,age_by_brackets_dic,contact_matrix_dic):

    syn_workplaces = []
    syn_workplace_uids = []
    age_range = np.arange(101)

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
            
            a_in_work_prob = np.sum([workers_by_age_to_assign_distr[a] for a in age_brackets[bi]])
            while bi <= 3 and bi >= 12: # census records say no one under 16 works so skip this bracket, matrix has low contact rates for ages above 65
                bi = sp.sample_single(b_prob)

            while a_in_work_prob == 0:
                bi = sp.sample_single(b_prob)
                a_in_work_prob = np.sum([workers_by_age_to_assign_distr[a] for a in age_brackets[bi]])

            ai = sp.sample_from_range(workers_by_age_to_assign_distr,age_brackets[bi][0],age_brackets[bi][-1])
    
            # while len(potential_worker_uids_by_age[ai]) == 0:
                # ai = sp.sample_from_range(workers_by_age_to_assign_distr,age_brackets[bi][0],age_brackets[bi][-1])
            if len(potential_worker_uids_by_age[ai]) > 0:

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


def write_schools_by_age_and_uid(datadir,location,state_location,country_location,n,schools_by_uids,age_by_uid_dic):
    
    file_path = os.path.join(datadir,'demographics','contact_matrices_152_countries',state_location)
    os.makedirs(file_path,exist_ok=True)
    schools_by_age_path = os.path.join(file_path,location + '_' + str(n) + '_synthetic_schools_with_ages.dat')
    schools_by_uid_path = os.path.join(file_path,location + '_' + str(n) + '_synthetic_schools_with_uids.dat')

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


def write_workplaces_by_age_and_uid(datadir,location,state_location,country_location,n,workplaces_by_uids,age_by_uid_dic):

    file_path = os.path.join(datadir,'demographics','contact_matrices_152_countries',state_location)
    os.makedirs(file_path,exist_ok=True)
    workplaces_by_age_path = os.path.join(file_path,location + '_' + str(n) + '_synthetic_workplaces_with_ages.dat')
    workplaces_by_uid_path = os.path.join(file_path,location + '_' + str(n) + '_synthetic_workplaces_with_uids.dat')

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


def generate_synthetic_population(n,datadir,location='seattle_metro',state_location='Washington',country_location='usa',sheet_name='United States of America',level='county',verbose=False,plot=False):

    age_brackets = sp.get_census_age_brackets(datadir,state_location,country_location)
    age_by_brackets_dic = sp.get_age_by_brackets_dic(age_brackets)

    num_agebrackets = len(age_brackets)
    contact_matrix_dic = sp.get_contact_matrix_dic(datadir,sheet_name)

    household_size_distr = sp.get_household_size_distr(datadir,location,state_location,country_location)

    if n < 5000:
        raise NotImplementedError("Population is too small to currently be generated properly. Try a size larger than 5000.")
    n = int(n)

    # this could be unnecessary if we get the single year age distribution in a different way.
    n_to_sample_smoothly = int(1e6)
    hh_sizes = generate_household_sizes(n_to_sample_smoothly,household_size_distr)
    totalpop = get_totalpopsize_from_household_sizes(hh_sizes)

    # create a rough single year age distribution to draw from instead of the distribution by age brackets.
    syn_ages,syn_sexes = sp.get_usa_age_sex_n(datadir,location,state_location,country_location,totalpop)
    syn_age_count = Counter(syn_ages)
    syn_age_distr = sp.norm_dic(Counter(syn_ages))

    # actual household sizes
    hh_sizes = generate_household_sizes_from_fixed_pop_size(n,household_size_distr)
    totalpop = get_totalpopsize_from_household_sizes(hh_sizes)

    hha_df = sp.get_household_head_age_by_size_df(datadir,state_location,country_location)
    hha_brackets = sp.get_head_age_brackets(datadir,country_location=country_location)
    hha_by_size = sp.get_head_age_by_size_distr(datadir,country_location=country_location)

    homes_dic,homes = generate_all_households(n,hh_sizes,hha_by_size,hha_brackets,age_brackets,age_by_brackets_dic,contact_matrix_dic,deepcopy(syn_age_distr))
    homes_by_uids, age_by_uid_dic = assign_uids_by_homes(homes)
    new_ages_count = Counter(age_by_uid_dic.values())

    # plot synthetic age distribution as a check
    if plot:

        cmap = mplt.cm.get_cmap(cmocean.cm.deep_r)
        cmap2 = mplt.cm.get_cmap(cmocean.cm.curl_r)
        cmap3 = mplt.cm.get_cmap(cmocean.cm.matter)

        fig = plt.figure(figsize = (7,5))
        ax = fig.add_subplot(111)

        x = np.arange(101)
        y_exp = np.zeros(101)
        y_sim = np.zeros(101)

        for a in range(101):
            expected = int(syn_age_distr[a] * totalpop)
            y_exp[a] = expected
            y_sim[a] = new_ages_count[a]

        ax.plot(x,y_exp,color = cmap(0.2), label = 'Expected')
        ax.plot(x,y_sim,color = cmap3(0.6), label = 'Simulated')
        leg = ax.legend(fontsize = 18)
        leg.draw_frame(False)
        ax.set_xlim(left = 0, right = 100)
        ax.set_xticks(np.arange(0,101,5))

        plt.show()
        fig.savefig('synthetic_age_comparison_' + str(n) + '.pdf',format = 'pdf')

    # save households and uids to file - always need to do this...
    write_homes_by_age_and_uid(datadir,location,state_location,country_location,homes_by_uids,age_by_uid_dic)

    age_by_uid_dic = read_in_age_by_uid(datadir,location,state_location,country_location,n)
    uids_by_age_dic = sp.get_ids_by_age_dic(age_by_uid_dic)


    # Generate school sizes
    school_sizes_count = get_school_sizes_by_bracket(datadir,location,state_location)
    # figure out who's going to go to school as a student
    uids_in_school,uids_in_school_by_age,ages_in_school_count = get_uids_in_school(datadir,location,state_location,country_location,level,n,age_by_uid_dic) # this will call in school enrollment rates

    gen_school_sizes = generate_school_sizes(school_sizes_count,uids_in_school)
    gen_schools,gen_school_uids = send_students_to_school(gen_school_sizes,uids_in_school,uids_in_school_by_age,ages_in_school_count,age_brackets,age_by_brackets_dic,contact_matrix_dic,verbose)

    # Figure out who's going to be working
    employment_rates = get_employment_rates(datadir,location,state_location,country_location)
    potential_worker_uids,potential_worker_uids_by_age,potential_worker_ages_left_count = get_uids_potential_workers(uids_in_school,uids_in_school_by_age,age_by_uid_dic)
    workers_by_age_to_assign_count = get_workers_by_age_to_assign(employment_rates,potential_worker_ages_left_count,uids_by_age_dic)

    # Assign teachers and update school lists
    gen_schools,gen_school_uids,potential_worker_uids,potential_worker_uids_by_age,workers_by_age_to_assign_count = assign_teachers_to_work(gen_schools,gen_school_uids,employment_rates,workers_by_age_to_assign_count,potential_worker_uids,potential_worker_uids_by_age,potential_worker_ages_left_count,verbose=True)


    # Generate non-school workplace sizes needed to send everyone to work
    workplace_size_brackets = get_workplace_size_brackets(datadir,country_location)
    workplace_size_count = get_workplace_sizes(datadir,country_location)
    workplace_sizes = generate_workplace_sizes(workplace_size_count,workplace_size_brackets,workers_by_age_to_assign_count)

    # Assign all workers who are not staff at schools to workplaces
    gen_workplaces,gen_workplace_uids,potential_worker_uids,potential_worker_uids_by_age,workers_by_age_to_assign_count = assign_rest_of_workers(workplace_sizes,potential_worker_uids,potential_worker_uids_by_age,workers_by_age_to_assign_count,age_brackets,age_by_brackets_dic,contact_matrix_dic)


    # save schools and workplace uids to file
    write_schools_by_age_and_uid(datadir,location,state_location,country_location,n,gen_school_uids,age_by_uid_dic)
    write_workplaces_by_age_and_uid(datadir,location,state_location,country_location,n,gen_workplace_uids,age_by_uid_dic)
