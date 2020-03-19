import sciris as sc
import numpy as np
from . import synthpops as sp
from .config import datadir
import os


def make_popdict(n=None, uids=None, ages=None, sexes=None, state_location=None, location=None, use_usa=True, id_len=6):
    """ Create a dictionary of n people with age, sex and loc keys """ #
    n = int(n)

    if n < 2000:
        raise NotImplementedError("Stop! I can't work with fewer than 2000 people currently.")

    # A list of UIDs was supplied as the first argument
    if isinstance(n, list):
        uids = n
        n = len(uids)
    elif uids is not None: # UIDs were supplied, use them
        n = len(uids)

    # Not supplied, generate
    if uids is None:
        uids = []
        for i in range(n):
            uids.append(sc.uuid(length=id_len))

    # Optionally take in either ages or sexes, too
    if ages is None or sexes is None:
        if use_usa:
            if location is None: location, state_location = 'seattle_metro', 'Washington'
            gen_ages,gen_sexes = sp.get_usa_age_sex_n(location,state_location,n_people=n)
        else:
            raise NotImplementedError('Currently, only locations in the US are supported. Next version!')
        random_inds = np.random.permutation(n)
        if ages is None:
            ages = [gen_ages[r] for r in random_inds]
        if sexes is None:
            sexes = [gen_sexes[r] for r in random_inds]

    popdict = {}
    for i,uid in enumerate(uids):
        popdict[uid] = {}
        popdict[uid]['age'] = ages[i]
        popdict[uid]['sex'] = sexes[i]
        popdict[uid]['loc'] = None
        popdict[uid]['contacts'] = {'M': set()}

    return popdict


def make_contacts_without_social_layers_usa(popdict,n_contacts_dic,state_location,location,network_distr_args):

    # using a flat contact matrix
    uids_by_age_dic = sp.get_uids_by_age_dic(popdict)

    dropbox_path = datadir
    num_agebrackets = 18
    country_location = 'usa'

    age_bracket_distr = sp.read_age_bracket_distr(dropbox_path,location,state_location,country_location)
    gender_fraction_by_age = sp.read_gender_fraction_by_age_bracket(dropbox_path,location,state_location,country_location)
    age_brackets_filepath = sp.get_census_age_brackets_path(dropbox_path)
    age_brackets = sp.get_age_brackets_from_df(age_brackets_filepath)
    age_by_brackets_dic = sp.get_age_by_brackets_dic(age_brackets)

    age_mixing_matrix_dic = sp.get_contact_matrix_dic(dropbox_path,state_location,num_agebrackets)
    age_mixing_matrix_dic['M'] = sp.combine_matrices(age_mixing_matrix_dic,n_contacts_dic,num_agebrackets)

    n_contacts = network_distr_args['average_degree']
    directed = network_distr_args['directed']
    network_type = network_distr_args['network_type']

    k = 'M'
    weights_dic = {k:1}

    if directed:
        if network_type == 'poisson_degree':
            for i in popdict:
                nc = sp.pt(n_contacts)
                contact_ages = sp.sample_n_contact_ages_with_matrix(nc,popdict[i]['age'],age_brackets,age_by_brackets_dic,age_mixing_matrix_dic['M'],num_agebrackets)
                popdict[i]['contacts']['M'] = popdict[i]['contacts']['M'].union(sp.get_n_contact_ids_by_age(uids_by_age_dic,contact_ages,age_brackets,age_by_brackets_dic))

    else:
        if network_type == 'poisson_degree':
            n_contacts = n_contacts/2
            for i in popdict:
                nc = sp.pt(n_contacts)
                contact_ages = sp.sample_n_contact_ages_with_matrix(nc,popdict[i]['age'],age_brackets,age_by_brackets_dic,age_mixing_matrix_dic['M'],num_agebrackets)
                popdict[i]['contacts']['M'] = popdict[i]['contacts']['M'].union(sp.get_n_contact_ids_by_age(uids_by_age_dic,contact_ages,age_brackets,age_by_brackets_dic))

                for c in popdict[i]['contacts']['M']:
                    popdict[c]['contacts']['M'].add(i)

    return popdict

def make_contacts_with_social_layers_usa(popdict,n_contacts_dic,state_location,location,activity_args,network_distr_args):

    # use a contact matrix dictionary and n_contacts_dic for the average number of contacts in each layer
    uids_by_age_dic = sp.get_uids_by_age_dic(popdict)

    dropbox_path = datadir
    num_agebrackets = 18
    country_location = 'usa'

    age_bracket_distr = sp.read_age_bracket_distr(dropbox_path,location,state_location,country_location)
    gender_fraction_by_age = sp.read_gender_fraction_by_age_bracket(dropbox_path,location,state_location,country_location)
    age_brackets_filepath = sp.get_census_age_brackets_path(dropbox_path)
    age_brackets = sp.get_age_brackets_from_df(age_brackets_filepath)
    age_by_brackets_dic = sp.get_age_by_brackets_dic(age_brackets)

    age_mixing_matrix_dic = sp.get_contact_matrix_dic(dropbox_path,state_location,num_agebrackets)

    n_contacts = network_distr_args['average_degree']
    directed = network_distr_args['directed']

    # weights_dic is calibrated to empirical survey data but for all people by all ages!
    # to figure out the weights for individual ranges, this is an approach to rescale school and workplace weights 

    # problems: this doesn't capture school enrollment rates or work enrollment rates
    student_n_dic = sc.dcp(n_contacts_dic)
    non_student_n_dic = sc.dcp(n_contacts_dic)

    # this might not be needed because students will choose their teachers, but if directed then this makes teachers point to students as well
    n_students = np.sum([len(uids_by_age_dic[a]) for a in range( activity_args['student_age_min'], activity_args['student_age_max']+1)])
    n_workers = np.sum([len(uids_by_age_dic[a]) for a in range( activity_args['worker_age_min'], activity_args['worker_age_max']+1)])
    n_teachers = n_students/activity_args['student_teacher_ratio']
    teachers_school_weight = n_teachers/n_workers

    student_n_dic['W'] = 0
    non_student_n_dic['S'] = teachers_school_weight # make some teachers
    # 5 categories : 
    # infants & toddlers : H, R
    # school-aged students : H, S, R 
    # college-aged students / workers : H, S, W, R
    # non-student workers : H, W, R
    # retired elderly : H, R - some may be workers too but it's low. directed means they won't have contacts in the workplace, but undirected means they will at least a little.

    # will follow degree distribution well
    for uid in popdict:
        for k in n_contacts_dic:
            popdict[uid]['contacts'][k] = set()

    if directed:

        for uid in popdict:
            age = popdict[uid]['age']
            if age < activity_args['student_age_min']:
                for k in ['H','R']:
                    if network_distr_args['network_type'] == 'poisson_degree':
                        nc = sp.pt(n_contacts_dic[k])
                        contact_ages = sp.sample_n_contact_ages_with_matrix(nc,age,age_brackets,age_by_brackets_dic,age_mixing_matrix_dic[k])
                        popdict[uid]['contacts'][k] = popdict[uid]['contacts'][k].union(sp.get_n_contact_ids_by_age(uids_by_age_dic,contact_ages,age_brackets,age_by_brackets_dic))

            elif age >= activity_args['student_age_min'] and age < activity_args['student_age_max']:
                for k in ['H','S','R']:
                    if network_distr_args['network_type'] == 'poisson_degree':
                        nc = sp.pt(n_contacts_dic[k])
                        contact_ages = sp.sample_n_contact_ages_with_matrix(nc,age,age_brackets,age_by_brackets_dic,age_mixing_matrix_dic[k])
                        popdict[uid]['contacts'][k] = popdict[uid]['contacts'][k].union(sp.get_n_contact_ids_by_age(uids_by_age_dic,contact_ages,age_brackets,age_by_brackets_dic))

            elif age >= activity_args['college_age_min'] and age < activity_args['college_age_max']:
                for k in ['H','S','R']:
                    if network_distr_args['network_type'] == 'poisson_degree':
                        # people at school and work? how??? college students going to school might actually look like their work environments anyways so for now this is just going to have schools and no work
                        nc = sp.pt(n_contacts_dic[k])
                        contact_ages = sp.sample_n_contact_ages_with_matrix(nc,age,age_brackets,age_by_brackets_dic,age_mixing_matrix_dic[k])
                        popdict[uid]['contacts'][k] = popdict[uid]['contacts'][k].union(sp.get_n_contact_ids_by_age(uids_by_age_dic,contact_ages,age_brackets,age_by_brackets_dic))

            elif age >= activity_args['worker_age_min'] and age < activity_args['worker_age_max']:
                for k in ['H','S','W','R']:
                    if network_distr_args['network_type'] == 'poisson_degree':
                        nc = sp.pt(non_student_n_dic[k])
                        contact_ages = sp.sample_n_contact_ages_with_matrix(nc,age,age_brackets,age_by_brackets_dic,age_mixing_matrix_dic[k])
                        popdict[uid]['contacts'][k] = popdict[uid]['contacts'][k].union(sp.get_n_contact_ids_by_age(uids_by_age_dic,contact_ages,age_brackets,age_by_brackets_dic))

            elif age >= activity_args['worker_age_max']:
                for k in ['H','R']:
                    if network_distr_args['network_type'] == 'poisson_degree':
                        nc = sp.pt(n_contacts_dic[k])
                        contact_ages = sp.sample_n_contact_ages_with_matrix(nc,age,age_brackets,age_by_brackets_dic,age_mixing_matrix_dic[k])
                        popdict[uid]['contacts'][k] = popdict[uid]['contacts'][k].union(sp.get_n_contact_ids_by_age(uids_by_age_dic,contact_ages,age_brackets,age_by_brackets_dic))

    # degree distribution may not be followed very well...
    else:
        for uid in popdict:
            age = popdict[uid]['age']
            if age < activity_args['student_age_min']:
                for k in ['H','R']:
                    if network_distr_args['network_type'] == 'poisson_degree':
                        nc = sp.pt(n_contacts_dic[k]/2)
                        contact_ages = sp.sample_n_contact_ages_with_matrix(nc,age,age_brackets,age_by_brackets_dic,age_mixing_matrix_dic[k])
                        popdict[uid]['contacts'][k] = popdict[uid]['contacts'][k].union(sp.get_n_contact_ids_by_age(uids_by_age_dic,contact_ages,age_brackets,age_by_brackets_dic))
                        for c in popdict[uid]['contacts'][k]:
                            popdict[c]['contacts'][k].add(uid)

            elif age >= activity_args['student_age_min'] and age < activity_args['student_age_max']:
                for k in ['H','S','R']:
                    if network_distr_args['network_type'] == 'poisson_degree':
                        nc = sp.pt(n_contacts_dic[k]/2)
                        contact_ages = sp.sample_n_contact_ages_with_matrix(nc,age,age_brackets,age_by_brackets_dic,age_mixing_matrix_dic[k])
                        popdict[uid]['contacts'][k] = popdict[uid]['contacts'][k].union(sp.get_n_contact_ids_by_age(uids_by_age_dic,contact_ages,age_brackets,age_by_brackets_dic))
                        for c in popdict[uid]['contacts'][k]:
                            popdict[c]['contacts'][k].add(uid)

            elif age >= activity_args['college_age_min'] and age < activity_args['college_age_max']:
                for k in ['H','S','R']:
                    if network_distr_args['network_type'] == 'poisson_degree':
                        nc = sp.pt(n_contacts_dic[k]/2)
                        contact_ages = sp.sample_n_contact_ages_with_matrix(nc,age,age_brackets,age_by_brackets_dic,age_mixing_matrix_dic[k])
                        popdict[uid]['contacts'][k] = popdict[uid]['contacts'][k].union(sp.get_n_contact_ids_by_age(uids_by_age_dic,contact_ages,age_brackets,age_by_brackets_dic))
                        for c in popdict[uid]['contacts'][k]:
                            popdict[c]['contacts'][k].add(uid)

            elif age >= activity_args['worker_age_min'] and age < activity_args['worker_age_max']:
                for k in ['H','W','R']:
                    if network_distr_args['network_type'] == 'poisson_degree':
                        nc = sp.pt(non_student_n_dic[k]/2)
                        contact_ages = sp.sample_n_contact_ages_with_matrix(nc,age,age_brackets,age_by_brackets_dic,age_mixing_matrix_dic[k])
                        popdict[uid]['contacts'][k] = popdict[uid]['contacts'][k].union(sp.get_n_contact_ids_by_age(uids_by_age_dic,contact_ages,age_brackets,age_by_brackets_dic))
                        for c in popdict[uid]['contacts'][k]:
                            popdict[c]['contacts'][k].add(uid)

            elif age >= activity_args['worker_age_max']:
                for k in ['H','R']:
                    if network_distr_args['network_type'] == 'poisson_degree':
                        nc = sp.pt(n_contacts_dic[k]/2)
                        contact_ages = sp.sample_n_contact_ages_with_matrix(nc,age,age_brackets,age_by_brackets_dic,age_mixing_matrix_dic[k])
                        popdict[uid]['contacts'][k] = popdict[uid]['contacts'][k].union(sp.get_n_contact_ids_by_age(uids_by_age_dic,contact_ages,age_brackets,age_by_brackets_dic))
                        for c in popdict[uid]['contacts'][k]:
                            popdict[c]['contacts'][k].add(uid)

    return popdict


def make_contacts(popdict,n_contacts_dic=None,state_location=None,location=None,options_args=None,activity_args=None,network_distr_args=None):
    '''
    Generates a list of contacts for everyone in the population. popdict is a
    dictionary with N keys (one for each person), with subkeys for age, sex, location,
    and potentially other factors. This function adds a new subkey, contacts, which
    is a list of contact IDs for each individual. If directed=False (default),
    if person A is a contact of person B, then person B is also a contact of person A.
    
    Example output (input is the same, minus the "contacts" field):
        popdict = {
            '8acf08f0': {
                'age': 57.3,
                'sex': 0,
                'loc': (47.6062, 122.3321),
                'contacts': ['43da76b5']
                },
            '43da76b5': {
                'age': 55.3,
                'sex': 1,
                'loc': (47.2473, 122.6482),
                'contacts': ['8acf08f0', '2d2ad46f']
                },
        }
    

    Parameters
    ----------
    popdict : dict
        From make_pop

    n_contacts_dic : dict
        Number of average contacts by setting

    state_location : str
        Name of state to call in state age mixing patterns

    location : str
        Name of location to call in age profile and in future household size

    options_args : dict
        Dictionary of options flags

    activity_args : dict
        Dictionary of actitivity age bounds, student-teacher ratio

    network_distr_args : dict
        Dictionary of network distribution args - average degree, direction, network type, 
        can also include powerlaw exponents, block sizes (re: SBMs), clustering distribution, or other properties needed to generate network structures
        checkout https://networkx.github.io/documentation/stable/reference/generators.html#module-networkx.generators for what's possible
        network_type : default is 'poisson_degree' for Erdos-Renyi random graphs in large n limit.
    '''

    if location             is None: location = 'seattle_metro'
    if state_location       is None: state_location = 'Washington'
    # using default influenza calibrated weights! Question whether these should be appropriate
    if n_contacts_dic       is None: n_contacts_dic = {'H': 4.11, 'S': 11.41, 'W': 8.07, 'R': 7}

    if network_distr_args   is None: network_distr_args = {'average_degree': 30, 'directed': False, 'network_type': 'poisson_degree'} # general we should default to undirected because directionality doesn't make sense for infectious diseases
    if 'network_type' not in network_distr_args: network_distr_args['network_type'] = 'poisson_degree'

    # why 0? Because daycares are included by default
    # why 22? Because many people in the usa context finish tertiary school of some form (vocational, community college, university)
    # why 30? best guess of student-teacher ratio - could vary and may need to be lowered to account for extra staff in schools
    # why 23 to keep ages for different activities clean
    # why 70? best guess of time when people stop working - one thing to note is eventually we'll need to consider that not everyone in the population works (somewhat accounted for by the poisson draw of n_contacts at the moment, but this should be age structure as data shows there is a definitive pattern to this)
    # activity_args might also include different n_contacts for college kids ....
    if activity_args        is None: activity_args = {'student_age_min': 0, 'student_age_max': 18, 'student_teacher_ratio': 30, 'worker_age_min': 23, 'worker_age_max': 70, 'college_age_min': 18, 'college_age_max': 23}

    options_keys = ['use_age','use_sex','use_loc','use_social_layers','use_activity_rates','use_usa']
    if options_args is None: options_args = dict.fromkeys(['use_age','use_sex','use_loc','use_social_layers','use_activity_rates','use_usa'],False)
        # here you should just create random contacts (regardless of age and sex)

    # fill in the other keys as False!
    # if any(key not in options_args for key in options_keys):
    for key in options_keys:
        if key not in options_args:
            options_args[key] = False


    if options_args['use_loc']:
        if options_args['use_usa']:

            if options_args['use_age'] and not options_args['use_social_layers']:
                popdict = make_contacts_without_social_layers_usa(popdict,n_contacts_dic,state_location,location,network_distr_args)

            if options_args['use_age'] and options_args['use_social_layers']:
                popdict = make_contacts_with_social_layers_usa(popdict,n_contacts_dic,state_location,location,activity_args,network_distr_args)
        else:
            raise NotImplementedError("Stop! I can't create other populations yet.")
    else:
        # this should probably be the generic case with a default age and sex distribution
        raise NotImplementedError("Stop! I can't create generic populations quite yet.")

    # print(options_args)
    return popdict

