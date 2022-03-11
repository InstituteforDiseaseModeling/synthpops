import synthpops as sp
import sciris as sc
import numpy as np

default_n = 1000

default_social_layers = True
directed = False

datadir = sp.settings.datadir
country_location = 'usa'
state_location = 'Washington'
location = 'seattle_metro'
sheet_name = 'United States of America'


def test_make_popdict(n=default_n):
    sc.heading(f'Making popdict for {n} people')

    popdict = sp.make_population(n=n)

    return popdict


def test_make_popdict_generic(n=default_n):
    sc.heading(f'Making popdict for {n} people')
    n = int(n)

    popdict = sp.make_population(n=n, use_demography=False)  # Non-USA not implemented

    return popdict


# Deprecated
# def test_make_popdict_supplied(n=default_n):
#     sc.heading(f'Making "supplied" popdict for {n} people')
#     n = int(n)
#     fixed_age = 40
#     fixed_sex = 1

#     uids = [i for i in np.arange(n)]
#     ages = fixed_age * np.ones(n)
#     sexes = fixed_sex * np.ones(n)

#     # Simply compile these into a dict
#     popdict = sp.make_population(uids=uids, ages=ages, sexes=sexes)

#     assert popdict[uids[0]]['age'] == fixed_age
#     assert popdict[uids[0]]['sex'] == fixed_sex

#     return popdict


# # Deprecated
# def test_make_popdict_supplied_ages(n=default_n):
#     sc.heading(f'Making "supplied" popdict for {n} people')
#     n = int(n)
#     fixed_age = 40

#     uids = [str(i) for i in np.arange(n)]
#     ages = fixed_age * np.ones(n)
#     ages[-10:] = fixed_age * 2

#     # generate sex
#     popdict = sp.make_population(uids=uids, ages=ages)

#     return popdict

# # Deprecated
# def test_make_popdict_supplied_sexes(n=default_n):
#     sc.heading(f'Making "supplied" popdict for {n} people -- skipping for now')
#     n = int(n)
#     fixed_p_sex = 0.4

#     uids = [str(i) for i in np.arange(n)]
#     sexes = np.random.binomial(1, p=fixed_p_sex, size=n)
#     sexes = None  # Skip for now since not working

#     # generate ages
#     country_location = 'usa'
#     popdict = sp.make_population(uids=uids, sexes=sexes, country_location=country_location)

#     return popdict


# # Deprecated
# def test_make_contacts(n=default_n):
#     sc.heading(f'Making contacts for {n} people')

#     popdict = popdict = sp.make_population(n=n)

#     options_args = dict.fromkeys(['use_age', 'use_sex', 'use_loc', 'use_social_layers'], True)
#     contacts = sp.make_contacts(popdict, options_args=options_args)

#     return contacts

# Skip: Deprecated API
# def test_make_contacts_and_show_some_layers(n=default_n, n_contacts_dic=None, state_location='Washington',
#                                             location='seattle_metro', country_location='usa'):
#     sc.heading(f'Make contacts for {int(n)} people and showing some layers')

#     popdict = sp.make_population(n=1e3, state_location=state_location, location=location)

#     options_args = dict.fromkeys(['use_age', 'use_sex', 'use_loc', 'use_age_mixing', 'use_social_layers'], True)
#     contacts = sp.make_contacts(popdict, n_contacts_dic=n_contacts_dic, state_location=state_location,
#                                 location=location, country_location=country_location, options_args=options_args)
#     uids = contacts.keys()
#     uids = [uid for uid in uids]
#     for n, uid in enumerate(uids):
#         if n > 20:
#             break
#         layers = contacts[uid]['contacts']
#         print('uid', uid, 'age', contacts[uid]['age'], 'total contacts',
#               np.sum([len(contacts[uid]['contacts'][k]) for k in layers]))
#         for k in layers:
#             contact_ages = [contacts[c]['age'] for c in contacts[uid]['contacts'][k]]
#             print(k, len(contact_ages), 'contact ages', contact_ages)
#         print()

#     return contacts


# # Deprecated
# def test_make_contacts_generic(n=default_n):
#     sc.heading(f'Making popdict for {n} people')
#     n = int(n)
#     popdict = sp.make_population(n=n, use_demography=False)

#     contacts = sp.make_contacts(popdict)
#     uids = contacts.keys()
#     uids = [uid for uid in uids]
#     for n, uid in enumerate(uids):
#         if n > 20:
#             break
#         layers = contacts[uid]['contacts']
#         print('uid', uid, 'age', contacts[uid]['age'], 'total contacts',
#               np.sum([len(contacts[uid]['contacts'][k]) for k in layers]))
#         for k in layers:
#             contact_ages = [contacts[c]['age'] for c in contacts[uid]['contacts'][k]]
#             print(k, len(contact_ages), 'contact ages', contact_ages)
#         print()

#     return contacts


# # Deprecated
# def test_make_contacts_use_microstructure(location='seattle_metro', state_location='Washington', n=default_n):

#     options_args = dict.fromkeys(['use_microstructure'], True)
#     network_distr_args = {'n': n}
#     contacts = sp.make_contacts(state_location=state_location, location=location, options_args=options_args, network_distr_args=network_distr_args)

#     uids = contacts.keys()
#     uids = [uid for uid in uids]
#     for n, uid in enumerate(uids):
#         if n > 20:
#             break
#         layers = contacts[uid]['contacts']
#         print('uid', uid, 'age',contacts[uid]['age'], 'total contacts', np.sum([len(contacts[uid]['contacts'][k]) for k in layers]))
#         for k in layers:
#             contact_ages = [contacts[c]['age'] for c in contacts[uid]['contacts'][k]]
#             print(k, len(contact_ages), 'contact ages', contact_ages)
#         print()

#     return contacts


# # Deprecated
# def test_make_contacts_generic_from_network_distr_args(Npop=5000):
#     popdict = sp.make_population(n=Npop)
#     network_distr_args = {'average_degree': 30, 'directed': False, 'network_type': 'poisson_degree'}
#     contacts = sp.make_contacts_generic(popdict=popdict, network_distr_args=network_distr_args)
#     uids = contacts.keys()
#     uids = [uid for uid in uids]
#     for n, uid in enumerate(uids):
#         if n > 20:
#             break
#         layers = contacts[uid]['contacts']
#         print('uid', uid, 'age', contacts[uid]['age'], 'total contacts',
#               np.sum([len(contacts[uid]['contacts'][k]) for k in layers]))
#         for k in layers:
#             contact_ages = [contacts[c]['age'] for c in contacts[uid]['contacts'][k]]
#             print(k, len(contact_ages), 'contact ages', contact_ages)
#         print()

#     return contacts


def test_make_contacts_with_facilities_from_microstructure(location='seattle_metro', state_location='Washington',
                                                           country_location='usa', n=1000):
    # First generate microstructure with facilities
    sp.make_population(n, datadir, location, state_location, country_location,
                                               write=False, plot=False, return_popdict=True)

    # sp.generate_microstructure_with_facilities(datadir, location, state_location,
    #                                         country_location, n, sheet_name='United States of America',
    #                                         use_two_group_reduction=False, average_LTCF_degree=20, ltcf_staff_age_min=20, ltcf_staff_age_max=60,
    #                                         with_school_types=False, school_mixing_type='random', average_class_size=20, inter_grade_mixing=0.1,
    #                                         average_student_teacher_ratio=20, average_teacher_teacher_degree=3, teacher_age_min=25, teacher_age_max=75,
    #                                         average_student_all_staff_ratio=15, average_additional_staff_degree=20, staff_age_min=20, staff_age_max=75,
    #                                         plot=False, write=False, return_popdict=False, use_default=False):


    # Then make contacts
    popdict = sp.make_population(n, datadir, location, state_location, country_location)

    contains_age = 74
    contains_sex = 1

    uids = popdict.keys()
    assert contains_age in uids
    assert contains_sex in uids

    return popdict


def test_make_population(location='seattle_metro', state_location='Washington', n=5000):
    contacts = sp.make_population(datadir=datadir, location=location, state_location=state_location,
                                  country_location=country_location, n=n, with_industry_code=False)
    uids = contacts.keys()
    uids = [uid for uid in uids]
    for n, uid in enumerate(uids):
        if n > 20:
            break
        layers = contacts[uid]['contacts']
        print('uid', uid, 'age', contacts[uid]['age'], 'total contacts',
              np.sum([len(contacts[uid]['contacts'][k]) for k in layers]))
        for k in layers:
            contact_ages = [contacts[c]['age'] for c in contacts[uid]['contacts'][k]]
            print(k, len(contact_ages), 'contact ages', contact_ages)
            schools = contacts[uid]['contacts']['S']
            school_id = contacts[uid]['scid']
            school_teacher = contacts[uid]['sc_teacher']
            school_student = contacts[uid]['sc_student']
            assert schools is not None
            if school_id is None:
                assert school_id is None
            else:
                assert school_id is not None
            if school_teacher is None:
                assert school_teacher is None
            else:
                assert school_teacher == 1
            if school_student is None:
                assert school_student is None
            else:

                assert school_student == 1

    return contacts


# %% Run as a script
if __name__ == '__main__':
    sc.tic()

    # datadir = sp.datadir
    datadir = sp.default_config.datadir

    location = 'seattle_metro'
    state_location = 'Washington'
    country_location = 'usa'

    n_contacts_dic = {'H': 3, 'S': 30, 'W': 30, 'C': 10}
    # contacts = test_make_contacts_and_show_some_layers(n=default_n, n_contacts_dic=n_contacts_dic,
    #                                                    state_location=state_location, location=location,
    #                                                    country_location=country_location)

    popdict = test_make_popdict(default_n)
    # popdict = test_make_popdict_supplied(default_n)
    # popdict = test_make_popdict_supplied_ages(default_n)
    # popdict = test_make_popdict_supplied_sexes(20)

    # contacts = test_make_contacts_use_microstructure(location='seattle_metro',state_location='Washington')

    popdict = test_make_popdict_generic(default_n)

    # contacts = test_make_contacts(default_n)
    # contacts = test_make_contacts_generic(default_n)
    # contacts = test_make_contacts_generic_from_network_distr_args(Npop=5000)
    contacts = test_make_contacts_with_facilities_from_microstructure(n=1000, location='seattle_metro',
                                                                      state_location='Washington',
                                                                      country_location='usa')
    # contacts = test_make_contacts_use_microstructure(location='seattle_metro', state_location='Washington', n=default_n)
    contacts = test_make_population(n=1000, location='seattle_metro', state_location='Washington')

    sc.toc()

print('Done.')
