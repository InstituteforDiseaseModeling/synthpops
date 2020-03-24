import synthpops as sp
import sciris as sc
import numpy as np
import pytest

default_n = 1000
# default_w = {'H': 4.11, 'S': 11.41, 'W': 8.07, 'R': 2.79} # default flu-like weights
# default_w['R'] = 7 # increase the general community weight because the calibrate weight 2.79 doesn't include contacts from the general community that you don't know but are near!

default_social_layers = True
directed = False

def test_make_popdict(n=default_n):
    sc.heading(f'Making popdict for {n} people')

    popdict = sp.make_popdict(n=n)

    return popdict


def test_make_popdict_generic(n=default_n):
    sc.heading(f'Making popdict for {n} people')
    n = int(n)
    popdict = None # Since expect to fail
    with pytest.raises(NotImplementedError):
        popdict = sp.make_popdict(n=n,use_usa=False) # Non-USA not implemented

    return popdict


def test_make_popdict_supplied(n=default_n):
    sc.heading(f'Making "supplied" popdict for {n} people')
    n = int(n)
    fixed_age = 40
    fixed_sex = 1

    uids = [str(i) for i in np.arange(n)]
    ages = fixed_age*np.ones(n)
    sexes = fixed_sex*np.ones(n)

    # Simply compile these into a dict
    popdict = sp.make_popdict(uids=uids, ages=ages, sexes=sexes)

    assert popdict[uids[0]]['age'] == fixed_age
    assert popdict[uids[0]]['sex'] == fixed_sex

    return popdict


def test_make_popdict_supplied_ages(n=default_n):
    sc.heading(f'Making "supplied" popdict for {n} people')
    n = int(n)
    fixed_age = 40

    uids = [str(i) for i in np.arange(n)]
    ages = fixed_age*np.ones(n)
    ages[-10:] = fixed_age*2

    # generate sex
    popdict = sp.make_popdict(uids=uids, ages=ages)

    return popdict


def test_make_popdict_supplied_sexes(n=default_n):
    sc.heading(f'Making "supplied" popdict for {n} people')
    n = int(n)
    fixed_p_sex = 0.6

    uids = [str(i) for i in np.arange(n)]
    sexes = np.random.binomial(1, p = fixed_p_sex,size = n)

    # generate ages
    popdict = sp.make_popdict(uids=uids,sexes=sexes)

    return popdict


def test_make_contacts(n=default_n):
    sc.heading(f'Making contacts for {n} people')

    popdict = popdict = sp.make_popdict(n=n)

    options_args = dict.fromkeys(['use_age','use_sex','use_loc','use_usa','use_social_layers'], True)
    contacts = sp.make_contacts(popdict,options_args = options_args)

    return contacts


def test_make_contacts_and_show_some_layers(n=default_n,n_contacts_dic=None,state_location='Oregon',location='portland_metro'):
    sc.heading(f'Make contacts for {int(n)} people and showing some layers')

    popdict = sp.make_popdict(n=1e4,state_location=state_location,location=location)

    options_args = dict.fromkeys(['use_age','use_sex','use_loc','use_usa','use_social_layers'], True)
    contacts = sp.make_contacts(popdict,n_contacts_dic=n_contacts_dic,state_location=state_location,location=location,options_args=options_args)
    uids = contacts.keys()
    uids = [uid for uid in uids]
    for n,uid in enumerate(uids):
        if n > 20:
            break
        layers = contacts[uid]['contacts']
        print('uid',uid,'age',contacts[uid]['age'],'total contacts', np.sum([len(contacts[uid]['contacts'][k]) for k in layers]))
        for k in layers:
            contact_ages = [contacts[c]['age'] for c in contacts[uid]['contacts'][k]]
            print(k,len(contact_ages),'contact ages',contact_ages)
        print()

    return contacts


def test_make_contacts_generic(n=default_n):
    sc.heading(f'Making popdict for {n} people')
    n = int(n)
    popdict = sp.make_popdict(n=n,use_usa=True)

    contacts = sp.make_contacts(popdict)
    uids = contacts.keys()
    uids = [uid for uid in uids]
    for n,uid in enumerate(uids):
        if n > 20:
            break
        layers = contacts[uid]['contacts']
        print('uid',uid,'age',contacts[uid]['age'],'total contacts', np.sum([len(contacts[uid]['contacts'][k]) for k in layers]))
        for k in layers:
            contact_ages = [contacts[c]['age'] for c in contacts[uid]['contacts'][k]]
            print(k,len(contact_ages),'contact ages',contact_ages)
        print()

    return contacts


def test_make_contacts_from_microstructure(location='seattle_metro',state_location='Washington',Nhomes=50000):

    popdict = {}
    options_args = dict.fromkeys(['use_microstructure'],True)
    contacts = sp.make_contacts(popdict,state_location=state_location,location=location,options_args=options_args)
    uids = contacts.keys()
    uids = [uid for uid in uids]
    for n,uid in enumerate(uids):
        if n > 20:
            break
        layers = contacts[uid]['contacts']
        print('uid',uid,'age',contacts[uid]['age'],'total contacts', np.sum([len(contacts[uid]['contacts'][k]) for k in layers]))
        for k in layers:
            contact_ages = [contacts[c]['age'] for c in contacts[uid]['contacts'][k]]
            print(k,len(contact_ages),'contact ages',contact_ages)
        print()

    return contacts


#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    popdict = test_make_popdict(default_n)
    contacts = test_make_contacts(default_n)

    location = 'portland_metro'
    state_location = 'Oregon'

    n_contacts_dic = {'H': 3, 'S': 30, 'W': 30, 'R': 10}
    contacts = test_make_contacts_and_show_some_layers(n=default_n,n_contacts_dic=n_contacts_dic,state_location=state_location,location=location)

    popdict = test_make_popdict_supplied(default_n)
    popdict = test_make_popdict_supplied_ages(default_n)
    popdict = test_make_popdict_supplied_sexes(20)
    popdict = test_make_popdict_generic(default_n)

    contacts = test_make_contacts_generic(default_n)
    test_make_contacts_from_microstructure(location='seattle_metro',state_location='Washington',Nhomes=50000)
    sc.toc()





print('Done.')
