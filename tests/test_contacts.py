import pylab as pl
import synthpops as sp
import sciris as sc
import numpy as np

default_n = 1000
# default_w = {'H': 4.11, 'S': 11.41, 'W': 8.07, 'R': 2.79} # default flu-like weights
# default_w['R'] = 7 # increase the general community weight because the calibrate weight 2.79 doesn't include contacts from the general community that you don't know but are near!

default_social_layers = True
directed = False

def test_make_popdict(n=default_n):
    sc.heading(f'Making popdict for {n} people')

    popdict = sp.make_popdict(n=n)

    return popdict


def test_make_popdict_supplied(n=default_n):
    sc.heading(f'Making "supplied" popdict for {n} people')
    
    fixed_age = 40
    fixed_sex = 1
    
    uids = [str(i) for i in pl.arange(n)]
    ages = fixed_age*pl.ones(n)
    sexes = fixed_sex*pl.ones(n)

    # Simply compile these into a dict
    popdict = sp.make_popdict(uids=uids, ages=ages, sexes=sexes)
    
    assert popdict[uids[0]]['age'] == fixed_age
    assert popdict[uids[0]]['sex'] == fixed_sex

    return popdict


# def test_make_contacts(n=default_n,weights_dic=default_w,use_social_layers=default_social_layers,directed=directed,n_contacts = 20):
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

#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    # popdict = test_make_popdict(10000)
    # contacts = test_make_contacts(10000)

    location = 'portland_metro'
    state_location = 'Oregon'

    n_contacts_dic = {'H': 3, 'S': 30, 'W': 30, 'R': 10}
    contacts = test_make_contacts_and_show_some_layers(n=10000,n_contacts_dic=n_contacts_dic,state_location=state_location,location=location)


    sc.toc()


print('Done.')
