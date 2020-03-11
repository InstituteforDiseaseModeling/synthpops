import pylab as pl
import synthpops as sp
import sciris as sc

default_n = 1000
default_w = {'H': 4.11, 'S': 11.41, 'W': 8.07, 'R': 2.79} # default flu-like weights


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


def test_make_contacts(n=default_n,weights_dic=default_w):
    sc.heading(f'Making contact matrix for {n} people')
    
    popdict = popdict = sp.make_popdict(n=n)
    contacts = sp.make_contacts(popdict,weights_dic=weights_dic)
    
    return contacts

#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    popdict = test_make_popdict()
    contacts = test_make_contacts()

    sc.toc()


print('Done.')