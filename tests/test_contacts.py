import pylab as pl
import synthpops as sp
import sciris as sc

default_n = 1000
default_w = {'H': 4.11, 'S': 11.41, 'W': 8.07, 'R': 2.79} # default flu-like weights
default_w['R'] = 7 # increase the general community weight because the calibrate weight 2.79 doesn't include contacts from the general community that you don't know but are near!

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


def test_make_contacts(n=default_n,weights_dic=default_w,use_social_layers=default_social_layers,directed=directed):
    sc.heading(f'Making contact matrix for {n} people')

    max_n = 20

    popdict = sp.make_popdict(n=n)
    contacts = sp.make_contacts(popdict,weights_dic=weights_dic,use_social_layers = use_social_layers,directed=directed,use_student_weights = True)

    uids = contacts.keys()
    uids = [uid for uid in uids]

    for n in range(max_n):
        uid = uids[n]

        print('uid',uid,'age',contacts[uid]['age'])
        for k in weights_dic.keys():
            contact_ages = [contacts[c]['age'] for c in contacts[uid]['contacts'][k]]
            print(k,'contact ages', contact_ages)

    return contacts

#%% Run as a script
if __name__ == '__main__':
    weights_dic = {'H': 4.11, 'S': 11.41, 'W': 8.07, 'R': 2.79}
    weights_dic['R'] = 7 # increase the general community weight because the calibrate weight 2.79 doesn't include contacts from the general community that you don't know but are near!

    sc.tic()
    popdict = test_make_popdict(3000)
    sc.toc()

    sc.tic()
    contacts = test_make_contacts(10000, weights_dic, use_social_layers=True, directed=False)
    sc.toc()



print('Done.')
