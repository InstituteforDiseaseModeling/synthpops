import os
import synthpops as sp
import sciris as sc

default_n = 1000
default_w = {'H': 4.11, 'S': 11.41, 'W': 8.07, 'R': 2.79} # default flu-like weights


# def test_make_popdict(n=default_n):
#     sc.heading(f'Making popdict for {n} people')
    
#     popdict = sp.make_popdict(n=n)
    
#     return popdict
    
def test_make_popdict(n=default_n):
    sc.heading(f'Making popdict for {n} people')

    popdict = sp.make_popdict_n(n=n)

    return popdict


def test_make_contacts(n=default_n,weights_dic=default_w):
    sc.heading(f'Making contact matrix for {n} people')
    
    popdict = popdict = sp.make_popdict_n(n=n)
    contacts = sp.make_contacts(popdict,weights_dic=weights_dic)
    
    return contacts

#%% Run as a script
if __name__ == '__main__':
    sc.tic()

    weights_dic = {'H': 4.11, 'S': 11.41, 'W': 8.07, 'R': 2.79}

    popdict = test_make_popdict(10000)
    contacts = test_make_contacts(10000,weights_dic)

    sc.toc()


print('Done.')