import os
import synthpops as sp
import sciris as sc

default_n = 1000

def test_make_popdict(n=default_n):
    sc.heading(f'Making popdict for {n} people')
    
    popdict = sp.make_popdict(n=n)
    
    return popdict
    


def test_make_contacts(n=default_n):
    sc.heading(f'Making contact matrix for {n} people')
    
    popdict = popdict = sp.make_popdict(n=n)
    contacts = sp.make_contacts(popdict)
    
    return contacts

#%% Run as a script
if __name__ == '__main__':
    sc.tic()
    popdict = test_make_popdict()
    contacts = test_make_contacts()
    sc.pp(contacts) # Pretty print
    sc.toc()


print('Done.')