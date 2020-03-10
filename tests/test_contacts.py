import os
import synthpops as sp
import sciris as sc


def test_make_contacts():
    ''' Run all tests -- TODO: separate into individual tests '''

    popdict = {
        '8acf08f0': {
            'age': 57.3,
            'sex': 0,
            'loc': (47.6062, 122.3321),
            # 'contacts': ['43da76b5']
            },
        '43da76b5': {
            'age': 55.3,
            'sex': 1,
            'loc': (47.2473, 122.6482),
            # 'contacts': ['8acf08f0', '2d2ad46f']
            },
        '2d2ad46f': {
            'age': 27.3,
            'sex': 1,
            'loc': (47.2508, 122.1492),
            # 'contacts': ['43da76b5', '5ebc3740']
            },
        '5ebc3740': {
            'age': 28.8,
            'sex': 1,
            'loc': (47.6841, 122.2085),
            # 'contacts': ['2d2ad46f']
            },
    }

    contacts = sp.make_contacts(popdict)
    return contacts

#%% Run as a script
if __name__ == '__main__':
    sc.tic()
    contacts = test_make_contacts()
    sc.pp(contacts) # Pretty print
    sc.toc()


print('Done.')