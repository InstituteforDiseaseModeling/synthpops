
def make_contacts(popdict, use_age=True, use_sex=True, use_loc=False, directed=False):
    '''
    Generates a list of contacts for everyone in the population. popdict is a
    dictionary with N keys (one for each person), with subkeys for age, sex, location,
    and potentially other factors. This function adds a new subkey, contacts, which
    is a list of contact IDs for each individual. If directed=False (default),
    if person A is a contact of person B, then person B is also a contact of person
    A.
    
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
            '2d2ad46f': {
                'age': 27.3,
                'sex': 1,
                'loc': (47.2508, 122.1492),
                'contacts': ['43da76b5', '5ebc3740']
                },
            '5ebc3740': {
                'age': 28.8,
                'sex': 1,
                'loc': (47.6841, 122.2085),
                'contacts': ['2d2ad46f']
                },
        }
    '''
    
    # Do stuff :)
    
    return popdict


