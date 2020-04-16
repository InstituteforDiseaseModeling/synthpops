import synthpops as sp
import covasim as cv


if __name__ == '__main__':
    
    location = 'seattle_metro'
    state_location = 'Washington'
    country_location = 'usa'

    n = int(5e3)

    options_args = {'use_microstructure': True}
    network_distr_args = {'Npop': n}

    age_distributions = cv.data.get_age_distribution()
    
    for c in age_distributions:
        print(c, age_distributions[c])