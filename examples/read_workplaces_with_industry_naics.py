import synthpops as sp

if __name__ == '__main__':
    n = int(5e4)
    population = sp.make_population(n=n, with_industry_code=True)

    uids = population.keys()
    uids = [uid for uid in uids]

    # show age, hhid, scid, wpid, and wnaics
    # default values for hhid, scid, wpid, wnaics = -1
    # everyone has a home so no one should have hhid = -1
    # not everyone will be in a school or workplace so some of these will be -1
    for i in range(100):
        person = population[uids[i]]
        print(person.keys())
        print('age', person['age'], 'household id', person['hhid'], 'school id', person['scid'], 'workplace id',
              person['wpid'], 'workplace 2017 code', person['wpindcode'])
