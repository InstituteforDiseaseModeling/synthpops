import pytest
import synthpops as sp


def test_sizes():
    ''' Synthpops should support populations down to 200 people '''

    nlist = [100, 246]

    for n in nlist:
        print(f'Working on {n}')
        pop = sp.make_population(n=n, generate=True)
        assert len(pop) == n

    with pytest.raises(NotImplementedError):
        sp.make_population(n=99, generate=True) # Too small

    return pop


if __name__ == '__main__':
    pop = test_sizes()
    print('Done.')