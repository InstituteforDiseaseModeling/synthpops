import synthpops as sp
import pytest

if not sp.config.full_data_available:
    pytest.skip("Data not available, tests not possible", allow_module_level=True)


def test_api():

    n = 5000
    max_contacts = {'S':20, 'W':10}

    network = sp.make_network(n=n, max_contacts=max_contacts)

    return network


if __name__ == '__main__':
    contacts = test_api()
    print('Done.')