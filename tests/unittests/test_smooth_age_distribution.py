import sciris as sc
import synthpops as sp
import pytest
import scipy
from scipy import spatial

# parameters to generate a test population
pars = [
    dict(
        country_location = 'usa',
        state_location   = 'Washington',
        location         = 'Spokane_County',
        use_default      = False,
    ),
    dict(
        country_location = 'usa',
        state_location   = 'Washington',
        location       = 'seattle_metro',
        use_default      = True,
    ),
    dict(
        country_location='usa',
        state_location='Oregon',
        location='portland_metro',
        use_default=False,
    ),
]


@pytest.mark.parametrize("w_len", [i for i in range(1, 10, 1)])
def test_smooth_binned_age_distribution(w_len):
    '''
    make sure window smoothing result is acceptable for valid windows length
    Args:
        w_len: test window length that is valid

    Returns:

    '''
    raw_age_distr = sp.get_smoothed_single_year_age_distr(sp.settings.datadir,
                                                          location=pars[0]['location'],
                                                          state_location=pars[0]['state_location'],
                                                          country_location=pars[0]['country_location'], window_length=1)
    smoothed_age_distr = sp.get_smoothed_single_year_age_distr(sp.settings.datadir, location=pars[0]['location'],
                                                               state_location=pars[0]['state_location'],
                                                               country_location=pars[0]['country_location'],
                                                               window_length=w_len)
    check_smooth_values(raw_age_distr, smoothed_age_distr)


@pytest.mark.parametrize("w_len", [-1, 10000])
@pytest.mark.xfail(raises=ValueError)
def test_smooth_binned_age_distribution_invalid(w_len):
    # if invalid values provided, value error should be raised
    with pytest.raises(ValueError, match=r".*non-negative integer value less than 10.*"):
        smoothed_age_distr = sp.get_smoothed_single_year_age_distr(sp.settings.datadir,
                                                                   location=pars[0]['location'],
                                                                   state_location=pars[0]['state_location'],
                                                                   country_location=pars[0]['country_location'],
                                                                   window_length=w_len)


@pytest.mark.parametrize("pars", pars)
def test_smooth_binned_age_distribution_location(pars):
    raw_age_distr = sp.get_smoothed_single_year_age_distr(sp.settings.datadir,
                                                          location=pars['location'],
                                                          state_location=pars['state_location'],
                                                          country_location=pars['country_location'], window_length=1)

    smoothed_age_distr = sp.get_smoothed_single_year_age_distr(sp.settings.datadir,
                                                               location=pars['location'],
                                                               state_location=pars['state_location'],
                                                               country_location=pars['country_location'],
                                                               use_default=pars['use_default'])

    check_smooth_values(raw_age_distr, smoothed_age_distr)


def check_smooth_values(raw_age_distr, smoothed_age_distr):
    s = [i for i in smoothed_age_distr.values()]
    r = [i for i in raw_age_distr.values()]
    d = spatial.distance.euclidean(s, r)
    print(f"distance: {str(d)}")
    # add some validation for smoothing results
    assert d < 1e-2

