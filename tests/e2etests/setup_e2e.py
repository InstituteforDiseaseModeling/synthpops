"""
This file defines the fixture of e2e tests, note that fixtures can be overridden if the same name is used
please be cautious when using the same name in your test module
more details see https://docs.pytest.org/en/stable/fixture.html#override-fixtures
"""
import os
import pytest
import pathlib
import sciris as sc
import synthpops as sp

# parameters to generate a default population
sample_pars = sc.objdict(
    n                               = 20001,
    rand_seed                       = 1,
    country_location                = 'usa',
    state_location                  = 'Washington',
    location                        = 'seattle_metro',
    use_default                     = True,

    with_facilities                 = 1,
    with_non_teaching_staff         = 1
)

@pytest.fixture(scope="session")
def create_sample_pop_e2e():
    """
        fixture to create and return a sample population for the session

    Returns:
        a sample population based on sample_pars
    """
    sp.logger.info("Use sample_pars to create sample_pop population")
    sample_pop = sp.Pop(**sample_pars)
    sp.logger.info("If you define a fixture with the same name, this fixture may be overridden!")
    return sample_pop

@pytest.fixture(scope="function")
def get_fig_dir(request, artifact_dir):
    """
        fixture to create and return a subdirectory with function name under artifact_dir

    Returns:
        filepath with function name as subfolder
    """
    testname = request.node.originalname if request.node.originalname is not None else request.node.name
    fig_dir = pathlib.Path(artifact_dir, testname)
    os.makedirs(fig_dir, exist_ok=True)
    return fig_dir

@pytest.fixture(scope="module")
def get_fig_dir_by_module(request, artifact_dir):
    """
        fixture to create and return a subdirectory with module name under artifact_dir

    Returns:
        filepath with module name as subfolder
    """
    modulename = request.node.name
    start = modulename.rfind("/")+1
    end = modulename.rfind(".")
    fig_dir = pathlib.Path(artifact_dir, modulename[start:end])
    os.makedirs(fig_dir, exist_ok=True)
    return fig_dir