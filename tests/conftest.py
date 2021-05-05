# allow pytest to run with --do_save --do_show --artifact parameter
# see https://docs.pytest.org/en/stable/example/parametrize.html
import pytest
import sciris as sc
import synthpops as sp

def pytest_addoption(parser):
    parser.addoption("--do-save", action="store_true", help="save all images produced by tests")
    parser.addoption("--do-show", action="store_true", help="show all images produced by tests")
    parser.addoption("--artifact-dir", action="store", help="set directory for test artifacts")


@pytest.fixture(scope='session')
def do_save(request):
    if request.config.getoption("--do-save"):
        return True
    else:
        return False


@pytest.fixture(scope='session')
def do_show(request):
    if request.config.getoption("--do-show"):
        return True
    else:
        return False


@pytest.fixture(scope='session')
def artifact_dir(request):
    if request.config.getoption("--artifact-dir"):
        return request.config.getoption("--artifact-dir")
    else:
        # set default to avoid None Type exceptions
        return "artifact"

# this is another way to set parameter for pytest but seems to work with str only
# def pytest_generate_tests(metafunc):
#     if "do_save" in metafunc.fixturenames and metafunc.config.getoption("do_save"):
#         metafunc.parametrize("do_save", metafunc.config.option.do_save)
#     if "do_show" in metafunc.fixturenames and metafunc.config.getoption("do_show"):
#         metafunc.parametrize("do_show",  metafunc.config.option.do_show)
#     if 'artifact_dir' in metafunc.fixturenames and metafunc.config.getoption("artifact_dir"):
#         metafunc.parametrize("artifact_dir", metafunc.config.option.artifact_dir)

@pytest.fixture(scope='session')
def create_default_pop():
    pars = sc.objdict(
        n=20001,
        rand_seed=1001,
        with_non_teaching_staff=True)
    return sp.Pop(**pars)
