import difflib
import filecmp
import fnmatch
import os
import pytest
from pathlib import Path
import shutil
import sciris as sc
import numpy as np
import pandas as pd
import synthpops as sp
import tempfile
from matplotlib import pyplot as plt

try:
    from fpdf import FPDF
except Exception as E:
    print(f'Note: could not import fpdf, report not available ({E})')


@pytest.fixture(scope='function')
def get_regression_dir(request):
    # this fixture will setup the subdirectory for module under regression folder
    regression_root = Path(__file__).parent.absolute().joinpath('regression')
    run_dir = regression_root.joinpath('report', request.node.name)
    expected_dir = regression_root.joinpath('expected', request.node.name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(expected_dir, exist_ok=True)
    return expected_dir, run_dir

@pytest.fixture(scope='function')
def regression_run(get_regression_dir):
    #cleanup files from previous runs
    if get_regression_dir[1].exists() and get_regression_dir[1].is_dir():
        shutil.rmtree(get_regression_dir[1])
    # run task generator will be returned when fixture is called and it is executed when parameters are passed in
    # a func and its input parameters is needed and func must output a valid object for persistence
    def run_task(get_regression_dir, func, params, filename, generate=False, decimal=3):
        # you can pass func and params to generate a baseline object or pass an object directly as func
        if callable(func):
            if len(str(func).rsplit('.', 1)) > 1:
                module_name, func_name = str(func).rsplit('.', 1)
                func = getattr(module_name, func_name)
            if params is None:
                obj = func()
            else:
                obj = func(**params)
        else:
            obj = func
        if type(obj) not in [list, np.ndarray, pd.DataFrame, str, dict, sp.Pop, sc.objdict]:
            raise ValueError("function must return a valid Pop object or string, dictionary, dataframe or array")
        generated_filename = str(Path(get_regression_dir[1], filename))
        regression_persist(obj, generated_filename)
        if generate:
            # copy files to expected location if generate is True
            expected_filename = str(Path(get_regression_dir[0], filename))
            shutil.copy(generated_filename, expected_filename)
    return run_task

@pytest.fixture(scope='function')
def regression_validate(get_regression_dir):
    def validate(get_regression_dir, decimal=3, generate=False, force_report=False):
        if generate:
            print("skip validation")
            return
        passed = True
        checked = False
        failed_cases = []
        expected_folder, actual_folder = get_regression_dir
        if not os.path.exists(actual_folder):
            raise FileNotFoundError(actual_folder)
        if not os.path.exists(expected_folder):
            raise AssertionError(f"{expected_folder} does not exist, use regenerate = True to generate them")
        expected_files = [f for f in os.listdir(expected_folder) if Path(f).suffix in [".csv", ".json", ".txt"]]
        if len(expected_files) == 0:
            raise AssertionError(f"no files to validate in {expected_folder}, use regenerate = True to generate them")
        #loop over all valid baseline files for comparison
        for f in expected_files:
            print(f"\n{f}")
            checked = True
            if f.endswith(".csv"):
                expected_data = np.loadtxt(os.path.join(expected_folder, f), delimiter=",")
                actual_data = np.loadtxt(os.path.join(actual_folder, f), delimiter=",")
                if (np.round(expected_data, decimal) == np.round(actual_data, decimal)).all():
                    print("values unchanged, passed")
                else:
                    passed = False
                    failed_cases.append(os.path.basename(f).replace(".csv", "*"))
                    print("result has been changed in these indexes:\n", np.where(expected_data != actual_data)[0])
            elif f.endswith(".json"):
                expected_data = sc.loadjson(os.path.join(expected_folder, f))
                actual_data = sc.loadjson(os.path.join(actual_folder, f))
                if expected_data == actual_data:
                    print("values unchanged, passed")
                else:
                    passed = False
                    failed_cases.append(os.path.basename(f).replace(".json", "*"))
            elif f.endswith(".txt"):
                result = check_files_diff(os.path.join(expected_folder, f), os.path.join(actual_folder, f))
                if not result:
                    failed_cases.append(os.path.basename(f).replace(".txt", "*"))
                passed = result
            else:
                print("ignored.\n")
        if not (passed & checked) or force_report:
            # generate test reports if failed
            pdf_path = generate_reports(get_regression_dir=get_regression_dir,
                                        failedcases=failed_cases,
                                        forced=force_report)
        assert passed & checked, f"regression test failed! Please review report at: {pdf_path}"
    return validate


def regression_persist(obj, filename, decimal=3):
    # persist files for regression run
    if str(filename).endswith('json'):
        if isinstance(obj, sp.Pop):
            obj.to_json(filename)
        else:
            sc.savejson(filename, obj)
    elif str(filename).endswith('csv') or str(filename).endswith('txt'):
        if isinstance(obj, np.ndarray) or isinstance(obj, list):
            fmt = f'%.{str(decimal)}f'
            np.savetxt(filename, np.array(obj), delimiter=",", fmt=fmt)
        elif isinstance(obj, pd.DataFrame):
            obj.to_csv(filename)
        elif isinstance(obj, str):
            with open(filename, "w") as f:
                f.write(obj)
        else:
            raise ValueError(f"Unrecognized object type: {type(obj)} ")
    else:
        raise ValueError(f"Invalid filename: {filename} only json / csv /txt supported!")
    print(f"file saved: {filename}")


def check_files_diff(expected_file, actual_file):
    """
        check if two files' contents are identical and show diff
        Args:
            expected_file (str): expected text file
            actual_file (str): actual text file

        Returns:
        tuple(bool, list):  first element set to True if files are identical, False otherwise.
                            second element is the list of differences
    """
    compare_result = filecmp.cmp(expected_file, actual_file, shallow=False)
    print(f"expected file: {expected_file}")
    print(f"actual file: {actual_file}")
    diffs = []
    if not compare_result:
        with open(expected_file) as ef, open(actual_file) as af:
            eflines = ef.readlines()
            aflines = af.readlines()
            d = difflib.Differ()
            diffs = [x for x in d.compare(eflines, aflines) if x[0] in ('+', '-')]
            if diffs:
                print("differences found:")
                for x in diffs:
                    print(x)
            else:
                print('No changes detected')
    else:
        print("file check completed.")
    return compare_result, diffs

def check_figs_diff(expected_figname, actual_figname, color=[1,0,0]):
    """
    compare expected and actual images and mark the differences to the color specified
    Args:
        expected_figname (str): expected image file
        actual_figname (str): actual image file
        color (list): [Red, Blue, Green] format to specify a color value to highlight with must be between 0 and 1

    Returns:
        diff image filename (str)
        a diff image file will be saved as actual_filename_diff in the folder which actual_figname resides
    """
    expected_fig = plt.imread(expected_figname)
    actual_fig = plt.imread(actual_figname)
    # Calculate difference between two images
    diff = np.subtract(expected_fig, actual_fig)
    expected_copy = expected_fig.copy()[:,:,0:3]
    r, g, b = diff[:, :, 0], diff[:, :, 1], diff[:, :, 2]
    # calculate grayscale of differences as mask
    diff_gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    # apply the mask to the expected images to make the differences obvious (red)
    expected_copy[diff_gray>0, :]=[1, 0, 0]
    diff_file = f"{os.path.splitext(actual_figname)[0]}_diff{os.path.splitext(actual_figname)[1]}"
    plt.imsave(diff_file, expected_copy)
    return diff_file

def generate_reports(get_regression_dir, failedcases=[], forced=False, name="report"):
    """
    generate pdf reports with expected /actual figs and their differences for specified test cases
    Args:
        test_prefix (str): test prefix to identify the testcase
        failedcases (list): list of failed cases, used to generate the graphs
        forced (bool): if specified, force the report generation
        name (str) : filename of the pdf report if specified

    Returns:
        str: the path of the pdf report file
    """
    if not forced and len(failedcases) == 0:
        print("no failures, skip reporting")
        return
    expected_dir, actual_dir = get_regression_dir
    pdf = FPDF()
    # pdf.add_page()
    pdf.set_font("Arial", size=12)
    print(f"failed cases:{failedcases}")
    if forced:
        print("force generating reports for all images and diffs")
        failedcases = [f for f in os.listdir(actual_dir)]
    for ff in [f for f in os.listdir(actual_dir)]:
        print(f"checking:{ff}")
        for fc in failedcases:
            if fnmatch.fnmatch(ff, fc):
                print(f"matching {fc} -> {ff}")
                pdf.add_page()
                expectedfile = str(Path(expected_dir,ff))
                actualfile = str(Path(actual_dir,ff))
                checked, diff = check_files_diff(expectedfile, actualfile)
                if not checked or forced:
                    if ff.endswith("txt") or ff.endswith("json"):
                        contents = "\n".join([line.strip() for line in diff])
                        contents = f"difference in {ff}\n {len(diff)} found.\n" + contents
                        print(contents)
                        pdf.multi_cell(w=180, h=5, txt=contents, border=1)
                    elif ff.endswith("csv"):
                        actual_data = np.loadtxt(actualfile, delimiter=',')
                        actual_data /= np.nanmax(actual_data)
                        expected_data = np.loadtxt(expectedfile, delimiter=',')
                        actual_data /= np.nanmax(actual_data)
                        expected_fig = f"{actualfile}.expected.png"
                        plt.imshow(expected_data)
                        plt.savefig(expected_fig)
                        actual_fig = f"{actualfile}.png"
                        plt.imshow(actual_data)
                        plt.savefig(actual_fig)
                        diff_file = check_figs_diff(expected_fig, actual_fig)
                        #pdf.add_page()
                        pdf.cell(w=180, h=5, txt=ff, border=1)
                        pdf.set_y(20)
                        pdf.image(actual_fig, w=60, h=60)
                        pdf.cell(w=180, h=5, txt="expected", border=1)
                        pdf.set_y(100)
                        pdf.image(expected_fig, w=60, h=60)
                        pdf.set_y(170)
                        pdf.cell(w=180, h=5, txt="diff with baseline highlighted in red", border=1)
                        pdf.set_y(180)
                        pdf.image(diff_file, w=60, h=60)
                    break
    pdf_path = os.path.join(actual_dir, f"{name}.pdf")
    pdf.output(pdf_path)
    print("report generated:", pdf_path)
    return pdf_path
