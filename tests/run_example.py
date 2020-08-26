'''
This file can be run with

    pytest -s run_example.py

or

    python run_example.py
'''

import unittest
import os
import sciris as sc
import subprocess
from pathlib import Path
from multiprocessing import Pool


class TestExample(unittest.TestCase):

    # some examples are under construction
    # or need to download external library (e.g. pymnet) manually
    excluded =['read_workplaces_with_industry_naics.py',
               'draw_multilayer_network.py']

    n_procs = 4  # Number of parallel processes to run
    parallelize = True

    @classmethod
    def setUpClass(cls) -> None:
        cls.example_path = Path(sc.thisdir(__file__)).joinpath('../examples')
        cls.resultdir = os.path.join(os.path.dirname(__file__), 'result')
        os.makedirs(cls.resultdir, exist_ok=True)

    @classmethod
    def run_examples(cls, workingdir, resultdir, commandargs):
        T = sc.tic()
        succeeded = False
        cmd = ['python']
        commandargs = [commandargs] if isinstance(commandargs, str) else commandargs
        for a in commandargs:
            cmd.append(str(a))
        try:
            print("run:", str(cmd))
            outfilename = str(commandargs[0]).replace('.py', '.log')
            errfilename = str(commandargs[0]).replace('.py', '.err.log')
            with open(os.path.join(resultdir, outfilename), 'w') as fout,\
                    open(os.path.join(resultdir, errfilename), 'w') as ferr:
                retcode = subprocess.check_call(cmd, shell=False, cwd=workingdir, stdout=fout, stderr=ferr)
            succeeded = True if retcode == 0 else False
            succeeded = succeeded and cls.check_log(os.path.join(resultdir, errfilename))
        except subprocess.CalledProcessError as e:
            print("error:", str(cmd), str(e))
        elapsed = sc.toc(T, output=True)
        succeeded = f"passed in {elapsed:0.1f} s" if succeeded else "failed"
        print(f"{commandargs[0]} finished: {succeeded}")
        return {commandargs[0]: succeeded}

    @classmethod
    def check_log(cls, logfile):
        if Path(logfile).stat().st_size > 0:
            with open(logfile) as f:
                output = f.readlines()
            if 'error' in str(output).lower():
                return False
        return True

    def test_examples(self):
        files = []
        finalresult = {}
        for root, dirnames, filenames in os.walk(self.example_path):
            for f in filenames:
                if str(f).endswith('.py') and f not in self.excluded:
                    #print("-------------")
                    #print(f)
                    files.append(f)

        t = [(self.example_path, self.resultdir, f) for f in files]
        #user 4 parallel processes to run
        if self.parallelize:
            with Pool(processes=self.n_procs) as pool:
                result = pool.starmap(self.run_examples, t)
        else:
            result = []
            for entry in t:
                result.append(self.run_examples(*entry))

        print("-------------")
        for r in result:
            finalresult.update(r)
            print(r)

        self.assertFalse('failed' in finalresult.values())


# Run unit tests if called as a script
if __name__ == '__main__':
    TestExample.parallelize = False
    unittest.main()