import unittest
import os
import pathlib
import subprocess

class TestExample(unittest.TestCase):

    #some example needs external library
    excluded =['draw_multilayer_network.py']

    @classmethod
    def setUpClass(cls) -> None:
        cls.example_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'examples')
        cls.resultdir = os.path.join(os.path.dirname(__file__), 'result')
        os.makedirs(cls.resultdir, exist_ok=True)

    def run_examples(self, commandargs, workingdir):
        succeeded = False
        cmd = ['python']
        commandargs = [commandargs] if isinstance(commandargs, str) else commandargs
        for a in commandargs:
            cmd.append(str(a))
        try:
            print("run:", str(cmd))
            outfilename = str(commandargs[0]).replace('.py', '.log')
            errfilename = str(commandargs[0]).replace('.py', '.err.log')
            with open(os.path.join(self.resultdir, outfilename), 'w') as fout,\
                    open(os.path.join(self.resultdir, errfilename), 'w') as ferr:
                retcode = subprocess.check_call(cmd, shell=False, cwd=workingdir, stdout=fout, stderr=ferr)
            succeeded = True if retcode == 0 else False
            succeeded = succeeded and self.check_log(os.path.join(self.resultdir, errfilename))
        except subprocess.CalledProcessError as e:
            print("error:", str(cmd))
            #print(str(e))
        print("result:", succeeded)
        return succeeded

    def check_log(self, logfile):
        if pathlib.Path(logfile).stat().st_size > 0:
            with open(logfile) as f:
                output = f.readlines()
            if 'error' in str(output).lower():
                return False
        return True

    def test_examples(self):
        passed = True
        result = {}
        for root, dirnames, filenames in os.walk(self.example_path):
            for f in filenames:
                if str(f).endswith('.py') and f not in self.excluded:
                    print("-------------")
                    print(f)
                    passed = self.run_examples(f, workingdir=root)
                    result[f] = 'passed' if passed else 'failed'

        for i in result:
            print(i, " : ", result[i])
        self.assertFalse('failed' in result.values())