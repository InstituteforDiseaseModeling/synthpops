"""
This is generic way to check for methods with "path" in name and taking localtion params
To run with better subtest logging use python -m unittest test_location_methods.py
"""
import synthpops as sp
import inspect
import unittest
import os
class TestLocation(unittest.TestCase):

    @unittest.skip("in progress")
    def test_usa_path_methods(self):
        keywords = ["get", "path"]
        methods = self.get_methods_to_test(keywords=keywords)
        datadir = os.path.join(os.path.dirname(os.path.dirname(__file__)),"data")
        sp.cfg.set_location_defaults("usa")
        testcase = {"country_location": "usa", "state_location": "Washington", "location":"seattle_metro"}
        self.run_tests(datadir, methods, testcase)

    #@unittest.skip("in progress")
    def test_Senegal_path_methods(self):
        exclde_pattern = ["get_usa", "get_gender_fraction_by_age"]
        keywords = ["get", "path"]
        methods = self.get_methods_to_test(exclude_pattern=exclde_pattern, keywords=keywords)
        datadir = os.path.join(os.path.dirname(os.path.dirname(__file__)),"data")
        sp.cfg.set_location_defaults("Senegal")
        testcase = {"country_location": "Senegal", "state_location": "Dakar", "location":"Dakar"}
        self.run_tests(datadir, methods, testcase)

    @unittest.skip("in progress")
    def test_usa_data_methods(self):
        keywords = ["get", "distr"]
        methods = self.get_methods_to_test(exclude_pattern=["path"], keywords=keywords)
        for m in methods:
            print(m[0])
        datadir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        sp.cfg.set_location_defaults("usa")
        testcase = {"country_location": "usa", "state_location": "Washington", "location": "seattle_metro"}
        self.run_tests(datadir, methods, testcase, ispath=False)

    @unittest.skip("in progress")
    def test_Senegal_data_methods(self):
        keywords = ["get", "distr"]
        exclude_pattern = ["get_usa", "path", "get_gender_fraction_by_age"]
        methods = self.get_methods_to_test(exclude_pattern=exclude_pattern, keywords=keywords)
        for m in methods:
            print(m[0])
        datadir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        sp.cfg.set_location_defaults("usa")
        testcase = {"country_location": "Senegal", "state_location": "Dakar", "location": "Dakar"}
        self.run_tests(datadir, methods, testcase, ispath=False)

    def run_tests(self, datadir, methods, testcase, ispath=True):
        for m in methods:
            with self.subTest(f"{m[0]}", m=m):
                print(m[0])
                t = testcase.copy()
                if m[1]:
                    t["datadir"] = datadir
                f = getattr(sp, m[0])
                # try:
                result = f(**t)
                print(result)
                self.assertIsNotNone(result, "returns None.")
                if isinstance(result, tuple):
                    for r in result:
                        if r is not None:
                            if ispath:
                                self.assertTrue(os.path.isfile(r), f"{r} does not exist!")
                            else:
                                self.assertTrue(len(r) > 0)
                else:
                    if ispath:
                        self.assertTrue(os.path.isfile(result), f"{result} does not exist!")
                    else:
                        self.assertTrue(len(result) > 0)
                        # except Exception as e:
            #    print(str(e))

    def get_methods_to_test(self, exclude_pattern=[], keywords=["path"]):
        methods = [o for o in inspect.getmembers(sp) if inspect.isfunction(o[1])]
        path_methods = []
        for m in methods:
            k = [k in m[0] for k in keywords]
            if False in k:
                continue
            ep = [p in m[0] for p in exclude_pattern]
            if True in ep:
                continue
            args = inspect.getfullargspec(m[1]).args
            matched = ([x in ['country_location', 'state_location', 'location'] for x in args].count(True) == 3)
            if matched:
                # print(m[0])
                if "datadir" in args:
                    path_methods.append((m[0], True))
                else:
                    path_methods.append((m[0], False))
        return path_methods