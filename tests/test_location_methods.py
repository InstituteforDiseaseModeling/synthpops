"""
This is generic way to check for methods with "path" in name and taking localtion params
To run with better subtest logging use python -m unittest test_location_methods.py
"""
import synthpops as sp
import inspect
import unittest
import os
class TestLocation(unittest.TestCase):

    @unittest.skip("this is used to check get path")
    def test_methods(self):
        methods = self.get_methods_to_test()
        datadir = os.path.join(os.path.dirname(os.path.dirname(__file__)),"data")
        testcases = [{"datadir": datadir, "country_location": "usa", "state_location": "Washington", "location":"seattle_metro"},
                     {"datadir": datadir, "country_location": "Senegal", "state_location": "Dakar", "location":"Dakar"}]
        for t in testcases:
            with self.subTest(f"test location:{t}", t=t):
                for m in methods:
                    with self.subTest(f"{m}", m=m):
                        print(m)
                        f = getattr(sp, m)
                        #try:
                        result = f(**t)
                        print(result)
                        self.assertIsNotNone(result, "returns None.")
                        if isinstance(result, tuple):
                            for r in result:
                                if r is not None:
                                    self.assertTrue(os.path.isfile(r), f"{r} does not exist!")
                        else:
                            self.assertTrue(os.path.isfile(result), f"{result} does not exist!")
                    #except Exception as e:
                    #    print(str(e))

    def get_methods_to_test(self):
        exclude = "get_usa_long_term_care_facility"
        methods = [o for o in inspect.getmembers(sp) if inspect.isfunction(o[1])]
        path_methods = []
        for m in methods:
            if exclude not in m[0] and 'path' in m[0]:
                args = inspect.getfullargspec(m[1]).args
                matched = ([x in ['datadir', 'country_location', 'state_location', 'location'] for x in args].count(True) == 4)
                if matched:
                    #print(m[0])
                    path_methods.append(m[0])
        return path_methods