"""
This is a generic way to check for methods with "path" in the name and taking location parameters.
To run with better subtest logging use python -m unittest test_location_methods.py

each test use keywords to identify methods to tests
and if the data is not available, exclude_pattern is used to ignore them
"""
import synthpops as sp
import inspect
import unittest
import os
from synthpops import cfg


class TestLocation(unittest.TestCase):
    # these methods takes "location" as arguments, however data was not available so tests will skip them
    location_ignoreArgs = [

    ]

    def test_spokane_path_methods(self):
        cfg.set_nbrackets(20)
        keywords = ["get", "path"]
        exclude_pattern = ["household_head_age_by_size", "head_age_brackets"]
        methods = self.get_methods_to_test(keywords=keywords, exclude_pattern=exclude_pattern)
        datadir = sp.settings.datadir
        # sp.cfg.set_location_defaults("usa")
        testcase = {"country_location": "usa", "state_location": "Washington", "location": "Spokane_County"}
        self.run_tests(datadir, methods, testcase)

    def test_seattle_path_methods(self):
        cfg.set_nbrackets(20)
        keywords = ["get", "path"]
        exclude_pattern = ["household_head_age_by_size"]
        methods = self.get_methods_to_test(keywords=keywords, exclude_pattern=exclude_pattern)
        datadir = sp.settings.datadir
        # sp.cfg.set_location_defaults("usa")
        testcase = {"country_location": "usa", "state_location": "Washington", "location": "seattle_metro"}
        self.run_tests(datadir, methods, testcase)

    def test_portland_methods(self):
        cfg.set_nbrackets(16)
        keywords = ["get", "path"]
        exclude_pattern = ["school", "long_term_care_facility", "household_head_age_by_size"]
        methods = self.get_methods_to_test(keywords=keywords, exclude_pattern=exclude_pattern)
        datadir = sp.settings.datadir
        # sp.cfg.set_location_defaults("usa")
        testcase = {"country_location": "usa", "state_location": "Oregon", "location": "portland_metro"}
        self.run_tests(datadir, methods, testcase)

    def test_usa_state_path_methods(self):
        cfg.set_nbrackets(20)
        keywords = ["get", "path"]
        args = ['country_location', 'state_location']
        ignored_args = ['location', 'part']
        exclude_pattern = ["school", "long_term_care_facility"]
        methods = self.get_methods_to_test(keywords=keywords, target_args=args,
                                           exclude_args=ignored_args, exclude_pattern=exclude_pattern)
        datadir = sp.settings.datadir
        # sp.cfg.set_location_defaults("usa")
        testcase = {"country_location": "usa", "state_location": "Washington"}
        self.run_tests(datadir, methods, testcase)

    @unittest.skip("Data not available in location DAKAR")
    def test_dakar_path_methods(self):
        cfg.set_nbrackets(18)
        exclude_pattern = ["get_usa", "get_gender_fraction_by_age",
                           "get_school_size_distr_by_type", "get_school_type_age_ranges_path",
                           "get_long_term_"]
        keywords = ["get", "path"]
        methods = self.get_methods_to_test(exclude_pattern=exclude_pattern, keywords=keywords)
        datadir = sp.settings.datadir
        # sp.cfg.set_location_defaults("Senegal")
        testcase = {"country_location": "Senegal", "state_location": "Dakar", "location": "Dakar"}
        self.run_tests(datadir, methods, testcase)

    def test_senegal_state_path_methods(self):
        cfg.set_nbrackets(18)
        exclude_pattern = ["get_usa", "get_gender_fraction_by_age",
                           "get_school_size_distr_by_type", "get_school_type_age_ranges_path",
                           "get_long_term_"]
        keywords = ["get", "path"]
        args = ['country_location', 'state_location']
        ignored_args = ['location', 'part']
        methods = self.get_methods_to_test(exclude_pattern=exclude_pattern, keywords=keywords, target_args=args,
                                           exclude_args=ignored_args)
        datadir = sp.settings.datadir
        # sp.cfg.set_location_defaults("Senegal")
        testcase = {"country_location": "Senegal", "state_location": "Dakar"}
        self.run_tests(datadir, methods, testcase)

    def test_seattle_data_methods(self):
        cfg.set_nbrackets(20)
        keywords = ["get", "distr"]
        exclde_pattern = ["path", "head_age_by_size", "workplace_size"]
        methods = self.get_methods_to_test(exclude_pattern=exclde_pattern, keywords=keywords)
        for m in methods:
            print(m[0])
        datadir = sp.settings.datadir
        # sp.cfg.set_location_defaults("usa")
        testcase = {"country_location": "usa", "state_location": "Washington", "location": "seattle_metro"}
        self.run_tests(datadir, methods, testcase, ispath=False)

    @unittest.skip("Data not available in location DAKAR")
    def test_dakar_data_methods(self):
        cfg.set_nbrackets(18)
        keywords = ["get", "distr"]
        exclude_pattern = ["get_usa", "path", "get_gender_fraction_by_age", "get_age_bracket",
                           "get_school_size_distr_by_type", "get_school_type_age_ranges_path",
                           "get_long_term_"]
        methods = self.get_methods_to_test(exclude_pattern=exclude_pattern, keywords=keywords)
        for m in methods:
            print(m[0])
        datadir = sp.settings.datadir
        testcase = {"country_location": "Senegal", "state_location": "Dakar", "location": "Dakar"}
        self.run_tests(datadir, methods, testcase, ispath=False)

    def run_tests(self, datadir, methods, testcase, ispath=True):
        """
        Passing testcase as input arguments to location methods
        and make sure either the path exists or method returns data

        Args:
            datadir  : the root of data directory
            methods  : list of methods to test, must be objects returned from inspect.getmembers()
            testcase : a dictionary of input test arguments
            ispath   : default to True. If True, method is expected to return a filepath, otherwise method is expected
                       to return data.

        Returns:
            None.
        """
        for m in methods:
            with self.subTest(f"{m[0]}", m=m):
                print(m[0])
                t = testcase.copy()
                if m[1]:
                    t["datadir"] = datadir
                f = getattr(sp, m[0])
                # if m[0] in self.location_ignoreArgs:
                #     t.pop(self.location_ignoreArgs[m[0]])
                # try:
                # make a functional call with test arguments
                result = f(**t)
                print(result)
                # method should return path or data but should not return None
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

    def get_methods_to_test(self, exclude_pattern=None,
                            keywords=None,
                            target_args=None,
                            exclude_args=None):
        """
        Retrieve the methods from the live synthpop object with keywords in their names.

        Args:
            exclude_pattern : pattern to be excluded in the method name
            keywords        : used to search for matching method names
            target_args     : used to identify the methods of which the arguments match
            exclude_args    : used to exclude methods with certain arguments

        Returns:
            list: List of methods matching the search criteria.
        """

        # setting defaults
        exclude_pattern = self.location_ignoreArgs if exclude_pattern is None \
            else self.location_ignoreArgs + exclude_pattern
        keywords = ["path"] if keywords is None else keywords
        target_args = ['country_location', 'state_location', 'location'] if target_args is None else target_args
        exclude_args = [] if exclude_pattern is None else exclude_pattern

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
            matched = ([x in target_args for x in args].count(True) == len(target_args))
            excluded = [x in exclude_args for x in args].count(True) > 0
            if matched and not excluded:
                if len(exclude_args) > 0 and [x in exclude_args for x in args].count(True) > 0:
                    continue
                # print(m[0])
                if "datadir" in args:
                    path_methods.append((m[0], True))
                else:
                    path_methods.append((m[0], False))
        return path_methods


# Run unit tests if called as a script
if __name__ == '__main__':
    unittest.main()
