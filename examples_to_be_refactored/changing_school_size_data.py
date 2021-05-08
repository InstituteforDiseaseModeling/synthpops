"""
How to change school size data for seattle_metro in SynthPops.
"""

import sciris as sc
import synthpops as sp
import json
import os

pars = dict(
    datadir          = sp.datadir,
    country_location = 'usa',
    state_location   = 'Washington',
    location         = 'seattle_metro'
    # location = 'Spokane_County'
    )


def write_school_size_distr_by_type(datadir, location_alias, location, state_location, country_location, school_size_distr_by_type):
    """"""
    file_path = os.path.join(datadir, country_location, state_location, location, 'schools')
    os.makedirs(file_path, exist_ok=True)
    file_name = os.path.join(file_path, f'{location_alias}_school_size_distribution_by_type.dat')
    with open(file_name, 'w') as f:
        json.dump(school_size_distr_by_type, f)
    f.close()


def split_elementary_school_types_and_save(do_save=False):
    """"""
    school_size_distr_by_type = sp.get_school_size_distr_by_type(**pars)
    school_types = sorted(school_size_distr_by_type.keys())

    # splitting up pk and es school size distributions if they're combined
    if 'pk-es' in school_types:

        new_distr = school_size_distr_by_type.copy()
        new_distr['pk'] = school_size_distr_by_type['pk-es'].copy()
        new_distr['es'] = school_size_distr_by_type['pk-es'].copy()

        new_distr.pop('pk-es', None)

        if do_save:
            # save a copy of the old distribution to another file
            location_alias_original = f"{pars['location']}_original"
            write_school_size_distr_by_type(pars['datadir'],
                                            location_alias=location_alias_original,
                                            location=pars['location'],
                                            state_location=pars['state_location'],
                                            country_location=pars['country_location'],
                                            school_size_distr_by_type=school_size_distr_by_type)

            # save a copy of the new distribution with pk-es split
            location_alias_new = f"{pars['location']}_split"
            write_school_size_distr_by_type(pars['datadir'],
                                            location_alias=location_alias_new,
                                            location=pars['location'],
                                            state_location=pars['state_location'],
                                            country_location=pars['country_location'],
                                            school_size_distr_by_type=new_distr)


if __name__ == '__main__':

    do_save = True  # set this to True if you want to save results

    split_elementary_school_types_and_save(do_save)

    print(f"\nNote:\n\tManual steps are required to complete the switching of files.\n")

    print(f"\t1. Go to {os.path.join(sp.datadir, pars['country_location'], pars['state_location'], pars['location'])}")
    print("\t2. Copy whichever version of the school size distribution by school types ('original' or 'split') to the file named as follows:")

    print(f"\n\t{pars['location']}_school_size_distr_by_type.dat")
    print(f"\n\tIf you want to work with school types 'pk' and 'es' separated, then choose the version with 'split' in the file name.")

    print(f"\n\tDo similar for {pars['location']}_school_types_by_age_range.dat files, switching between 'original' and 'split' depending on whether you want 'pk' and 'es' together or split.")