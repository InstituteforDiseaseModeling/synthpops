"""
Make a population using synthpops with school types and mixing types within
schools defined.
"""
import sciris as sc
import synthpops as sp
import matplotlib.pyplot as plt
import cmasher as cmr
import cmocean

pars = dict(
    n                       = 40e3,
    rand_seed               = 123,
    location                = 'Spokane_County',
    state_location          = 'Washington',
    country_location        = 'usa',
    smooth_ages             = 1,
    household_method        = 'fixed_ages',

    with_facilities         = 1,
    with_non_teaching_staff = 1,  # also include non teaching staff
    with_school_types       = 1,
    school_mixing_type      = {'pk': 'random',
                               'es': 'age_and_class_clustered', 
                               'ms': 'age_and_class_clustered', 
                               'hs': 'age_clustered',
                               }
)


pop = sp.Pop(**pars)
kwargs = sc.dcp(pars)
kwargs['cmap'] = cmr.get_sub_cmap('cmo.curl', 0.05, 1)  # let's change the colormap used a little
fig, ax = pop.plot_school_sizes(**kwargs)  # plot school sizes by school type

plt.show()