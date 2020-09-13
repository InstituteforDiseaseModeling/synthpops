import synthpops as sp
import sciris as sc
import pylab as pl

from examples.plot_age_mixing_matrices import test_plot_generated_contact_matrix as plotmatrix # WARNING, refactor

do_save = 0
stride = 10 # Keep one out of every this many people

sp.logger.setLevel('DEBUG')

pars = dict(
    n                               = 10001,
    rand_seed                       = 123,
    max_contacts                    = None,
    generate                        = True,

    with_industry_code              = 0,
    with_facilities                 = 1,
    with_non_teaching_staff         = 1,
    use_two_group_reduction         = 1,
    with_school_types               = 1,

    average_LTCF_degree             = 20,
    ltcf_staff_age_min              = 20,
    ltcf_staff_age_max              = 60,

    school_mixing_type              = 'age_and_class_clustered',
    average_class_size              = 20,
    inter_grade_mixing              = 0.1,
    teacher_age_min                 = 25,
    teacher_age_max                 = 75,
    staff_age_min                   = 20,
    staff_age_max                   = 75,

    average_student_teacher_ratio   = 20,
    average_teacher_teacher_degree  = 3,
    average_student_all_staff_ratio = 15,
    average_additional_staff_degree = 20,
)

if __name__ == '__main__':

    T = sc.tic()
    pop = sp.make_population(**pars)
    elapsed = sc.toc(T, output=True)

    for person in [6, 66, 666]:
        print(f'\n\nPerson {person}')
        sc.pp(pop[person])
    print('\n\n')
    print(sc.gitinfo(sp.__file__))
    print(sp.version.__version__)

    popkeys = list(pop.keys())
    stridekeys = [popkeys[i] for i in range(0, len(pop), stride)]
    subpop = {k:pop[k] for k in stridekeys}

    if do_save:
        sc.savejson(f'pop_v{sp.version.__version__}.json', subpop, indent=2)

    print('\n\n')
    pps = pars["n"]/elapsed
    print(f'Total time: {elapsed:0.3f} s for {pars["n"]} people ({pps:0.0f} people/second)')


    #%% Plotting
    fig, axes = pl.subplots(2,3, figsize=(32,18))
    expected = sc.loadjson('expected/pop_2001_seed1001.json')
    expected = {int(key): val for key, val in expected.items()}
    actual = pop
    for c,code in enumerate(['H', 'W', 'S']):
        args = dict(setting_code=code, density_or_frequency='density')
        fig = plotmatrix(population=expected, title_prefix="Expected ", fig=fig, ax=axes[0,c], **args)
        fig = plotmatrix(population=actual,   title_prefix="Actual",    fig=fig, ax=axes[1,c], **args)