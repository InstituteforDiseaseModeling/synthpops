"""
This module plots the age-specific contact matrix in different settings.
"""
import os
import sciris as sc
import numpy as np
import covasim as cv
import matplotlib as mplt
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import Counter
import cmasher as cmr
import cmocean as cmo
from . import config as cfg
from . import base as spb
from . import data_distributions as spdata
from . import schools as spsch
from . import pop as sppop

# Pretty fonts  --- move this into a class that gets called instead of directly being set
try:
    fontstyle = 'Roboto Condensed'
    mplt.rcParams['font.family'] = fontstyle
except:
    mplt.rcParams['font.family'] = 'Roboto'
mplt.rcParams['font.size'] = 16

try:
    import cmasher
    default_colormap = 'cmr.freeze_r'
except:
    default_colormap = 'bone_r'
    print(f"Note: cmasher import failed; defaulting to regular colormap: {default_colormap}")


__all__ = ['calculate_contact_matrix', 'plot_contacts', 'plot_age_comparison',
           'plot_school_sizes_by_type']  # defines what will be * imported from synthpops, eveything else will need to be imported as synthpops.plotting.method_a, etc.


class plotting_kwargs(sc.prettyobj):
    """
    A class to set and operate on plotting kwargs throughout synthpops.

    Args:
        kwargs (dict)
    """


def default_plotting_kwargs():
    """Define default plotting kwargs to be used in plotting methods."""

    default_kwargs = sc.objdict()
    default_kwargs.fontfamily = 'Roboto Condensed'
    default_kwargs.fontsize = 12
    default_kwargs.format = 'png'
    default_kwargs.rotation = 0
    default_kwargs.subplot_height = 5
    default_kwargs.subplot_width = 8
    default_kwargs.hspace = 0.4
    default_kwargs.wspace = 0.3
    default_kwargs.nrows = 1
    default_kwargs.ncols = 1
    default_kwargs.height = default_kwargs.nrows * default_kwargs.subplot_height
    default_kwargs.width = default_kwargs.ncols * default_kwargs.subplot_width
    default_kwargs.show = 1
    default_kwargs.cmap = 'cmr.freeze_r'
    default_kwargs.markersize = 6
    # default_kwargs.dpi = int(os.getenv('COVASIM_DPI', plt.rcParams['figure.dpi']))
    default_kwargs.display_dpi = int(os.getenv('SYNTHPOPS_DPI', plt.rcParams['figure.dpi']))
    default_kwargs.save_dpi = 300
    default_kwargs.screen_width = 1366
    default_kwargs.screen_height = 768
    default_kwargs.screen_height_factor = 0.85
    default_kwargs.screen_width_factor = 0.3
    default_kwargs.do_show = False
    default_kwargs.do_save = False
    default_kwargs.figdir = None
    # default_kwargs.figdir = cfg.datadir.replace('data', 'figures')
    print(default_kwargs.display_dpi)
    return default_kwargs


def set_figure_display_size(kwargs, nrows):
    """
    Update plotting kwargs with display sizes.

    Args:
        kwargs (sc.objdict): plotting kwargs
        nrows (int): number of subplots stacked vertically

    Return:
        Updated plotting kwargs.
    """
    screen_height_factor = kwargs.screen_height_factor / nrows
    kwargs.display_height = np.round((kwargs.screen_height * screen_height_factor) * nrows / kwargs.display_dpi, 2)
    kwargs.display_width = np.round((kwargs.screen_width * kwargs.screen_width_factor) / kwargs.display_dpi, 2)

    return kwargs


def calculate_contact_matrix(population, density_or_frequency='density', setting_code='H'):
    """
    Calculate the symmetric age-specific contact matrix from the connections
    for all people in the population. density_or_frequency sets the type of
    contact matrix calculated.

    When density_or_frequency is set to 'frequency' each person is assumed to
    have a fixed amount of contact with others they are connected to in a
    setting so each person will split their contact amount equally among their
    connections. This means that if a person has links to 4 other individuals
    then 1/4 will be added to the matrix element matrix[age_i][age_j] for each
    link, where age_i is the age of the individual and age_j is the age of the
    contact. This follows the mass action principle such that increased density
    or number of people a person is in contact with leads to decreased per-link
    or connection contact rate.

    When density_or_frequency is set to 'density' the amount of contact each
    person has with others scales with the number of people they are connected
    to. This means that a person with connections to 4 individuals has a higher
    total contact rate than a person with connection to 3 individuals. For this
    definition if a person has links to 4 other individuals then 1 will be
    added to the matrix element matrix[age_i][age_j] for each contact. This
    follows the 'pseudo'mass action principle such that the per-link or
    connection contact rate is constant.

    Args:
        population (dict)          : A dictionary of a population with attributes.
        density_or_frequency (str) : option for the type of contact matrix calculated.
        setting_code (str)         : name of the physial contact setting: H for households, S for schools, W for workplaces, C for community or other, and 'lTCF' for long term care facilities

    Returns:
        Symmetric age specific contact matrix.

    """
    uids = population.keys()
    uids = [uid for uid in uids]

    num_ages = 101

    M = np.zeros((num_ages, num_ages))

    for n, uid in enumerate(uids):
        age = population[uid]['age']
        contact_ages = [population[c]['age'] for c in population[uid]['contacts'][setting_code]]
        contact_ages = np.array([int(a) for a in contact_ages])

        if len(contact_ages) > 0:
            if density_or_frequency == 'frequency':
                for ca in contact_ages:
                    M[age, ca] += 1.0 / len(contact_ages)
            elif density_or_frequency == 'density':
                for ca in contact_ages:
                    M[age, ca] += 1.0
    return M


def plot_contact_matrix(matrix, age_count, aggregate_age_count, age_brackets, age_by_brackets_dic,
                        setting_code='H', density_or_frequency='density', logcolors_flag=False,
                        aggregate_flag=True, cmap=default_colormap, fontsize=16, rotation=50,
                        title_prefix=None, fig=None, ax=None):
    """
    Plots the age specific contact matrix where the matrix element matrix_ij is the contact rate or frequency
    for the average individual in age group i with all of their contacts in age group j. Can either be density
    or frequency definition, as well as a single year age contact matrix or a contact matrix for aggregated
    age brackets.

    Args:
        matrix (np.array)                : symmetric contact matrix, element ij is the contact for an average individual in age group i with all of their contacts in age group j
        age_count (dict)                 : dictionary with the count of individuals in the population for each age
        aggregate_age_count (dict)       : dictionary with the count of individuals in the population in each age bracket
        age_brackets (dict)              : dictionary mapping age bracket keys to age bracket range
        age_by_brackets_dic (dict)       : dictionary mapping age to the age bracket range it falls in
        setting_code (str)               : name of the physial contact setting: H for households, S for schools, W for workplaces, C for community or other
        density_or_frequency (str)       : If 'density', then each contact counts for 1/(group size -1) of a person's contact in a group, elif 'frequency' then count each contact. This means that more people in a group leads to higher rates of contact/exposure.
        logcolors_flag (bool)            : If True, plot heatmap in logscale
        aggregate_flag (bool)            : If True, plot the contact matrix for aggregate age brackets, else single year age contact matrix.
        cmap(str or matplotlib colormap) : colormap
        fontsize (int)                   : base font size
        rotation (int)                   : rotation for x axis labels
        title_prefix(str)                : optional title prefix for the figure
        fig (Figure)                     : if supplied, use this figure instead of generating one
        ax (Axes)                        : if supplied, use these axes instead of generating one

    Returns:
        A fig object.

    Note:
        For the long term care facilities you may want the age count and the aggregate age count to only consider those who live or work in long term care facilities because otherwise this will be the whole population wide average mixing in that setting

    """
    cmap = mplt.cm.get_cmap(cmap)

    if fig is None:
        fig = plt.figure(figsize=(10, 10), tight_layout=True)
    if ax is None:
        ax = [fig.add_subplot(1, 1, 1)]
    else:
        ax = [ax]
    cax = []
    cbar = []
    implot = []

    titles = {'H': 'Household', 'S': 'School', 'W': 'Work', 'LTCF': 'Long Term Care Facilities'}

    if aggregate_flag:
        aggregate_M = spb.get_aggregate_matrix(matrix, age_by_brackets_dic)
        asymmetric_M = spb.get_asymmetric_matrix(aggregate_M, aggregate_age_count)
    else:
        asymmetric_M = spb.get_asymmetric_matrix(matrix, age_count)

    if logcolors_flag:

        vbounds = {}
        if density_or_frequency == 'frequency':
            if aggregate_flag:
                vbounds['H'] = {'vmin': 1e-2, 'vmax': 1e-0}
                vbounds['S'] = {'vmin': 1e-3, 'vmax': 1e-0}
                vbounds['W'] = {'vmin': 1e-3, 'vmax': 1e-0}
                vbounds['LTCF'] = {'vmin': 1e-3, 'vmax': 1e-1}
            else:
                vbounds['H'] = {'vmin': 1e-3, 'vmax': 1e-1}
                vbounds['S'] = {'vmin': 1e-3, 'vmax': 1e-1}
                vbounds['W'] = {'vmin': 1e-3, 'vmax': 1e-1}
                vbounds['LTCF'] = {'vmin': 1e-3, 'vmax': 1e-0}

        elif density_or_frequency == 'density':
            if aggregate_flag:
                vbounds['H'] = {'vmin': 1e-2, 'vmax': 1e0}
                vbounds['S'] = {'vmin': 1e-2, 'vmax': 1e1}
                vbounds['W'] = {'vmin': 1e-2, 'vmax': 1e1}
                vbounds['LTCF'] = {'vmin': 1e-3, 'vmax': 1e-0}

            else:
                vbounds['H'] = {'vmin': 1e-2, 'vmax': 1e0}
                vbounds['S'] = {'vmin': 1e-2, 'vmax': 1e0}
                vbounds['W'] = {'vmin': 1e-2, 'vmax': 1e0}
                vbounds['LTCF'] = {'vmin': 1e-2, 'vmax': 1e-0}

        im = ax[0].imshow(asymmetric_M.T, origin='lower', interpolation='nearest', cmap=cmap, norm=LogNorm(vmin=vbounds[setting_code]['vmin'], vmax=vbounds[setting_code]['vmax']))

    else:

        im = ax[0].imshow(asymmetric_M.T, origin='lower', interpolation='nearest', cmap=cmap)

    implot.append(im)

    if fontsize > 20:
        rotation = 90

    for i in range(len(ax)):
        divider = make_axes_locatable(ax[i])
        cax.append(divider.new_horizontal(size="4%", pad=0.15))

        fig.add_axes(cax[i])
        cbar.append(fig.colorbar(implot[i], cax=cax[i]))
        cbar[i].ax.tick_params(axis='y', labelsize=fontsize + 4)
        if density_or_frequency == 'frequency':
            cbar[i].ax.set_ylabel('Frequency of Contacts', fontsize=fontsize + 2)
        else:
            cbar[i].ax.set_ylabel('Density of Contacts', fontsize=fontsize + 2)
        ax[i].tick_params(labelsize=fontsize + 2)
        ax[i].set_xlabel('Age', fontsize=fontsize + 6)
        ax[i].set_ylabel('Age of Contacts', fontsize=fontsize + 6)
        # ax[i].set_title(titles[setting_code] + ' Contact Patterns', fontsize=fontsize + 10)
        ax[i].set_title(
            (title_prefix if title_prefix is not None else '') + titles[setting_code] + ' Age Mixing', fontsize=fontsize + 10)

        if aggregate_flag:
            tick_labels = [str(age_brackets[b][0]) + '-' + str(age_brackets[b][-1]) for b in age_brackets]
            ax[i].set_xticks(np.arange(len(tick_labels)))
            ax[i].set_xticklabels(tick_labels, fontsize=fontsize)
            ax[i].set_xticklabels(tick_labels, fontsize=fontsize, rotation=rotation)
            ax[i].set_yticks(np.arange(len(tick_labels)))
            ax[i].set_yticklabels(tick_labels, fontsize=fontsize)
        else:
            ax[i].set_xticks(np.arange(0, len(age_count) + 1, 10))
            ax[i].set_yticks(np.arange(0, len(age_count) + 1, 10))

    return fig


def plot_contacts(population,
                  setting_code='H',
                  aggregate_flag=True,
                  logcolors_flag=True,
                  density_or_frequency='density',
                  cmap=default_colormap,
                  fontsize=16,
                  rotation=50,
                  title_prefix=None,
                  fig=None,
                  ax=None,
                  do_show=True):
    """
    Plot the age mixing matrix for a specific setting.

    TODO: rename setting_code to layer

    Args:
        population(dict)                 : population to be plotted, if None, code will generate it
        setting_code (str)               : name of the physial contact setting: H for households, S for schools, W for workplaces, C for community or other
        aggregate_flag (bool)            : If True, plot the contact matrix for aggregate age brackets, else single year age contact matrix.
        logcolors_flag (bool)            : If True, plot heatmap in logscale
        density_or_frequency (str)       : If 'density', then each contact counts for 1/(group size -1) of a person's contact in a group, elif 'frequency' then count each contact. This means that more people in a group leads to higher rates of contact/exposure.
        cmap(str or matplotlib colormap) : colormap
        fontsize (int)                   : base font size
        rotation (int)                   : rotation for x axis labels
        title_prefix(str)                : optional title prefix for the figure
        fig (Figure)                     : if supplied, use this figure instead of generating one
        ax (Axes)                        : if supplied, use these axes instead of generating one
        do_show (bool)                   : whether to show the plot

    Returns:
        A fig object.

    """
    datadir = cfg.datadir

    state_location = 'Washington'
    country_location = 'usa'

    age_brackets = spdata.get_census_age_brackets(datadir, state_location=state_location, country_location=country_location)
    age_by_brackets_dic = spb.get_age_by_brackets_dic(age_brackets)

    ages = []
    # if setting_code == 'LTCF':
    #     ltcf_ages = []

    for uid in population:
        ages.append(population[uid]['age'])
        # if setting_code == 'LTCF':
        #     if population[uid]['snf_res'] or population[uid]['snf_staff']:
        #         ltcf_ages.append(population[uid]['age'])

    age_count = Counter(ages)
    aggregate_age_count = spb.get_aggregate_ages(age_count, age_by_brackets_dic)

    # if setting_code == 'LTCF':
    #     ltcf_age_count = Counter(ltcf_ages)
    #     aggregate_ltcf_age_count = sp.get_aggregate_ages(ltcf_age_count, age_by_brackets_dic)

    matrix = calculate_contact_matrix(population, density_or_frequency, setting_code)

    # if setting_code == 'LTCF':
    #     fig = sp.plot_contact_frequency(matrix, ltcf_age_count, aggregate_ltcf_age_count, age_brackets, age_by_brackets_dic,
    #                                     setting_code, density_or_frequency, logcolors_flag, aggregate_flag, cmap, fontsize, rotation)
    # else:
    #     fig = sp.plot_contact_frequency(matrix, age_count, aggregate_age_count, age_brackets, age_by_brackets_dic,
    #                                     setting_code, density_or_frequency, logcolors_flag, aggregate_flag, cmap, fontsize, rotation)

    fig = plot_contact_matrix(matrix, age_count, aggregate_age_count, age_brackets, age_by_brackets_dic,
                              setting_code, density_or_frequency, logcolors_flag, aggregate_flag, cmap, fontsize, rotation, title_prefix,
                              fig=fig, ax=ax)

    if do_show:
        plt.show()

    return fig


def plot_array(expected,
               generated=None,
               names=None,
               figdir=None,
               prefix="test",
               do_show=True,
               do_save=False,
               expect_label='Expected',
               value_text=False,
               xlabels=None,
               xlabel_rotation=0,
               binned=True,
               fig=None,
               ax=None,
               color_1=None,
               color_2=None,
               markersize=6,
               fontsize=12):
    """
    Plot histogram on a sorted array based by names. If names not provided the
    order will be used. If generate data is not provided, plot only the expected values.
    Note this can only be used with the limitation that data that has already been binned

    Args:
        expected        : Array of expected values
        generated       : Array of values generated using a model
        names           : names to display on x-axis, default is set to the indexes of data
        figdir          : directory to save the plot if provided
        testprefix      : used to prefix the title of the plot
        do_close        : close the plot immediately if set to True
        expect_label    : Label to show in the plot, default to "expected"
        value_text      : display the values on top of the bar if specified
        xlabel_rotation : rotation degree for x labels if specified
        binned          : default to True, if False, it will just plot a simple histogram for expected data

    Returns:
        None.

    Plot will be saved in datadir if given
    """
    if fig is None:
        fig, ax = plt.subplots(1, 1)

    mplt.rcParams['font.family'] = 'Roboto Condensed'
    if color_1 is None:
        color_1 = 'mediumseagreen'
    if color_2 is None:
        color_2 = '#236a54'

    title = prefix.replace('_', ' ').title() if generated is None else f"{prefix.replace('_', ' ').title()} Comparison"
    ax.set_title(title, fontsize=fontsize + 2)
    x = np.arange(len(expected))

    if not binned:
        ax.hist(expected, label=expect_label.title(), color='mediumseagreen')
    else:
        rect1 = ax.bar(x, expected, label=expect_label.title(), color=color_1, zorder=0)
        if generated is not None:
            line, = ax.plot(x, generated, color=color_2, markeredgecolor='white', marker='o', markersize=markersize, label='Generated', zorder=1)
        if value_text:
            autolabel(ax, rect1, 0, 5)
            if generated is not None:
                for j, v in enumerate(generated):
                    ax.text(j, v, str(round(v, 3)), fontsize=10, horizontalalignment='right', verticalalignment='top', color='#3f75a2')
        if names is not None:
            if isinstance(names, dict):
                xticks = sorted(names.keys())
                xticklabels = [names[k] for k in xticks]
            else:
                xticks = np.arange(len(names))
                xticklabels = names
            # if there are too many labels, only show every 10 ticks
            if len(names) > 30:
                xticks = xticks[0::10]
                xticklabels = xticklabels[0::10]
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels, rotation=xlabel_rotation)

    leg = ax.legend(loc='upper right', fontsize=fontsize)
    leg.draw_frame(False)
    ax.set_xlim(x[0] - 1, x[-1] + 1)

    if do_save:
        if figdir is None:
            datadir = cfg.datadir
            figdir = datadir.replace('data', 'figures')
        os.makedirs(figdir, exist_ok=True)
        plt.savefig(os.path.join(figdir, f"{prefix}.png".replace('\n', '_')), format="png")
    if do_show:
        plt.show()
    return fig, ax


def autolabel(ax, rects, h_offset=0, v_offset=0.3):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    Args:
        ax       : matplotlib.axes figure object
        rects    : matplotlib.container.BarContainer
        h_offset : The position x to place the text at.
        v_offset : The position y to place the text at.

    Returns:
        None.

    Set the annotation according to the input parameters
    """
    for rect in rects:
        height = rect.get_height()
        text = ax.annotate('{}'.format(round(height, 3)),
                           xy=(rect.get_x() + rect.get_width() / 2, height),
                           xytext=(h_offset, v_offset),
                           textcoords="offset points",
                           ha='center', va='bottom')
        text.set_fontsize(10)


def plot_age_comparison(pop, *args, **kwargs):
    """
    Plot a comparison of the expected and generated age distribution.

    Args:
        pop (pop object): population, either synthpops.pop.Pop, covasim.people.People, or dict

    Returns:
        Matplotlib figure and ax.

    Note:
        If using pop with type covasim.people.Pop or dict, args must be supplied
        for the location parameters to get the expected distribution.

    **Example**::

        pars = {'n': 10e3, location='seattle_metro', state_location='Washington', country_location='usa'}
        pop = sp.Pop(**pars)
        fig, ax = pop.plot_age_distribution_comparison()

        popdict = pop.to_dict()
        kwargs = pars.copy()
        kwargs['datadir'] = sp.datadir
        fig, ax = sp.plot_age_distribution_comparison(popdict, **kwargs)
    """
    default_kwargs = default_plotting_kwargs()

    default_kwargs = sc.mergedicts(default_kwargs, dict(color_1='#55afe1', color_2='#0a6299',
                                   left=0.10, right=0.95, top=0.90, bottom=0.10, figname=f"age_distribution_comparison"))

    kwargs = sc.mergedicts(default_kwargs, kwargs)
    kwargs = sc.objdict(kwargs)
    kwargs.axis = sc.objdict({'left': kwargs.left, 'right': kwargs.right,
                              'top': kwargs.top, 'bottom': kwargs.bottom,
                              'hspace': kwargs.hspace, 'wspace': kwargs.wspace}
                             )

    # supporting three pop object types: synthpops.pop.Pop class, covasim.people.People class, and dictionaries (generated from or in the style of synthpops.pop.Pop.to_dict())

    # set up location parameters to grab expected data
    if isinstance(pop, (dict, cv.people.People)):
        loc_pars = sc.objdict({'datadir': kwargs.datadir, 'location': kwargs.location, 'state_location': kwargs.state_location,
                               'country_location': kwargs.country_location, 'use_default': kwargs.use_default})

    elif isinstance(pop, sppop.Pop):
        loc_pars = sc.objdict({'datadir': cfg.datadir, 'location': pop.location, 'state_location': pop.state_location,
                               'country_location': pop.country_location, 'use_default': pop.use_default})
        kwargs.smooth_ages = pop.smooth_ages
        kwargs.window_length = pop.window_length

    else:
        raise ValueError(f"This method does not support pop objects with the type {type(pop)}. Please look at the notes and try another supported pop type.")

    if not kwargs.smooth_ages:
        kwargs.window_length = 1

    # get the expected age distribution
    expected_age_distr = spdata.get_smoothed_single_year_age_distr(**sc.mergedicts(loc_pars, {'window_length': kwargs.window_length}))
    generated_age_count = dict.fromkeys(expected_age_distr.keys(), 0)  # sets ordering of keys consistently

    # get the generated age distribution
    if isinstance(pop, sppop.Pop):
        for i, person in pop.popdict.items():
            generated_age_count[person['age']] += 1

    elif isinstance(pop, dict):
        for i, person in pop.items():
            generated_age_count[person['age']] += 1

    elif isinstance(pop, cv.people.People):
        generated_age_count = Counter(pop.age)

    generated_age_distr = spb.norm_dic(generated_age_count)

    expected_age_distr_array = [expected_age_distr[k] * 100 for k in sorted(expected_age_distr.keys())]
    generated_age_distr_array = [generated_age_distr[k] * 100 for k in sorted(generated_age_distr.keys())]

    # update the fig
    fig, ax = plt.subplots(1, 1, figsize=(kwargs.width, kwargs.height), dpi=kwargs.display_dpi)
    fig.subplots_adjust(**kwargs.axis)

    fig, ax = plot_array(expected_age_distr_array, generated_age_distr_array,
                         do_show=False, xlabel_rotation=kwargs.rotation,
                         prefix=f"{loc_pars.location}_age_distribution", binned=True,
                         fig=fig, ax=ax, color_1=kwargs.color_1, color_2=kwargs.color_2, markersize=kwargs.markersize)
    ax.set_xlabel('Age', fontsize=kwargs.fontsize)
    ax.set_ylabel('Distribution (%)', fontsize=kwargs.fontsize)
    ax.set_xlim(-1., len(expected_age_distr_array) + 0.)
    max_y = np.ceil(max(max(expected_age_distr_array), max(generated_age_distr_array)))
    ax.set_ylim(0, max_y)

    ax.tick_params(labelsize=kwargs.fontsize)

    if kwargs.do_save:
        if kwargs.figdir is not None:
            os.makedirs(kwargs.figdir, exist_ok=True)
            figpath = os.path.join(kwargs.figdir, f"{kwargs.figname}.{kwargs.format}")
        else:
            figpath = f"{kwargs.figname}.{kwargs.format}"
        fig.savefig(figpath, format=kwargs.format, dpi=kwargs.save_dpi)

    if kwargs.do_show:
        plt.show()

    return fig, ax


def plot_school_sizes_by_type(pop, *args, **kwargs):
    """
    Plot a comparison of the expected and generated school size distribution for each type of school expected.

    Args:
        pop (pop object): population, either synthpops.pop.Pop, covasim.people.People, or dict

    Returns:
        Matplotlib figure and ax.

    Note:
        If using pop with type covasim.people.Pop or dict, args must be supplied
        for the location parameters to get the expected distribution.

    **Example**::

        pars = {'n': 10e3, location='seattle_metro', state_location='Washington', country_location='usa'}
        pop = sp.Pop(**pars)
        fig, ax = pop.plot_school_sizes_by_type()

        popdict = pop.to_dict()
        kwargs = pars.copy()
        kwargs['datadir'] = sp.datadir
        fig, ax = sp.plot_school_sizes_by_type(popdict, **kwargs)
    """
    default_kwargs = default_plotting_kwargs()
    default_kwargs.school_type_labels = spsch.get_school_type_labels()

    # TODO: consolidate the following method specific default parameters in dictionary
    default_kwargs = sc.mergedicts(default_kwargs, dict(left=0.11, right=0.94,
                                   top=0.96, bottom=0.08, hspace=0.75,
                                   subplot_height=2.8, subplot_width=4.2,
                                   screen_height_factor=0.17, fontsize=8,
                                   rotation=20,
                                   cmap='cmo.curl', figname=f"school_size_distribution_by_type",
                                   keys_to_exclude=['uv'],
                                   location_text_y=117,
                                   )
                                   )

    kwargs = sc.mergedicts(default_kwargs, kwargs)
    kwargs = sc.objdict(kwargs)
    kwargs.axis = sc.objdict({'left': kwargs.left, 'right': kwargs.right, 'top': kwargs.top,
                              'bottom': kwargs.bottom, 'hspace': kwargs.hspace, 'wspace': kwargs.wspace})

    # supporting three pop object types: synthpops.pop.Pop class, covasim.people.People class, and dictionaries (generated from or in the style of synthpops.pop.Pop.to_dict())

    # set up location parameters to grab expected data
    if isinstance(pop, (dict, cv.people.People)):
        loc_pars = sc.objdict({'datadir': kwargs.datadir, 'location': kwargs.location, 'state_location': kwargs.state_location,
                               'country_location': kwargs.country_location, 'use_default': kwargs.use_default})
        if isinstance(pop, dict):
            popdict = sc.dcp(pop)
        else:
            raise NotImplementedError('This method is not yet implemented for covasim people objects.')

    elif isinstance(pop, sppop.Pop):
        loc_pars = sc.objdict({'datadir': cfg.datadir, 'location': pop.location, 'state_location': pop.state_location,
                               'country_location': pop.country_location, 'use_default': pop.use_default})
        kwargs.with_school_types = pop.school_pars.with_school_types
        popdict = sc.dcp(pop.popdict)

    else:
        raise ValueError(f"This method does not support pop objects with the type {type(pop)}. Please look at the notes and try another supported pop type.")

    kwargs = sc.objdict(sc.mergedicts(kwargs, loc_pars))  # add all location keys to this as well

    if kwargs.with_school_types:
        expected_school_size_distr = spdata.get_school_size_distr_by_type(**loc_pars)
    else:
        expected_school_size_distr = {None: spdata.get_school_size_distr_by_brackets(**loc_pars)}
    school_size_brackets = spdata.get_school_size_brackets(**loc_pars)

    bins = [school_size_brackets[0][0]] + [school_size_brackets[b][-1] + 1 for b in school_size_brackets]
    bin_labels = [f"{school_size_brackets[b][0]}-{school_size_brackets[b][-1]}" for b in school_size_brackets]

    # calculate how many students are in each school

    generated_school_size_distr = dict()
    if isinstance(pop, (sppop.Pop, dict)):
        enrollment_by_school_type = spsch.get_enrollment_by_school_type(popdict)

    for sc_type in enrollment_by_school_type:
        sizes = enrollment_by_school_type[sc_type]
        hist, bins = np.histogram(sizes, bins=bins, density=0)
        generated_school_size_distr[sc_type] = {i: hist[i] / sum(hist) for i in school_size_brackets}

    generated_school_size_distr = sc.objdict(generated_school_size_distr)

    for school_type in kwargs.keys_to_exclude:
        enrollment_by_school_type.pop(school_type, None)
        generated_school_size_distr.pop(school_type, None)
        kwargs.school_type_labels.pop(school_type, None)

    sorted_school_types = sorted(generated_school_size_distr.keys())
    n_school_types = len(sorted_school_types)

    # location text
    if kwargs.location is not None:
        location_text = f"{kwargs.location.replace('_', ' ').title()}"
    else:
        location_text = f"{cfg.default_location.replace('_', ' ').title()}"

    # cmap
    if kwargs.cmap == 'cmo.curl':
        cmap = cmr.get_sub_cmap(kwargs.cmap, 0.08, 1)  # remove --- users can supply their own cmaps however modified
    else:
        cmap = mplt.cm.get_cmap(kwargs.cmap)

    # set up screen size
    kwargs.display_height = np.round((kwargs.screen_height * kwargs.screen_height_factor) * n_school_types / kwargs.display_dpi, 2)
    kwargs.display_width = np.round((kwargs.screen_width * kwargs.screen_width_factor) / kwargs.display_dpi, 2)

    # create fig, ax
    fig, ax = plt.subplots(n_school_types, 1, figsize=(kwargs.display_width, kwargs.display_height), dpi=kwargs.display_dpi)

    # readjust figure parameters
    if n_school_types == 1:
        ax = [ax]
        fig.set_size_inches(kwargs.display_width, kwargs.display_height * 1.7)
        kwargs.axis = sc.objdict(sc.mergedicts(kwargs.axis, {'top': 0.88, 'bottom': 0.18, 'left': 0.12}))
        kwargs.location_text_y = 108

    # update the fig
    fig.subplots_adjust(**kwargs.axis)

    # for ns, school_type in enumerate(sorted_school_types):
    for ns, school_type in enumerate(kwargs.school_type_labels.keys()):
        x = np.arange(len(school_size_brackets))  # potentially will use different bins for each school type so placeholder for now
        c = ns / n_school_types
        c2 = min(c + 0.1, 1)

        sorted_bins = sorted(expected_school_size_distr[school_type].keys())

        ax[ns].bar(x, [expected_school_size_distr[school_type][b] * 100 for b in sorted_bins], color=cmap(c), edgecolor='white', label='Expected', zorder=0)
        ax[ns].plot(x, [generated_school_size_distr[school_type][b] * 100 for b in sorted_bins], color=cmap(c2), ls='--',
                    marker='o', markerfacecolor=cmap(c2), markeredgecolor='white', markeredgewidth=.5, markersize=5, label='Generated', zorder=1)

        leg = ax[ns].legend(loc=1, fontsize=kwargs.fontsize)
        leg.draw_frame(False)
        ax[ns].set_xticks(x)
        ax[ns].set_xticklabels(bin_labels, rotation=kwargs.rotation, fontsize=kwargs.fontsize,
                               verticalalignment='center_baseline')
        ax[ns].set_xlim(0, x[-1])
        ax[ns].set_ylim(0, 100)
        ax[ns].set_ylabel('%', fontsize=kwargs.fontsize + 1)
        ax[ns].tick_params(labelsize=kwargs.fontsize - 1)
        if school_type is None:
            title = "without school types defined"
        else:
            title = f"{kwargs.school_type_labels[school_type]}"
        if ns == 0:
            ax[ns].text(0., kwargs.location_text_y, location_text, horizontalalignment='left', fontsize=kwargs.fontsize + 1, verticalalignment='top')
        ax[ns].set_title(title, fontsize=kwargs.fontsize + 1, verticalalignment='top')
    ax[ns].set_xlabel('School size', fontsize=kwargs.fontsize + 1, verticalalignment='center_baseline')
    # ax[ns].set_xlabel('School size', fontsize=kwargs.fontsize + 1, verticalalignment='center_baseline')

    if kwargs.do_show:
        plt.show()

    # update fig before saving to disk since display will modify things
    if kwargs.do_save:
        if len(ax) == 1:
            fig.set_size_inches(kwargs.display_width, kwargs.display_height * 1.7)  # too brittle

        else:
            fig.set_size_inches(kwargs.display_width, kwargs.display_height)
            kwargs.axis = sc.objdict(sc.mergedicts(kwargs.axis, {'bottom': 0.075, 'hspace': 0.52, 'left': 0.12}))  # too brittle --- can supply defaults that look nice and that's about it

        fig.subplots_adjust(**kwargs.axis)
        if kwargs.figdir is not None:
            os.makedirs(kwargs.figdir, exist_ok=True)
            figpath = os.path.join(kwargs.figdir, f"{kwargs.figname}.{kwargs.format}")
        else:
            figpath = f"{kwargs.figname}.{kwargs.format}"
        fig.savefig(figpath, format=kwargs.format, dpi=kwargs.save_dpi)

    return fig, ax
