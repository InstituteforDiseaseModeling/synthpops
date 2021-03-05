"""
This module provides plotting methods including methods to plot the age-specific contact matrix in different contact layers.
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


__all__ = ['calculate_contact_matrix', 'plot_contacts',
           'plot_ages', 'plot_enrollment_rates',
           'plot_school_sizes', 'plotting_kwargs']  # defines what will be * imported from synthpops, eveything else will need to be imported as synthpops.plotting.method_a, etc.


class plotting_kwargs(sc.objdict):
    """
    A class to set and operate on plotting kwargs throughout synthpops.

    Args:
        kwargs (dict): dictionary of plotting parameters to be used.
    """

    def __init__(self, *args, **kwargs):
        """Class constructor for plotting_kwargs."""
        kwargs = sc.mergedicts(self.default_plotting_kwargs(), kwargs)

        self.update(kwargs)
        self.initialize()

        return

    def __repr__(self):
        output = sc.objrepr(self)
        output += sc.objdict.__repr__(self)
        return output

    def initialize(self):
        """Initialize plot settings."""
        self.set_figure_display_size()
        self.set_font()

        return

    def set_font(self, *args, **font):
        """Set font styles."""
        default_font = dict(family=self.fontfamily, style=self.fontstyle,
                            variant=self.fontvariant, weight=self.fontweight,
                            size=self.fontsize
                            )
        font = sc.mergedicts(default_font, font)
        mplt.rc('font', **font)

        return

    def default_plotting_kwargs(self):
        """Define default plotting kwargs to be used in plotting methods."""
        default_kwargs = sc.objdict()
        default_kwargs.fontfamily = 'Roboto Condensed'
        default_kwargs.fontstyle = 'normal'
        default_kwargs.fontvariant = 'normal'
        default_kwargs.fontweight = 400
        default_kwargs.fontsize = 8
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
        default_kwargs.display_dpi = int(os.getenv('SYNTHPOPS_DPI', plt.rcParams['figure.dpi']))
        default_kwargs.save_dpi = 300
        default_kwargs.screen_width = 1366
        default_kwargs.screen_height = 768
        default_kwargs.screen_height_factor = 0.85
        default_kwargs.screen_width_factor = 0.3
        default_kwargs.do_show = False
        default_kwargs.do_save = False
        default_kwargs.figdir = None

        return default_kwargs

    def set_figure_display_size(self, *args, **kwargs):
        """
        Update plotting kwargs with new calculated display sizes.

        Args:
            kwargs (sc.objdict): new values to update with

        Return:
            Updated kwargs and recalculating the display sizes.
        """
        self.update(kwargs)
        self.display_height = np.round(self.screen_height * self.screen_height_factor / self.display_dpi, 2)
        self.display_width = np.round(self.screen_width * self.screen_width_factor / self.display_dpi, 2)

        return

    def set_default_pop_pars(self):
        """
        Check if method has some key pop parameters to call on data. If not, use
        defaults and warn user of their use and value.
        """
        default_pop_pars = sc.objdict(datadir=cfg.datadir, location=cfg.default_location, state_location=cfg.default_state,
                                      country_location=cfg.default_country, use_default=False)
        default_age_pars = sc.objdict(smooth_ages=False, window_length=7)

        # if loc_pars exists, then update the default_pop_pars with that information
        if 'loc_pars' in self:
            default_pop_pars.update(self['loc_pars'])

        # sometimes when not working with a pop object you might be missing location information directly as kwargs and need to use defaults or set the information
        for k in default_pop_pars:
            if k not in self:
                cfg.logger.info(f"kwargs is missing key: {k}. Using the default value: {default_pop_pars[k]}.")
                self[k] = default_pop_pars[k]

        for k in default_age_pars:
            if k not in self:
                cfg.logger.info(f"kwargs is missing key: {k}. Using the default value: {default_age_pars[k]}.")
                self[k] = default_age_pars[k]

        # loc_pars not in self yet
        if 'loc_pars' not in self:
            self['loc_pars'] = sc.objdict({k: self[k] for k in default_pop_pars})

        if not self.smooth_ages:
            self.window_length = 1

        return

    def restore_defaults(self):
        """Reset matplotlib defaults."""
        mplt.rcParams.update(mplt.rcParamsDefault)
        return

    def update_defaults(self, method_defaults, kwargs):
        """Update defaults with method defaults and kwargs."""
        kwargs = sc.objdict(sc.mergedicts(method_defaults, kwargs))
        self.update(kwargs)

        return

    @property
    def axis(self):
        """ Dictionary of axis settings."""
        return sc.objdict({k: self[k] for k in ['left', 'right', 'top', 'bottom', 'hspace', 'wspace']})


def finalize_figure(fig, plkwargs, **new_plkwargs):
    """
    Update any parameters and then return figpath.

    Args:
        fig (matplotlib.Figure)    : figure
        plkwargs (plotting_kwargs) : plotting kwargs class
        new_plkwargs (dict)        : dictionary of new plotting kwargs to update with

    Returns:
        Matplotlib figure.
    """
    plkwargs = sc.dcp(plkwargs)
    plkwargs.update(new_plkwargs)
    if plkwargs.do_save:
        plkwargs.figpath = sc.makefilepath(filename=plkwargs.figname, folder=plkwargs.figdir, ext=plkwargs.format)
        fig.savefig(plkwargs.figpath, format=plkwargs.format, dpi=plkwargs.save_dpi)

    if plkwargs.do_show:
        plt.show()

    return fig


def get_plkwargs(pop):
    """
    Check if pop has plkwargs and return a copy of it. Otherwise, create a new
    instance and return that.

    Args:
        pop (dict or sp.Pop): population object, either a dictionary or a synthpops.pop.Pop object

    Returns:
        plotting_kwargs object
    """
    if isinstance(pop, sppop.Pop):
        if pop.plkwargs is None:
            plkwargs = plotting_kwargs()
            pop.plkwargs = sc.dcp(plkwargs)
        else:
            plkwargs = sc.dcp(pop.plkwargs)  # grab a copy so you don't modify the version pop has
    else:
        plkwargs = plotting_kwargs()

    return plkwargs


def calculate_contact_matrix(population, density_or_frequency='density', layer='H'):
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
        layer (str)                : name of the physial contact setting, see notes.

    Returns:
        np.ndarray: Symmetric age specific contact matrix.

    Note:
        H for households, S for schools, W for workplaces, C for community or
        other, and 'LTCF' for long term care facilities.
    """
    uids = population.keys()
    uids = [uid for uid in uids]

    num_ages = 101

    M = np.zeros((num_ages, num_ages))

    for n, uid in enumerate(uids):
        age = population[uid]['age']
        contact_ages = [population[c]['age'] for c in population[uid]['contacts'][layer]]
        contact_ages = np.array([int(a) for a in contact_ages])

        if len(contact_ages) > 0:
            if density_or_frequency == 'frequency':
                for ca in contact_ages:
                    M[age, ca] += 1.0 / len(contact_ages)
            elif density_or_frequency == 'density':
                for ca in contact_ages:
                    M[age, ca] += 1.0
    return M


def plot_contact_matrix(matrix, age_count, aggregate_age_count, age_brackets, age_by_brackets_dic, **kwargs):
    """
    Plots the age specific contact matrix where the matrix element matrix_ij is
    the contact rate or frequency for the average individual in age group i with
    all of their contacts in age group j. Can either be density or frequency
    definition, as well as a single year age contact matrix or a contact matrix
    for aggregated age brackets.

    Args:
        matrix (np.array)                  : symmetric contact matrix, element ij is the contact for an average individual in age group i with all of their contacts in age group j
        age_count (dict)                   : dictionary with the count of individuals in the population for each age
        aggregate_age_count (dict)         : dictionary with the count of individuals in the population in each age bracket
        age_brackets (dict)                : dictionary mapping age bracket keys to age bracket range
        age_by_brackets_dic (dict)         : dictionary mapping age to the age bracket range it falls in
        **layer (str)                      : name of the physial contact layer: H for households, S for schools, W for workplaces, C for community, etc.
        **density_or_frequency (str)       : Default value is 'density', see notes for more details.
        **logcolors_flag (bool)            : If True, plot heatmap in logscale
        **aggregate_flag (bool)            : If True, plot the contact matrix for aggregate age brackets, else single year age contact matrix.
        **cmap(str or matplotlib colormap) : colormap
        **fontsize (int)                   : base font size
        **rotation (int)                   : rotation for x axis labels
        **title_prefix(str)                : optional title prefix for the figure
        **fig (Figure)                     : if supplied, use this figure instead of generating one
        **ax (Axes)                        : if supplied, use these axes instead of generating one
        **titles (dict)                    : dictionary of titles to be used for different layers

    Returns:
        Matplotlib figure and axes.

    Note:
        For the long term care facilities layer you may want the age count and
        the aggregate age count to only consider those who live or work in long
        term care facilities. Otherwise, without counting these individuals
        separately, this matrix calculation and figure will be representative of
        the average mixing in the long term care facilities layer across the
        entire population. What will be produced is a matrix that shows little
        mixing between individuals in this layer as it is a representation of
        the average mixing and not just those present in this layer.

        The argument density_or_frequency (str) has two values : 'density' or
        'frequency'. See the description of sp.calculate_contact_matrix for more
        details. In brief,  'density' means that each contact counts for
        1/(group_size -1) of a person's contact in a group and 'frequency'
        counts each contact as 1. This means that in the 'frequency'
        description, the more people in a group or in contact with someone, the
        more higher rates of contact/exposure. In some disease contexts, this is
        the right description of contact/exposure. In others, a 'density'
        description is more appropriate. As always, how to define contact is
        disease specific and we suggest you look to literature on the specific
        disease you are modeling to decide which is best for your use.
    """
    plkwargs = plotting_kwargs()
    # method specific plotting defaults
    method_defaults = sc.objdict(layer='H', density_or_frequency='density',
                                 logcolors_flag=False, aggregate_flag=True,
                                 cmap='cmr.freeze_r', fontsize=16, rotation=50,
                                 title_prefix=None, fig=None, ax=None, titles=None)
    method_defaults.figname = f"contact_matrix_{method_defaults.layer}"  # by defining this here, we can at least ensure that default names connect to the layer being modeled

    plkwargs.update_defaults(method_defaults, kwargs)
    plkwargs.set_default_pop_pars()

    cmap = mplt.cm.get_cmap(plkwargs.cmap)

    if plkwargs.fig is None:
        fig = plt.figure(figsize=(10, 10), tight_layout=True)
    else:
        fig = plkwargs.fig
    if plkwargs.ax is None:
        ax = [fig.add_subplot(1, 1, 1)]
    else:
        ax = [plkwargs.ax]
    cax = []
    cbar = []
    implot = []

    if plkwargs.titles is None:
        plkwargs.titles = sc.objdict(H='Household', S='School',
                                     W='Work', LTCF='Long Term Care Facilities')

    if plkwargs.aggregate_flag:
        aggregate_M = spb.get_aggregate_matrix(matrix, age_by_brackets_dic)
        asymmetric_M = spb.get_asymmetric_matrix(aggregate_M, aggregate_age_count)
    else:
        asymmetric_M = spb.get_asymmetric_matrix(matrix, age_count)

    if plkwargs.logcolors_flag:

        vbounds = {}
        if plkwargs.density_or_frequency == 'frequency':
            if plkwargs.aggregate_flag:
                vbounds['H'] = {'vmin': 1e-2, 'vmax': 1e-0}
                vbounds['S'] = {'vmin': 1e-3, 'vmax': 1e-0}
                vbounds['W'] = {'vmin': 1e-3, 'vmax': 1e-0}
                vbounds['LTCF'] = {'vmin': 1e-3, 'vmax': 1e-1}
            else:
                vbounds['H'] = {'vmin': 1e-3, 'vmax': 1e-1}
                vbounds['S'] = {'vmin': 1e-3, 'vmax': 1e-1}
                vbounds['W'] = {'vmin': 1e-3, 'vmax': 1e-1}
                vbounds['LTCF'] = {'vmin': 1e-3, 'vmax': 1e-0}

        elif plkwargs.density_or_frequency == 'density':
            if plkwargs.aggregate_flag:
                vbounds['H'] = {'vmin': 1e-2, 'vmax': 1e0}
                vbounds['S'] = {'vmin': 1e-2, 'vmax': 1e1}
                vbounds['W'] = {'vmin': 1e-2, 'vmax': 1e1}
                vbounds['LTCF'] = {'vmin': 1e-3, 'vmax': 1e-0}

            else:
                vbounds['H'] = {'vmin': 1e-2, 'vmax': 1e0}
                vbounds['S'] = {'vmin': 1e-2, 'vmax': 1e0}
                vbounds['W'] = {'vmin': 1e-2, 'vmax': 1e0}
                vbounds['LTCF'] = {'vmin': 1e-2, 'vmax': 1e-0}

        im = ax[0].imshow(asymmetric_M.T, origin='lower',
                          interpolation='nearest', cmap=cmap,
                          norm=LogNorm(vmin=vbounds[plkwargs.layer]['vmin'],
                                       vmax=vbounds[plkwargs.layer]['vmax']))

    else:

        im = ax[0].imshow(asymmetric_M.T, origin='lower', interpolation='nearest', cmap=cmap)

    implot.append(im)

    if plkwargs.fontsize > 20:
        plkwargs.rotation = 90

    for i in range(len(ax)):
        divider = make_axes_locatable(ax[i])
        cax.append(divider.new_horizontal(size="4%", pad=0.15))

        fig.add_axes(cax[i])
        cbar.append(fig.colorbar(implot[i], cax=cax[i]))
        cbar[i].ax.tick_params(axis='y', labelsize=plkwargs.fontsize + 4)
        if plkwargs.density_or_frequency == 'frequency':
            cbar[i].ax.set_ylabel('Frequency of Contacts', fontsize=plkwargs.fontsize + 2)
        else:
            cbar[i].ax.set_ylabel('Density of Contacts', fontsize=plkwargs.fontsize + 2)
        ax[i].tick_params(labelsize=plkwargs.fontsize + 2)
        ax[i].set_xlabel('Age', fontsize=plkwargs.fontsize + 6)
        ax[i].set_ylabel('Age of Contacts', fontsize=plkwargs.fontsize + 6)
        ax[i].set_title(
            (plkwargs.title_prefix if plkwargs.title_prefix is not None else '') + plkwargs.titles[plkwargs.layer] + ' Age Mixing', fontsize=plkwargs.fontsize + 10)

        if plkwargs.aggregate_flag:
            tick_labels = [str(age_brackets[b][0]) + '-' + str(age_brackets[b][-1]) for b in age_brackets]
            ax[i].set_xticks(np.arange(len(tick_labels)))
            ax[i].set_xticklabels(tick_labels, fontsize=plkwargs.fontsize)
            ax[i].set_xticklabels(tick_labels, fontsize=plkwargs.fontsize, rotation=plkwargs.rotation)
            ax[i].set_yticks(np.arange(len(tick_labels)))
            ax[i].set_yticklabels(tick_labels, fontsize=plkwargs.fontsize)
        else:
            ax[i].set_xticks(np.arange(0, len(age_count) + 1, 10))
            ax[i].set_yticks(np.arange(0, len(age_count) + 1, 10))

    return fig, ax


def plot_contacts(pop, **kwargs):
    """
    Plot the age mixing matrix for a specific contact layer.

    Args:
        pop (pop object)                : population, either synthpops.pop.Pop or dict
        **layer (str)                   : name of the physial contact layer: H for households, S for schools, W for workplaces, C for community or other
        **aggregate_flag (bool)         : If True, plot the contact matrix for aggregate age brackets, else single year age contact matrix.
        **logcolors_flag (bool)         : If True, plot heatmap in logscale
        **density_or_frequency (str)    : If 'density', then each contact counts for 1/(group size -1) of a person's contact in a group, elif 'frequency' then count each contact. This means that more people in a group leads to higher rates of contact/exposure.
        **state_location (string)       : name of the state the location is in
        **country_location (string)     : name of the country the location is in
        **cmap (str or matplotlib cmap) : colormap
        **fontsize (int)                : base font size
        **rotation (int)                : rotation for x axis labels
        **title_prefix(str)             : optional title prefix for the figure
        **fig (matplotlib.figure)       : If supplied, use this figure instead of generating one
        **ax (matplotlib.axes)          : If supplied, use these axes instead of generating one
        **do_show (bool)                : If True, show the plot
        **do_save (bool)                : If True, save the plot to disk

    Returns:
        Matplotlib figure.
    """
    plkwargs = get_plkwargs(pop)

    # method specific plotting defaults
    method_defaults = sc.objdict(layer='H', density_or_frequency='density',
                                 aggregate_flag=True, logcolors_flag=True,
                                 cmap='cmr.freeze_r', fontsize=16, rotation=50,
                                 title_prefix=None, fig=None, ax=None, do_show=False, do_save=False,
                                 state_location=cfg.default_state, country_location=cfg.default_country
                                 )
    method_defaults.figname = f"contact_matrix_{method_defaults.layer}"  # by defining this here, we can at least ensure that default names connect to the layer being modeled

    plkwargs.update_defaults(method_defaults, kwargs)

    # now knowing the kwargs, update the location kwargs stored
    plkwargs.set_default_pop_pars()

    if isinstance(pop, sppop.Pop):
        population = pop.to_dict()
        age_brackets = pop.age_brackets
        age_by_brackets_dic = pop.age_by_brackets_dic
        age_count = pop.age_count

    elif isinstance(pop, dict):
        population = sc.dcp(pop)
        age_count = spb.count_ages(population)
        age_brackets = spdata.get_census_age_brackets(**plkwargs.loc_pars)
        age_by_brackets_dic = spb.get_age_by_brackets_dic(age_brackets)

    else:
        raise NotImplementedError(f"This method is not yet implemented for pop type: {type(pop)}")

    aggregate_age_count = spb.get_aggregate_ages(age_count, age_by_brackets_dic)
    matrix = calculate_contact_matrix(population, plkwargs.density_or_frequency, plkwargs.layer)

    fig, ax = plot_contact_matrix(matrix, age_count, aggregate_age_count, age_brackets, age_by_brackets_dic, **plkwargs)

    finalize_figure(fig, plkwargs)  # set figpath, and save and / or show figure

    return fig


def plot_array(expected, fig=None, ax=None, **kwargs):

    """
    Plot histogram on a sorted array based by names. If names not provided the
    order will be used. If generate data is not provided, plot only the expected
    values. Note this can only be used with the limitation that data that has
    already been binned. Figure will be saved in figdir if given or else working
    directory.

    Args:
        expected (array)        : Array of expected values
        fig (matplotlib.figure) : Matplotlib.figure object
        ax (matplotlib.axis)    : Matplotlib.axes object
        **generated (array)     : Array of values generated using a model
        **names (list or dict)  : names to display on x-axis, default is set to the indexes of data
        **figname (str)         : name to save figure to disk
        **figdir (str)          : directory to save the plot if provided
        **prefix (str)          : used to prefix the title of the plot
        **fontsize (float)      : default fontsize
        **color_1 (str)         : color for expected data
        **color_2 (str)         : color for generated data
        **expect_label (str)    : Label to show in the plot, default to "expected"
        **value_text (bool)     : If True, display the values on top of the bar if specified
        **rotation (float)      : rotation angle for xticklabels
        **binned (bool)         : If True, data are binned. Else, if False, plot a simple histogram for expected data.

    Returns:
        Matplotlib figure and axes.
    """
    plkwargs = plotting_kwargs()

    # method specific plotting defaults
    method_defaults = dict(generated=None, names=None, figdir=None, title_prefix="",
                           fontsize=12, color_1='mediumseagreen', color_2='#236a54',
                           expect_label='Expected', value_text=False, rotation=0,
                           tick_interval=10, tick_threshold=30, binned=True,
                           fig=fig, ax=ax, figname='example_figure')

    plkwargs.update_defaults(method_defaults, kwargs)
    plkwargs.set_font()  # font styles to be updated

    if fig is None:
        fig, ax = plt.subplots(1, 1)

    title = plkwargs.title_prefix.replace('_', ' ').title() if plkwargs.generated is None else f"{plkwargs.title_prefix.replace('_', ' ').title()} Comparison"
    ax.set_title(title, fontsize=plkwargs.fontsize + 2)
    x = np.arange(len(expected))

    if not plkwargs.binned:
        ax.hist(expected, label=plkwargs.expect_label.title(), color=plkwargs.color_1)
    else:
        rect1 = ax.bar(x, expected, label=plkwargs.expect_label.title(), color=plkwargs.color_1, zorder=0)
        if plkwargs.generated is not None:
            line, = ax.plot(x, plkwargs.generated, color=plkwargs.color_2, markeredgecolor='white', marker='o', markersize=plkwargs.markersize, label='Generated', zorder=1)
        if plkwargs.value_text:
            autolabel(ax, rect1, 0, 5)
            if plkwargs.generated is not None:
                for j, v in enumerate(plkwargs.generated):
                    ax.text(j, v, str(round(v, 3)), fontsize=10, horizontalalignment='right', verticalalignment='top', color=plkwargs.color_2)

        if plkwargs.names is not None:
            if isinstance(plkwargs.names, dict):
                xticks = sorted(plkwargs.names.keys())
                xticklabels = [plkwargs.names[k] for k in xticks]
            else:
                xticks = np.arange(len(plkwargs.names))
                xticklabels = plkwargs.names

            # if there are too many labels, only show every interval of ticks
            if len(plkwargs.names) > plkwargs.tick_threshold:
                xticks = xticks[0::plkwargs.tick_interval]
                xticklabels = xticklabels[0::plkwargs.tick_interval]
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels, rotation=plkwargs.rotation)

    leg = ax.legend(loc='upper right', fontsize=plkwargs.fontsize)
    leg.draw_frame(False)
    ax.set_xlim(x[0] - 1, x[-1] + 1)

    fig = finalize_figure(fig, plkwargs)  # set figpath, and save and / or show figure

    return fig, ax


def autolabel(ax, rects, h_offset=0, v_offset=0.3, **kwargs):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    Args:
        ax                 : Matplotlib.axes object
        rects              : Matplotlib.container.BarContainer
        h_offset (float)   : The position x to place the text at.
        v_offset (float)   : The position y to place the text at.
        **fontsize (float) : Default fontsize

    Returns:
        None.
    """

    # Set the annotation according to the input parameters
    method_defaults = dict(fontsize=10)  # in case kwargs does not have fontsize, add it
    kwargs = sc.mergedicts(method_defaults, kwargs)  # let kwargs override method defaults
    kwargs = sc.objdict(kwargs)
    for rect in rects:
        height = rect.get_height()
        text = ax.annotate('{}'.format(round(height, 3)),
                           xy=(rect.get_x() + rect.get_width() / 2, height),
                           xytext=(h_offset, v_offset),
                           textcoords="offset points",
                           ha='center', va='bottom')
        text.set_fontsize(kwargs.fontsize)


def plot_ages(pop, **kwargs):
    """
    Plot a comparison of the expected and generated age distribution.

    Args:
        pop (pop object)    : population, either synthpops.pop.Pop, covasim.people.People, or dict
        **left (float)      : Matplotlib.figure.subplot.left
        **right (float)     : Matplotlib.figure.subplot.right
        **top (float)       : Matplotlib.figure.subplot.top
        **bottom (float)    : Matplotlib.figure.subplot.bottom
        **color_1 (str)     : color for expected data
        **color_2 (str)     : color for data from generated population
        **fontsize (float)  : Matplotlib.figure.fontsize
        **figname (str)     : name to save figure to disk
        **comparison (bool) : If True, plot comparison to the generated population

    Returns:
        Matplotlib figure and axes.

    Note:
        If using pop with type covasim.people.Pop or dict, args must be supplied
        for the location parameters to get the expected distribution.

    **Example**::

        pars = {'n': 10e3, 'location':'seattle_metro', 'state_location':'Washington', 'country_location':'usa'}
        pop = sp.Pop(**pars)
        fig, ax = pop.plot_age_distribution_comparison()

        popdict = pop.to_dict()
        kwargs = pars.copy()
        kwargs['datadir'] = sp.datadir
        fig, ax = sp.plot_age_distribution_comparison(popdict, **kwargs)
    """
    plkwargs = get_plkwargs(pop)

    # method specific plotting defaults
    method_defaults = dict(left=0.10, right=0.95, top=0.90, bottom=0.10, color_1='#55afe1', color_2='#0a6299',
                           fontsize=12, figname='age_distribution_comparison', comparison=True, binned=True)

    plkwargs.update_defaults(method_defaults, kwargs)

    # define after plkwargs gets updated
    if isinstance(pop, sppop.Pop):
        plkwargs.loc_pars = pop.loc_pars
        plkwargs.smooth_ages = pop.smooth_ages
        plkwargs.window_length = pop.window_length

    elif not isinstance(pop, (dict, cv.people.People)):
        raise ValueError(f"This method does not support pop objects with the type {type(pop)}. Please look at the notes and try another supported pop type.")

    # now check for missing plkwargs and use default values if not found
    plkwargs.set_default_pop_pars()
    if 'title_prefix' not in plkwargs or plkwargs.title_prefix is None:
        plkwargs.title_prefix = f"{plkwargs.location}_age_distribution"

    # get the expected age distribution
    expected_age_dist = spdata.get_smoothed_single_year_age_distr(**sc.mergedicts(plkwargs.loc_pars, dict(window_length= plkwargs.window_length)))
    expected_age_dist_values = [expected_age_dist[k] * 100 for k in sorted(expected_age_dist.keys())]

    if plkwargs.comparison:
        generated_age_count = dict.fromkeys(expected_age_dist.keys(), 0)  # sets ordering of keys consistently

        # get the generated age distribution
        if isinstance(pop, sppop.Pop):
            generated_age_count = pop.age_count

        elif isinstance(pop, dict):
            for i, person in pop.items():
                generated_age_count[person['age']] += 1

        elif isinstance(pop, cv.people.People):
            generated_age_count = Counter(pop.age)

        generated_age_dist = spb.norm_dic(generated_age_count)
        generated_age_dist_values = [generated_age_dist[k] * 100 for k in sorted(generated_age_dist.keys())]
        max_y = np.ceil(max(0, max(expected_age_dist_values), max(generated_age_dist_values)))

    else:
        generated_age_dist_values = None
        max_y = np.ceil(max(0, max(expected_age_dist_values)))

    # update the fig
    fig, ax = plt.subplots(1, 1, figsize=(plkwargs.width, plkwargs.height), dpi=plkwargs.display_dpi)
    fig.subplots_adjust(**plkwargs.axis)

    fig, ax = plot_array(expected_age_dist_values, fig=fig, ax=ax, generated=generated_age_dist_values,
                         **sc.mergedicts(plkwargs, sc.objdict(do_show=False, do_save=False)))  # instead of saving now, will save after customizing the figure some more below

    ax.set_xlabel('Age', fontsize=plkwargs.fontsize)
    ax.set_ylabel('Distribution (%)', fontsize=plkwargs.fontsize)
    ax.set_xlim(-1, len(expected_age_dist_values))
    ax.set_ylim(0, max_y)
    ax.tick_params(labelsize=plkwargs.fontsize)

    fig = finalize_figure(fig, plkwargs)  # set figpath, and save and / or show figure

    return fig, ax


def plot_enrollment_rates(pop, **kwargs):
    """
    Plot a comparison of the expected and generated school enrollment rates by
    age.

    Args:
        pop (pop object)    : population, either synthpops.pop.Pop, covasim.people.People, or dict
        **left (float)      : Matplotlib.figure.subplot.left
        **right (float)     : Matplotlib.figure.subplot.right
        **top (float)       : Matplotlib.figure.subplot.top
        **bottom (float)    : Matplotlib.figure.subplot.bottom
        **color_1 (str)     : color for expected data
        **color_2 (str)     : color for data from generated population
        **fontsize (float)  : Matplotlib.figure.fontsize
        **figname (str)     : name to save figure to disk
        **comparison (bool) : If True, plot comparison to the generated population

    Returns:
        Matplotlib figure and axes.

    Note:
        If using pop with type covasim.people.Pop or dict, args must be supplied
        for the location parameter to get the expected rates.

    **Example**::

        pars = {'n': 10e3, 'location':'seattle_metro', 'state_location':'Washington', 'country_location':'usa'}
        pop = sp.Pop(**pars)
        fig, ax = pop.plot_enrollment_rates()

        popdict = pop.to_dict()
        kwargs = pars.copy()
        kwargs['datadir'] = sp.datadir
        fig, ax = sp.plot_enrollment_rates(popdit, **kwargs)
    """
    plkwargs = get_plkwargs(pop)

    # method specific plotting defaults
    method_defaults = dict(left=0.10, right=0.95, top=0.90, bottom=0.10, color_1='#55afe1', color_2='#0a6299',
                           fontsize=12, figname='enrollment_rates_by_age', comparison=True, binned=True)

    plkwargs.update_defaults(method_defaults, kwargs)

    # define after plkwargs get updated
    if isinstance(pop, sppop.Pop):
        plkwargs.loc_pars = pop.loc_pars
    elif not isinstance(pop, (dict, cv.people.People)):
        raise ValueError(f"This method does not support pop objects with the type {type(pop)}. Please look at the notes and try another supported pop type.")

    # now check for the missing plkwargs and use default values if not found
    plkwargs.set_default_pop_pars()
    if 'title_prefix' not in plkwargs or plkwargs.title_prefix is None:
        plkwargs.title_prefix = f"{plkwargs.location}_enrollment_rates_by_age"

    # get the expected enrollment rates
    expected_enrollment_rates = spdata.get_school_enrollment_rates(**plkwargs.loc_pars)
    expected_enrollment_rates_values = [expected_enrollment_rates[k] * 100 for k in sorted(expected_enrollment_rates.keys())]

    # if plkwargs.comparison:
        # generated_enrollment_rates


    fig, ax = plt.subplots(1, 1, figsize=(plkwargs.width, plkwargs.height), dpi=plkwargs.display_dpi)

    fig = finalize_figure(fig, plkwargs)

    return fig, ax


def plot_school_sizes(pop, **kwargs):
    """
    Plot a comparison of the expected and generated school size distribution for
    each type of school expected.

    Args:
        pop (pop object)                : population, either synthpops.pop.Pop, or dict
        **with_school_types (type)      : If True, plot school size distributions by type, else plot overall school size distributions
        **keys_to_exclude (str or list) : school types to exclude
        **left (float)                  : Matplotlib.figure.subplot.left
        **right (float)                 : Matplotlib.figure.subplot.right
        **top (float)                   : Matplotlib.figure.subplot.top
        **bottom (float)                : Matplotlib.figure.subplot.bottom
        **hspace (float)                : Matplotlib.figure.subplot.hspace
        **subplot_height (float)        : height of subplot in inches
        **subplot_width (float)         : width of subplot in inches
        **screen_height_factor (float)  : fraction of the screen height to use for display
        **location_text_y (float)       : height to add location text to figure
        **fontsize (float)              : Matplotlib.figure.fontsize
        **rotation (float)              : rotation angle for xticklabels
        **cmap (str)                    : colormap
        **figname (str)                 : name to save figure to disk
        **comparison (bool)             : If True, plot comparison to the generated population

    Returns:
        Matplotlib figure and axes.

    Note:
        If using pop with type covasim.people.Pop or dict, args must be supplied
        for the location parameters to get the expected distribution.

    **Example**::

        pars = {'n': 10e3, 'location'='seattle_metro', 'state_location'='Washington', 'country_location'='usa'}
        pop = sp.Pop(**pars)
        fig, ax = pop.plot_school_sizes_by_type()

        popdict = pop.to_dict()
        kwargs = pars.copy()
        kwargs['datadir'] = sp.datadir
        fig, ax = sp.plot_school_sizes(popdict, **kwargs)
    """
    plkwargs = get_plkwargs(pop)

    # method specific plotting defaults
    method_defaults = dict(with_school_types=False, keys_to_exclude=['uv'],
                           left=0.11, right=0.94, top=0.96, bottom=0.08, hspace=0.75,
                           subplot_height=2.8, subplot_width=4.2, screen_height_factor=0.85,
                           location_text_y=113, fontsize=8, rotation=25, cmap='cmo.curl',
                           figname='school_size_distribution_by_type', comparison=True,
                           school_type_labels=spsch.get_school_type_labels(),
                           )

    plkwargs.update_defaults(method_defaults, kwargs)
    plkwargs.set_font()

    if isinstance(plkwargs.keys_to_exclude, str):
        plkwargs.keys_to_exclude = [plkwargs.keys_to_exclude]  # ensure this is treated as a list

    # define after plkwargs gets updated
    if isinstance(pop, sppop.Pop):
        plkwargs.loc_pars = pop.loc_pars
        plkwargs.smooth_ages = pop.smooth_ages
        plkwargs.window_length = pop.window_length
        popdict = sc.dcp(pop.to_dict())

    elif isinstance(pop, dict):
        popdict = sc.dcp(pop)

    else:
        raise ValueError(f"This method does not support pop objects with the type {type(pop)}. Please look at the notes and try another supported pop type.")

    # now check for missing plkwargs and use default values if not found
    plkwargs.set_default_pop_pars()

    if plkwargs.with_school_types:
        expected_school_size_dist = spdata.get_school_size_distr_by_type(**plkwargs.loc_pars)
    else:
        plkwargs.school_type_labels = {None: ''}
        expected_school_size_dist = {None: spdata.get_school_size_distr_by_brackets(**plkwargs.loc_pars)}

    school_size_brackets = spdata.get_school_size_brackets(**plkwargs.loc_pars)
    bins = spsch.get_bin_edges(school_size_brackets)
    bin_labels = spsch.get_bin_labels(school_size_brackets)

    # calculate how many students are in each school
    if plkwargs.comparison:
        enrollment_by_school_type = spsch.get_enrollment_by_school_type(popdict, **dict(with_school_types=plkwargs.with_school_types, keys_to_exclude=plkwargs.keys_to_exclude))
        generated_school_size_dist = sc.objdict(spsch.get_generated_school_size_distributions(enrollment_by_school_type, bins))

    for school_type in plkwargs.keys_to_exclude:
        expected_school_size_dist.pop(school_type, None)
        plkwargs.school_type_labels.pop(school_type, None)

    sorted_school_types = sorted(expected_school_size_dist.keys())
    n_school_types = len(sorted_school_types)
    plkwargs.nrows = n_school_types

    # location text
    if plkwargs.location is not None:
        location_text = f"{plkwargs.location.replace('_', ' ').title()}"
    else:
        location_text = f"{cfg.default_location.replace('_', ' ').title()}"

    # create fig, ax, set cmap
    fig, ax = plt.subplots(n_school_types, 1, figsize=(plkwargs.display_width, plkwargs.display_height), dpi=plkwargs.display_dpi)
    cmap = mplt.cm.get_cmap(plkwargs.cmap)

    # readjust figure parameters
    if plkwargs.nrows == 1:
        ax = [ax]
        fig.set_size_inches(plkwargs.display_width, plkwargs.display_height * 0.47)
        plkwargs.update(dict(top=0.88, bottom=0.18, left=0.12))
        plkwargs.location_text_y = 105.5  # default value for singular school type -- you have the ability to change this by supplying the kwarg location_text_y

    # update the fig
    fig.subplots_adjust(**plkwargs.axis)

    for ns, school_type in enumerate(plkwargs.school_type_labels.keys()):
        x = np.arange(len(school_size_brackets))  # potentially will use different bins for each school type so placeholder for now
        c = ns / n_school_types
        c2 = min(c + 0.12, 1)

        sorted_bins = sorted(expected_school_size_dist[school_type].keys())

        ax[ns].bar(x, [expected_school_size_dist[school_type][b] * 100 for b in sorted_bins], color=cmap(c), edgecolor=cmap(c2), label='Expected', zorder=0)
        if plkwargs.comparison:
            ax[ns].plot(x, [generated_school_size_dist[school_type][b] * 100 for b in sorted_bins], color=cmap(c2), ls='--',
                        marker='o', markerfacecolor=cmap(c2), markeredgecolor='white', markeredgewidth=.5, markersize=plkwargs.markersize, label='Generated', zorder=1)
            leg = ax[ns].legend(loc=1, fontsize=plkwargs.fontsize)
            leg.draw_frame(False)
        ax[ns].set_xticks(x)
        ax[ns].set_xticklabels(bin_labels, rotation=plkwargs.rotation, fontsize=plkwargs.fontsize, verticalalignment='center_baseline')
        ax[ns].set_xlim(-0.6 + x[0], x[-1] + 0.6)
        ax[ns].set_ylim(0, 100)
        ax[ns].set_ylabel('%', fontsize=plkwargs.fontsize + 1)
        ax[ns].tick_params(labelsize=plkwargs.fontsize - 1)
        if school_type is None:
            title = "Without school types defined"
        else:
            title = f"{plkwargs.school_type_labels[school_type]}"
        if ns == 0:
            ax[ns].text(-0.6, plkwargs.location_text_y, location_text, horizontalalignment='left', fontsize=plkwargs.fontsize + 1, verticalalignment='top')
        ax[ns].set_title(title, fontsize=plkwargs.fontsize + 1, verticalalignment='top')
    ax[ns].set_xlabel('School size', fontsize=plkwargs.fontsize + 1, verticalalignment='center_baseline')

    # for multipanel figures, first display then re-adjust it and save to disk
    if plkwargs.do_show:
        plt.show()

    # update fig before saving to disk since display will modify things
    if plkwargs.do_save:
        if len(ax) == 1:
            fig.set_size_inches(plkwargs.width, plkwargs.height)

        else:
            cfg.logger.info("Setting default plotting parameters to save figure to disk. If these settings produce figures you would prefer to change, this method returns the figure and ax for you to modify and save to disk.")
            fig.set_size_inches(plkwargs.display_width, plkwargs.display_height)
            plkwargs.update(dict(bottom=0.075, hspace=0.52, left=0.12))

        fig.subplots_adjust(**plkwargs.axis)

        plkwargs.figpath = sc.makefilepath(filename=plkwargs.figname, folder=plkwargs.figdir, ext=plkwargs.format)
        fig.savefig(plkwargs.figpath, format=plkwargs.format, dpi=plkwargs.save_dpi)

    return fig, ax
