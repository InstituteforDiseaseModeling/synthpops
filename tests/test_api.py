import numpy as np
import pylab as pl
import sciris as sc
import pytest
import synthpops as sp


if not sp.config.full_data_available:
    pytest.skip("Data not available, tests not possible", allow_module_level=True)


def test_api():

    n = 2000
    max_contacts = {'S': 20, 'W': 10}

    population = sp.make_population(n=n, max_contacts=max_contacts)

    with pytest.raises(ValueError):
        population = sp.make_population(n=298437, generate=False)  # Not a supported number

    return population


def test_plot_pop(do_show=False, pause=0.2):

    plotconnections = True
    n = 5000
    alpha = 0.5

    # indices = pl.arange(1000)
    pl.seed(1)
    indices = pl.randint(0, n, 20)

    max_contacts = {'S': 20, 'W': 10}
    population = sp.make_population(n=n, max_contacts=max_contacts)

    nside = np.ceil(np.sqrt(n))
    x, y = np.meshgrid(np.arange(nside), np.arange(nside))
    x = x.flatten()[:n]
    y = y.flatten()[:n]

    people = list(population.values())
    for p, person in enumerate(people):
        person['loc'] = dict(x=x[p], y=y[p])
    ages = np.array([person['age'] for person in people])
    f_inds = [ind for ind, person in enumerate(people) if not person['sex']]
    m_inds = [ind for ind, person in enumerate(people) if person['sex']]

    if do_show:

        use_terrain = False
        if use_terrain:
            import matplotlib.pyplot as plt
            import matplotlib.colors as colors
            colors_undersea = plt.cm.terrain(np.linspace(0, 0.17, 256))
            colors_land = plt.cm.terrain(np.linspace(0.25, 1, 256))
            all_colors = np.vstack((colors_undersea, colors_land))
            terrain_map = colors.LinearSegmentedColormap.from_list('terrain_map',
                all_colors)
            pl.set_cmap(terrain_map)

        fig = pl.figure(figsize=(24, 18))
        pl.subplot(111)
        minval = 0  # ages.min()
        maxval = 100  # ages.min()
        colors = sc.vectocolor(ages, minval=minval, maxval=maxval)
        for i, inds in enumerate([f_inds, m_inds]):
            pl.scatter(x[inds], y[inds], marker='os'[i], c=colors[inds])
        pl.clim([minval, maxval])
        pl.colorbar()

        if plotconnections:
            lcols = dict(H=[0, 0, 0], S=[0, 0.5, 1], W=[0, 0.7, 0], C=[1, 1, 0])
            for index in indices:
                person = people[index]
                contacts = person['contacts']
                lines = []
                for lkey in lcols.keys():
                    for contactkey in contacts[lkey]:
                        contact = population[contactkey]
                        tmp = pl.plot([person['loc']['x'], contact['loc']['x']], [person['loc']['y'], contact['loc']['y']], c=lcols[lkey], alpha=alpha)
                        lines.append(tmp)
                if pause:
                    pl.pause(pause)

        return fig


if __name__ == '__main__':
    pop1 = test_api()
    fig  = test_plot_pop(do_show=True)
    print('Done.')
