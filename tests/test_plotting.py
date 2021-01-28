'''
Test plotting functions
'''

import pylab as pl
import sciris as sc
import synthpops as sp

def test_plots(do_plot=False):
    ''' Basic plots '''
    if not do_plot:
        pl.switch_backend('agg') # Plot but don't show
    pop = sp.Pop(n=5000) # default parameters, 5k people
    fig1 = pop.plot_people() # equivalent to cv.Sim.people.plot()
    fig2 = pop.plot_contacts() # equivalent to sp.plot_contact_matrix(popdict)
    return [fig1, fig2]


if __name__ == '__main__':

    T = sc.tic()

    figs = test_plots(do_plot=True)

    sc.toc(T)
    print('Done.')
