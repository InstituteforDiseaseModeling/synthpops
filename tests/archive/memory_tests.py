import sciris  as sc
import covasim as cv
import synthpops as sp

ps1 = 10e3
ps2 = 10023

def synth():

    p1 = sp.make_population(ps1)
    p2 = sp.make_population(ps2)

    return [p1, p2]


def cova():

    sc.tic()
    s1 = cv.Sim(pop_size=ps2, pop_type='random')
    s1.initialize()
    sc.toc()

    sc.tic()
    s2 = cv.Sim(pop_size=ps2, pop_type='hybrid')
    s2.initialize()
    sc.toc()

    sc.tic()
    s3 = cv.Sim(pop_size=ps2, pop_type='synthpops')
    s3.initialize()
    sc.toc()

    sc.tic()
    s4 = cv.Sim(pop_size=ps1, pop_type='synthpops')
    s4.initialize()
    sc.toc()

    return [s1, s2, s3, s4]

ps = synth()

sc.mprofile(run=synth, follow=synth)