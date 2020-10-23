"""
this test uses covasim people plot tool for contact distribution
results will be saved in covasim_report folder
note that tests looping over seeds 1 to 5000 with increment of 100, please change line 13 if needed
"""
import covasim as cv
import os
import pathlib

reportdir = pathlib.Path(os.path.dirname(__file__), "covasim_report")
os.makedirs(reportdir, exist_ok=True)

n = 2e4+1  # Total population size
for seed in range(1, 500, 100):
    for ltcf in [True, False]:
        label = "ltcf_" if ltcf else ""
        print("seed:", seed) # Random seed
        # Everything here gets passed to sp.make_population()
        pop_pars = dict(
            generate=True,
            with_facilities=ltcf,
        )
        if ltcf:
            pop_pars["layer_mapping"] = {'LTCF':'l'}
            sim = cv.Sim(pop_size=n, rand_seed=seed, pop_type='synthpops', beta_layer ={k: 1 for k in 'hscwl'})
        else:
            sim = cv.Sim(pop_size=n, rand_seed=seed, pop_type='synthpops')  # Make the Covasim oject
        ppl = cv.make_people(sim, **pop_pars)  # Create the corresponding population
        fig = ppl.plot()
        fig.savefig(os.path.join(reportdir, f"{label}covasim_graph_{n}_seed{seed}_new.png"), format="png")
