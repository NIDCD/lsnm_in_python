################################################################################################
# The following script was courtesy of Michael Marmaduke Woodman, of the TVB development team
# June 18 2015
###############################################################################################

from tvb.simulator.lab import *

import numpy as np
import matplotlib.pyplot as pl


class WilsonCowanPositive(models.WilsonCowan):
    "W-C Equations clamped > 0 for stochastic methods"
    def dfun(self, state_variables, coupling, local_coupling=0.0):
        state_variables[state_variables < 0.0] = 0.0
        return super(WilsonCowanPositive, self).dfun(state_variables, coupling, local_coupling)

n = 5
conn = connectivity.Connectivity()
conn.weights = np.random.rand(n, n)
conn.tract_lengths = np.zeros((n, n))

scheme = integrators.EulerStochastic(
    dt=0.01,
    noise=noise.Additive(nsig=0.01)
)

def run_plot(model_class, subplot_idx, title):
    sim = simulator.Simulator(
        model=model_class(),
        integrator=scheme,
        monitors=monitors.Raw(),
        connectivity=conn
    )
    sim.configure()
    ys = []
    for (t, y), in sim(100.0):
        ys.append(y)
    ys = np.array(ys)

    pl.subplot(2, 1, subplot_idx)
    pl.plot(ys[:, 0, :, 0], 'k')
    pl.grid(True)
    pl.title(title)


pl.figure()
run_plot(models.WilsonCowan, 1, 'Wilson-Cowan additive noise, non-bounded (incorrect)')
run_plot(WilsonCowanPositive, 2, 'Wilson-Cowan additive noise, bounded (correct)')
pl.tight_layout()
pl.show()
