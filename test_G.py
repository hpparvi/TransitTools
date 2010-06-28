import Gimenez_f as G
import numpy as np
import pylab as pl

z = np.linspace(-2, 2, 150)

print dir(G)

f = G.gimenez_f.gimenez(np.abs(z), 0.1, [0.1, 0.0], 500)

pl.plot(z, f)
pl.show()
