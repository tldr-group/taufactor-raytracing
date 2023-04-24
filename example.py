import tifffile
import taufactor
from raytrace import rt
import numpy as np

img = np.random.randint(0, 2, (300,300,300))
solver = taufactor.Solver(img)
solver.solve()
# can change maptype to flux.
rt(solver, maptype='conc', savepth='test')