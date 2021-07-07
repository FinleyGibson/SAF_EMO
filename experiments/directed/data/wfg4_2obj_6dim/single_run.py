import sys
import rootpath
sys.path.append(rootpath.detect())
from testsuite.surrogates import GP, MultiSurrogate
from testsuite.directed_optimisers import DirectedSaf
from problem_setup import func, objective_function, limits
from generate_queue import targets

multi_surrogate = MultiSurrogate(GP, scaled=True)
budget = 150
seed = 0


print(targets)
opt = DirectedSaf(objective_function=objective_function, ei=False,  targets=targets[0], w=0.5, limits=limits, surrogate=multi_surrogate, n_initial=10, budget=budget, seed=seed)

print()
print(opt.y.shape)
opt.optimise()
