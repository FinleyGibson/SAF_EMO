import wfg
from testsuite.analysis_tools import draw_samples, attainment_sample
from testsuite.utilities import Pareto_split
from generate_problems import problem_list, strip_problem_names
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import pickle


def detect_peaks(bar_heights):
    return find_peaks(bar_heights)[0]


def quick_plot(data, c="C0", title=None):
    """
    shortcut to make simple scatter plots of data
    :param data: the points to be scatters
    :param c: color
    :param title: str title
    :return: matplotlib figure.
    """
    n_obj = data.shape[1]
    fig = plt.figure(figsize=[8,8])
    if n_obj == 2 :
        ax = fig.gca()
    elif n_obj == 3:
        ax = fig.gca(projection="3d")
    else:
        return None
    ax.scatter(*data.T, s=5, c=c)
    ax.set_title(title)
    return fig


# draw samples from all problems
prob_n = 3
for prob in problem_list[prob_n : prob_n+1]:
    prob, obj, dim = strip_problem_names(prob)
    func = getattr(wfg, f"WFG{prob}")
    x, y = draw_samples(func, n_obj=obj, n_dim=dim, n_samples=25000)
    y = Pareto_split(y)[0]
    self_D = distance_matrix(y, y)
    self_D[self_D == 0.] = self_D.max()
    self_D = self_D.min(axis=0)


hist_plot = plt.figure()
hist_ax = hist_plot.gca()
hist_p = hist_ax.hist(self_D, bins=40)
peak_ind = detect_peaks(hist_p[0])

for p in peak_ind:
    b = hist_p[1][p]
    hist_ax.axvline(hist_p[1][p], c="C2")
plt.show()

thresh0 = hist_p[1][peak_ind[0]]
# y = y[self_D<thresh0]

print(f"WFG{prob}_{obj}obj_{dim}dim")
print(func)
print(x.shape)
print(y.shape)

fig0 = quick_plot(y, c="C0", title="Pareto samples")
fig0.show()

ya = attainment_sample(y, n_samples=50000)
fig1 = quick_plot(ya, c="C1", title="Attainment sample")

yf = Pareto_split(ya)[0]
fig2 = quick_plot(yf, c="C2", title="Pareto optimal attainment samples")

D = distance_matrix(ya, Pareto_split(y)[0])
min_dif0 = D.min(axis=0)
min_dif1 = D.min(axis=1)

thresh = max(min_dif0)*0.9
yff = ya[min_dif1<thresh]

fig3 = quick_plot(yff, "C3", title="Thresholded attainment samples")

fig4 = plt.figure()
ax = fig4.gca()
ax.plot(sorted(min_dif0))
ax.plot(sorted(min_dif1))
ax.axhline(thresh, c="C2")

plt.show()

try:
    with open("reference_points", 'rb') as infile:
        reference_dict = pickle.load(infile)
except FileNotFoundError:
        reference_dict = {}

reference_dict[f"WFG{prob}_{obj}obj_{dim}dim"] = yff

with open("reference_points", 'wb') as outfile:
    reference_dict = pickle.dump(outfile, reference_dict)
