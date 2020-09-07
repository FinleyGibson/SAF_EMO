import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pyplot as plt
from testsuite.utilities import Pareto_split
import numpy as np
from collections import OrderedDict

CMAPS = OrderedDict()
CMAPS['Sequential'] = [
            'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

PLOT_STYLE = {"scatter_cmap": mpl.cm.Purples,
              "scatter_cmaps": CMAPS['Sequential'],
              "scatter_style": 'seaborn',
              "plot_style": 'seaborn',
              "plot_cmap": mpl.cm.rainbow}


def plot_pareto_2d(result, axis=None):

    y = result["y"]

    p_inds, d_inds = Pareto_split(y, return_indices=True)

    cmap = PLOT_STYLE['scatter_cmap']
    colors = cmap(np.linspace(0, 1, len(y)))
    face_colors = [colors[i] if i in p_inds else "None" for i in range(len(y))]
    norm = mpl.colors.Normalize(vmin=0, vmax=len(y))
    mapable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    with plt.style.context(PLOT_STYLE["scatter_style"]):
        # create figure if axis is not provided.
        if axis is None:
            fig = plt.figure(figsize=[8.5, 7])
            ax = fig.gca()

        ax.scatter(y[:, 0], y[:, 1], marker="o", edgecolors=colors,
                   linewidth=2., facecolors=face_colors)
        ax.set_xlabel("$f^1(x)$")
        ax.set_ylabel("$f^2(x)$")
        plt.colorbar(mapable)

    if axis is not None:
        return fig
    else:
        pass


def plot_all_pareto_2d(results, axis=None, plot_indices=None):
    ys = np.array(results["y"])
    plot_indices = range(len(ys)) if plot_indices is None else plot_indices

    with plt.style.context(PLOT_STYLE["scatter_style"]):
        # create figure if axis is not provided.
        if axis is None:
            fig = plt.figure(figsize=[8.5, 7])
            ax = fig.gca()

        for i, y in enumerate(ys[plot_indices]):
            p_inds, d_inds = Pareto_split(y, return_indices=True)
            cmap = getattr(mpl.cm, PLOT_STYLE['scatter_cmaps'][i])
            colors = cmap(np.linspace(0, 1, len(y)))
            face_colors = [colors[i] if i in p_inds else "None" for i in
                           range(len(y))]
            norm = mpl.colors.Normalize(vmin=0, vmax=len(y))
            mapable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

            ax.scatter(y[:, 0], y[:, 1], marker="o", edgecolors=colors,
                       linewidth=2., facecolors=face_colors)
        ax.set_xlabel("$f^1(x)$")
        ax.set_ylabel("$f^2(x)$")
        plt.colorbar(mapable)

    if axis is not None:
        return fig
    else:
        pass


def plot_measure(results, measure, axis=None, label=None, plot_individuals=False,
                     color="C0"):
    if axis is None:
        fig = plt.figure(figsize=[12, 8])
        axis = fig.gca()

    hvs = results[measure]

    with plt.style.context(PLOT_STYLE["plot_style"]):
        if plot_individuals:
            for i, hv in enumerate(hvs):
                # plot idividual results
                n_inital = results["n_initial"][i]
                n_total = results["n_evaluations"][i]
                bo_steps = range(n_inital, n_total + 1)
                axis.plot(bo_steps, hv, linestyle=":", c=color, alpha=0.4)

        # trim hvs to min length so as to compute mean
        array_hvs = np.array([hv[:min([len(hv) for hv in hvs])] for hv in hvs])
        n_inital = results["n_initial"][0]
        bo_steps = range(n_inital, array_hvs.shape[1]+n_inital)
        # plot mean and standard deviations
        axis.plot(bo_steps, np.median(array_hvs, axis=0), linestyle="-", c=color,
                  alpha=1., label=label)
        # axis.fill_between(bo_steps,
        #                   np.mean(array_hvs, axis=0) - np.std(array_hvs, axis=0),
        #                   np.mean(array_hvs, axis=0) + np.std(array_hvs, axis=0),
        #                   color=color, alpha=0.2)
        lower_qa = np.array([np.quantile(i, 0.25) for i in np.array(array_hvs).T])
        upper_qa = np.array([np.quantile(i, 0.75) for i in np.array(array_hvs).T])
        axis.fill_between(bo_steps,
                          lower_qa,
                          upper_qa,
                          color=color, alpha=0.2)

    if axis is None:
        return fig
