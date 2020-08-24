import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pyplot as plt
from testsuite.utilities import Pareto_split
import numpy as np

PLOT_STYLE = {"scatter_cmap": mpl.cm.Purples,
              "scatter_style": 'seaborn'}


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
