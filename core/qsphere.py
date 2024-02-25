
from functools import reduce
import colorsys

import numpy as np
from qiskit.utils.deprecation import deprecate_arg
from qiskit.utils import optionals as _optionals
from qiskit.circuit.tools.pi_check import pi_check
from qiskit.visualization.utils import matplotlib_close_if_inline
from qiskit.visualization.exceptions import VisualizationError

def n_choose_k(n, k):

    if n == 0:
        return 0
    return reduce(lambda x, y: x * y[0] / y[1], zip(range(n - k + 1, n + 1), range(1, k + 1)), 1)


def lex_index(n, k, lst):

    if len(lst) != k:
        raise VisualizationError("list should have length k")
    comb = [n - 1 - x for x in lst]
    dualm = sum(n_choose_k(comb[k - 1 - i], i + 1) for i in range(k))
    return int(dualm)


def bit_string_index(s):
    """Return the index of a string of 0s and 1s."""
    n = len(s)
    k = s.count("1")
    if s.count("0") != n - k:
        raise VisualizationError("s must be a string of 0 and 1")
    ones = [pos for pos, char in enumerate(s) if char == "1"]
    return lex_index(n, k, ones)


def phase_to_rgb(complex_number):

    angles = (np.angle(complex_number) + (np.pi * 5 / 4)) % (np.pi * 2)
    rgb = colorsys.hls_to_rgb(angles / (np.pi * 2), 0.5, 0.5)
    return rgb



@deprecate_arg("rho", new_alias="state", since="0.15.1")
@_optionals.HAS_MATPLOTLIB.require_in_call
@_optionals.HAS_SEABORN.require_in_call
def plot_state_qsphere(
    sparse_eigenvecs,
    eigvals_diag,
    state,
    num_qubits,
    figsize=None,
    ax=None,
    show_state_labels=True,
    show_state_phases=False,
    use_degrees=False,
    *,
    rho=None,
    filename=None,
):

    from matplotlib import gridspec
    from matplotlib import pyplot as plt
    from matplotlib.patches import Circle
    import seaborn as sns
    from qiskit.visualization.bloch import Arrow3D

    #rho = DensityMatrix(state)
    #num = rho.num_qubits
    num = num_qubits
    if num is None:
        raise VisualizationError("Input is not a multi-qubit quantum state.")
    # get the eigenvectors and eigenvalues
    #eigvals, eigvecs = linalg.eigh(rho.data)

    eigvals = eigvals_diag

    eigvecs = sparse_eigenvecs.T.toarray()

    if figsize is None:
        figsize = (7, 7)

    if ax is None:
        return_fig = True
        fig = plt.figure(figsize=figsize)
    else:
        return_fig = False
        fig = ax.get_figure()

    gs = gridspec.GridSpec(nrows=3, ncols=3)

    ax = fig.add_subplot(gs[0:3, 0:3], projection="3d")
    ax.axes.set_xlim3d(-1.0, 1.0)
    ax.axes.set_ylim3d(-1.0, 1.0)
    ax.axes.set_zlim3d(-1.0, 1.0)
    ax.axes.grid(False)
    ax.view_init(elev=5, azim=275)

    # Force aspect ratio
    # MPL 3.2 or previous do not have set_box_aspect
    if hasattr(ax.axes, "set_box_aspect"):
        ax.axes.set_box_aspect((1, 1, 1))

    # start the plotting
    # Plot semi-transparent sphere
    u = np.linspace(0, 2 * np.pi, 25)
    v = np.linspace(0, np.pi, 25)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(
        x, y, z, rstride=1, cstride=1, color=plt.rcParams["grid.color"], alpha=0.2, linewidth=0
    )

    # Get rid of the panes
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # Get rid of the spines
    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    # Get rid of the ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # traversing the eigvals/vecs backward as sorted low->high
    for idx in range(eigvals.shape[0] - 1, -1, -1):
        if eigvals[idx] > 0.001:
            # get the max eigenvalue
            state = eigvecs[:, idx]
            loc = np.absolute(state).argmax()
            # remove the global phase from max element
            angles = (np.angle(state[loc]) + 2 * np.pi) % (2 * np.pi)
            angleset = np.exp(-1j * angles)
            state = angleset * state
            
            d = num
            nz = np.array(state).nonzero()[0]
            for i in nz:
                # get x,y,z points
                element = bin(i)[2:].zfill(num)
                #print(element)
                weight = element.count("1")
                zvalue = -2 * weight / d + 1
                number_of_divisions = n_choose_k(d, weight)
                weight_order = bit_string_index(element)
                angle = (float(weight) / d) * (np.pi * 2) + (
                    weight_order * 2 * (np.pi / number_of_divisions)
                )

                if (weight > d / 2) or (
                    (weight == d / 2) and (weight_order >= number_of_divisions / 2)
                ):
                    angle = np.pi - angle - (2 * np.pi / number_of_divisions)

                xvalue = np.sqrt(1 - zvalue**2) * np.cos(angle)
                yvalue = np.sqrt(1 - zvalue**2) * np.sin(angle)

                # get prob and angle - prob will be shade and angle color
                prob = np.real(np.dot(state[i], state[i].conj()))
                #print(prob)
                prob = min(prob, 1)  # See https://github.com/Qiskit/qiskit-terra/issues/4666
                colorstate = phase_to_rgb(state[i])

                alfa = 1
                if yvalue >= 0.1:
                    alfa = 1.0 - yvalue

                if not np.isclose(prob, 0) and show_state_labels:
                    rprime = 1.3
                    angle_theta = np.arctan2(np.sqrt(1 - zvalue**2), zvalue)
                    xvalue_text = rprime * np.sin(angle_theta) * np.cos(angle)
                    yvalue_text = rprime * np.sin(angle_theta) * np.sin(angle)
                    zvalue_text = rprime * np.cos(angle_theta)
                    element_text = "$\\vert" + element + "\\rangle$"
                    #print(element_text)
                    if show_state_phases:
                        element_angle = (np.angle(state[i]) + (np.pi * 4)) % (np.pi * 2)
                        if use_degrees:
                            element_text += "\n$%.1f^\\circ$" % (element_angle * 180 / np.pi)
                        else:
                            element_angle = pi_check(element_angle, ndigits=3).replace("pi", "\\pi")
                            element_text += "\n$%s$" % (element_angle)
                    ax.text(
                        xvalue_text,
                        yvalue_text,
                        zvalue_text,
                        element_text,
                        ha="center",
                        va="center",
                        size=12,
                    )

                ax.plot(
                    [xvalue],
                    [yvalue],
                    [zvalue],
                    markerfacecolor=colorstate,
                    markeredgecolor=colorstate,
                    marker="o",
                    markersize=np.sqrt(prob) * 30,
                    alpha=alfa,
                )
                #print(ax)

                a = Arrow3D(
                    [0, xvalue],
                    [0, yvalue],
                    [0, zvalue],
                    mutation_scale=20,
                    alpha=prob,
                    arrowstyle="-",
                    color=colorstate,
                    lw=2,
                )
                #print(element)
                ax.add_artist(a)
                #print(a)

            

            # add weight lines
            for weight in range(d + 1):
                theta = np.linspace(-2 * np.pi, 2 * np.pi, 100)
                z = -2 * weight / d + 1
                r = np.sqrt(1 - z**2)
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                ax.plot(x, y, z, color=(0.5, 0.5, 0.5), lw=1, ls=":", alpha=0.5)

            # add center point
            ax.plot(
                [0],
                [0],
                [0],
                markerfacecolor=(0.5, 0.5, 0.5),
                markeredgecolor=(0.5, 0.5, 0.5),
                marker="o",
                markersize=3,
                alpha=1,
            )
        else:
            break

    n = 64
    theta = np.ones(n)
    colors = sns.hls_palette(n)

    ax2 = fig.add_subplot(gs[2:, 2:])
    ax2.pie(theta, colors=colors[5 * n // 8 :] + colors[: 5 * n // 8], radius=0.75)
    ax2.add_artist(Circle((0, 0), 0.5, color="white", zorder=1))
    offset = 0.95  # since radius of sphere is one.

    if use_degrees:
        labels = ["Phase\n(Deg)", "0", "90", "180   ", "270"]
    else:
        labels = ["Phase", "$0$", "$\\pi/2$", "$\\pi$", "$3\\pi/2$"]

    ax2.text(0, 0, labels[0], horizontalalignment="center", verticalalignment="center", fontsize=14)
    ax2.text(
        offset, 0, labels[1], horizontalalignment="center", verticalalignment="center", fontsize=14
    )
    ax2.text(
        0, offset, labels[2], horizontalalignment="center", verticalalignment="center", fontsize=14
    )
    ax2.text(
        -offset, 0, labels[3], horizontalalignment="center", verticalalignment="center", fontsize=14
    )
    ax2.text(
        0, -offset, labels[4], horizontalalignment="center", verticalalignment="center", fontsize=14
    )

    if return_fig:
        matplotlib_close_if_inline(fig)
    if filename is None:
        return fig
    else:
        return fig.savefig(filename)
