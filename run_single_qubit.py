import os
from itertools import product
import matplotlib.pyplot as plt
import scienceplots
import jax
from dill import dump, load

import pennylane as qml
from pennylane import numpy as np

from single_qubit import (
    get_random_observable,
    evaluate_on_grid,
    setup_grad_fn,
)

jax.config.update("jax_enable_x64", True)

jnp = jax.numpy

ad = "autodiff"
ps = r"$\mathrm{SU}(N)$ parameter-shift"
fd = "Finite difference"
spsr = "Stochastic parameter-shift"

def exact_grad(num_samples, delta):
    np.random.seed(21207)
    nqubits = 1
    observable = qml.Hermitian(get_random_observable(nqubits), wires=list(range(nqubits)))

    # a-axis resolution
    gran = 50

    # Fixed values of b for which to compute the gradient values
    b_lines = jnp.array([0.5, 1.0, 2.0])
    # grid for a-axis
    a_grid = jnp.linspace(0, np.pi + 0.001, gran)

    # Data generation/loading
    filename = f"data/paper/exact_grad_{num_samples}_{delta}.dill"
    if not os.path.exists(filename):
        grad_fns = {
            method: setup_grad_fn(method, observable, shots=None, delta=delta, num_samples=num_samples) for method in [ad, ps, fd, spsr]
        }
        grads = {method: evaluate_on_grid(grad_fns[method], a_grid, b_lines, observable=observable) for method in [ad, fd]}
        grads[spsr], spsr_std = evaluate_on_grid(grad_fns[spsr], a_grid, b_lines, observable=observable, sampled=True)
        grads[ps] = evaluate_on_grid(grad_fns[ps], a_grid, b_lines, jnp.array(0.0), observable=observable)
        with open(filename, "wb") as f:
            dump((grads, spsr_std), f)
    else:
        with open(filename, "rb") as f:
            grads, spsr_std = load(f)
            

    # Plotting
    plt.style.use("science")
    plt.rcParams.update({"font.size": 13})
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(5, 5)
    plot_kwargs = {
        ad: {"ls": "-", "lw": 0.8, "zorder": 10, "color": "k"},
        ps: {"ls": ":", "lw": 3, "zorder": 4, "color": "xkcd:bright blue"},
        fd: {"ls": "--", "lw": 1.8, "zorder": 2, "color": "xkcd:pinkish red"},
        spsr: {"ls": "-", "lw": 0.8, "zorder": 1, "color": "xkcd:grass green"},
    }
    text_pos = [(0.6, 0.88), (0.7, 0.7), (0.2, 0.3)]
    text_align = [("bottom", "left"), ("bottom", "left"), ("top", "center")]
    targ_pos = [(0.463, 0.78), (0.48, 0.6), (0.25, 0.465)]
    labels = [ad, ps, fd, spsr]
    for i in range(3):
        plots = []
        for method in labels:
            p, = ax.plot(a_grid, grads[method][:, i], **plot_kwargs[method])
            plots.append(p)

        for sign in [1, -1]:
            ax.plot(a_grid, grads[spsr][:, i] + sign * spsr_std[:, i], **plot_kwargs[spsr])
        fill = ax.fill_between(
            a_grid, 
            grads[spsr][:, i] - spsr_std[:, i],
            grads[spsr][:, i] + spsr_std[:, i],
            color=plot_kwargs[spsr]["color"],
            alpha=0.2, 
            zorder=plot_kwargs[spsr]["zorder"],
        )
        ax.text(*text_pos[i], rf"$b = ${b_lines[i]:1.1f}", transform=ax.transAxes, va=text_align[i][0], ha=text_align[i][1])
        ax.plot(*zip(text_pos[i], targ_pos[i]), lw=0.8, c="0.8", transform=ax.transAxes)
    labels[0] = "Exact value"

    ax.legend(plots[:3]+[(plots[3],fill)], labels, bbox_to_anchor=(0, 1), loc="lower left")
    ax.set(xlabel="$a$", xticks=[0, np.pi/2, np.pi], xticklabels=["$0$", "$\pi/2$", "$\pi$"])
    ax.set_ylabel(r"$\partial_a C(\boldmath{\theta})$")
    plt.tight_layout()
    if not os.path.exists("figures"):
        os.makedirs("figures")
    fig.savefig("./figures/exact_grad.pdf")
    plt.show()

def finite_diff_tune(shots, deltas, reps):
    np.random.seed(21207)
    nqubits = 1
    observable = qml.Hermitian(get_random_observable(nqubits), wires=list(range(nqubits)))

    # a-axis resolution
    gran = 50

    # Fixed values of b for which to compute the gradient values
    b_lines = jnp.array([0.5, 1.0, 2.0])
    # grid for a-axis
    a_grid = jnp.linspace(0, np.pi + 0.001, gran)

    # Data generation/loading
    deltas_str = str(deltas).replace("[", "").replace("]", "").replace(" ", "").replace(",", "_")
    filename = f"data/paper/finite_diff_tune_{shots}_{deltas}_{reps}.dill"
    if not os.path.exists(filename):
        grad_fns = {
            delta: setup_grad_fn(fd, observable, shots=shots, delta=delta) for delta in deltas if delta > 0
        }
        grad_fns[0] = setup_grad_fn(ad, observable)
        grads = {}
        stds = {}
        for delta in deltas:
            if delta==0:
                grads[delta] = evaluate_on_grid(grad_fns[delta], a_grid, b_lines, observable=observable)
            else:
                _grads = []
                _stds = []
                for _ in range(reps):
                    __grads, __stds = evaluate_on_grid(grad_fns[delta], a_grid, b_lines, observable=observable, sampled=True)
                    _grads.append(__grads)
                    _stds.append(__stds)
                grads[delta] = np.mean(_grads, axis=0)
                stds[delta] = np.mean(_stds, axis=0)
                stds[delta] /= np.sqrt(shots)
                grads[delta] -= grads[0]
        with open(filename, "wb") as f:
            dump((grads, stds), f)
    else:
        with open(filename, "rb") as f:
            grads, stds = load(f)
    grads[0] -= grads[0]

    # Plotting
    plt.style.use("science")
    plt.rcParams.update({"font.size": 15})
    fig, axs = plt.subplots(1, 3, figsize=(12, 4), gridspec_kw={"wspace": 0.05}, sharey=True)
    plot_kwargs = {
        deltas[0]: {"ls": "-", "lw": 0.8, "zorder": 0, "color": "0.7"},
        deltas[1]: {"ls": "-", "lw": 1.1, "zorder": 1, "color": "xkcd:grass green"},
        deltas[2]: {"ls": "--", "lw": 1.5, "zorder": 2, "color": "xkcd:pinkish red"},
        deltas[3]: {"ls": ":", "lw": 1.8, "zorder": 3, "color": "xkcd:bright blue"},
    }
    for i, ax in enumerate(axs):
        labels = []
        plots = []
        fills = []
        for delta in deltas:
            p, = ax.plot(a_grid, grads[delta][:, i], **plot_kwargs[delta])

            if delta>0:
                labels.append(f"$\delta={delta}$")
                plots.append(p)
                ls, lw, zorder, color = [plot_kwargs[delta].get(v) for v in ["ls", "lw", "zorder", "color"]]
                for sign in [1, -1]:
                    ax.plot(a_grid, 
                            grads[delta][:, i] + sign * stds[delta][:, i],
                            ls=ls, lw=lw/2, zorder=zorder, color=color)
                fills.append(ax.fill_between(
                    a_grid, 
                    grads[delta][:, i] - stds[delta][:, i],
                    grads[delta][:, i] + stds[delta][:, i],
                    color=color,
                    alpha=0.2, 
                    zorder=zorder,
                ))
        ax.set(xlabel="$a$", xticks=[0, np.pi/2, np.pi], xticklabels=["$0$", "$\pi/2$", "$\pi$"])
        ax.text(
            0.95, 0.05, rf"$b = ${b_lines[i]:1.1f}", transform=ax.transAxes, va="bottom", ha="right",
        )
        ax.set_xlim((0, np.pi))
        #ax.xaxis.set_ticks_position("bottom")
        #if i>0:
            #ax.set_yticks([])

    axs[1].legend(list(zip(plots, fills)), labels, bbox_to_anchor=(0.5, 1), loc="lower center", ncols=4)
    axs[0].set_ylabel(r"$[\partial_{\text{FD}, a}-\partial_a] C(\boldmath{\theta})$")
    plt.tight_layout()
    if not os.path.exists("figures"):
        os.makedirs("figures")
    fig.savefig("./figures/finite_diff_tune.pdf")
    plt.show()


def sampled_grad(shots, num_samples, delta):
    np.random.seed(21207)
    nqubits = 1
    observable = qml.Hermitian(get_random_observable(nqubits), wires=list(range(nqubits)))

    # a-axis resolution
    gran = 50

    # Fixed values of b for which to compute the gradient values
    b_lines = jnp.array([0.5, 1.0, 2.0])
    # grid for a-axis
    a_grid = jnp.linspace(0, np.pi + 0.001, gran)

    # Data generation/loading
    filename = f"data/paper/sampled_grad_{num_samples}_{shots}_{delta}.dill"
    if not os.path.exists(filename):
        shots = {ad: None, ps: shots, fd: shots, spsr: shots//num_samples}
        grad_fns = {
            method: setup_grad_fn(method, observable, shots=shots[method], delta=delta, num_samples=num_samples) for method in [ad, ps, fd, spsr]
        }
        grads = {ad: evaluate_on_grid(grad_fns[ad], a_grid, b_lines, observable=observable)}
        stds = {}
        grads[ps], stds[ps] = evaluate_on_grid(grad_fns[ps], a_grid, b_lines, jnp.array(0.0), observable=observable, sampled=True)
        grads[fd], stds[fd] = evaluate_on_grid(grad_fns[fd], a_grid, b_lines, observable=observable, sampled=True)
        grads[spsr], stds[spsr] = evaluate_on_grid(grad_fns[spsr], a_grid, b_lines, observable=observable, sampled=True)
        with open(filename, "wb") as f:
            dump((grads, stds), f)
    else:
        with open(filename, "rb") as f:
            grads, stds = load(f)
            

    # Plotting
    plt.style.use("science")
    plt.rcParams.update({"font.size": 13})
    fig, axs = plt.subplots(3, 1, figsize=(5, 8), gridspec_kw={"hspace": 0.0})
    plot_kwargs = {
        ad: {"ls": "-", "lw": 0.8, "zorder": 10, "color": "k"},
        ps: {"ls": "-", "lw": 0.8, "zorder": 4, "color": "xkcd:bright blue"},
        fd: {"ls": "-", "lw": 0.8, "zorder": 2, "color": "xkcd:pinkish red"},
        spsr: {"ls": "-", "lw": 0.8, "zorder": 1, "color": "xkcd:grass green"},
    }
    for i, ax in enumerate(axs):
        labels = []
        plots = []
        fills = []
        for method in [ad, ps, fd, spsr]:
            labels.append("Exact value" if method==ad else method)
            plots.append(ax.plot(a_grid, grads[method][:, i], **plot_kwargs[method])[0])

            if method!=ad:
                for sign in [1, -1]:
                    ax.plot(a_grid, grads[method][:, i] + sign * stds[method][:, i], **plot_kwargs[method])
                fills.append(ax.fill_between(
                    a_grid, 
                    grads[method][:, i] - stds[method][:, i],
                    grads[method][:, i] + stds[method][:, i],
                    color=plot_kwargs[method]["color"],
                    alpha=0.2, 
                    zorder=plot_kwargs[method]["zorder"],
                ))
        ax.set_ylabel(r"$\partial_a C(\boldmath{\theta})$")
        ax.text(
            0.05, 0.12, rf"$b = ${b_lines[i]:1.1f}", transform=ax.transAxes, va="top"
        )
        ax.xaxis.set_ticks_position("bottom")
        if i<2:
            ax.set_xticks([])

    handles = [plots[0]] + list(zip(plots[1:], fills))
    axs[0].legend(handles, labels, bbox_to_anchor=(0, 1), loc="lower left")
    ax.set(xlabel="$a$", xticks=[0, np.pi/2, np.pi], xticklabels=["$0$", "$\pi/2$", "$\pi$"])
    plt.tight_layout()
    if not os.path.exists("figures"):
        os.makedirs("figures")
    fig.savefig("./figures/sampled_grad.pdf")
    plt.show()


if __name__ == "__main__":
    shots = 1000
    num_samples = 100
    delta = 0.75
    exact_grad(num_samples, delta)
    sampled_grad(shots, num_samples, delta)

    shots = 100
    reps = 100
    deltas = [0, 0.5, 0.75, 1.]
    finite_diff_tune(shots, deltas, reps)
