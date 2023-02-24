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

def exact_grad(num_samples):
    np.random.seed(21207)
    nqubits = 1
    observable = qml.Hermitian(get_random_observable(nqubits), wires=list(range(nqubits)))

    # a-axis resolution
    gran = 50
    # Finite difference spacing
    delta = 0.5

    # Fixed values of b for which to compute the gradient values
    b_lines = jnp.array([0.5, 1.0, 2.0])
    # grid for a-axis
    a_grid = jnp.linspace(0, np.pi + 0.001, gran)

    # Data generation/loading
    filename = f"data/exact_grad_{num_samples}.dill"
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
    for i in range(3):
        for method in [ad, ps, fd, spsr]:
            label = method if i==0 else ""
            if method==ad:
                label = "Exact value" if i==0 else ""
            ax.plot(a_grid, grads[method][:, i], label=label, **plot_kwargs[method])

        for sign in [1, -1]:
            ax.plot(a_grid, grads[spsr][:, i] + sign * spsr_std[:, i], **plot_kwargs[spsr])
        ax.fill_between(
            a_grid, 
            grads[spsr][:, i] - sign * spsr_std[:, i],
            grads[spsr][:, i] + sign * spsr_std[:, i],
            color=plot_kwargs[spsr]["color"],
            alpha=0.2, 
            zorder=plot_kwargs[spsr]["zorder"],
        )

    ax.legend(bbox_to_anchor=(0, 1), loc="lower left")
    ax.set_xlabel("$a$")
    ax.set_ylabel(r"$\frac{\partial C(\boldmath{\theta})}{\partial a}$")
    plt.tight_layout()
    if not os.path.exists("figures"):
        os.makedirs("figures")
    fig.savefig("./figures/exact_grad.pdf")
    plt.show()

def sampled_grad(shots, num_samples):
    np.random.seed(21207)
    nqubits = 1
    observable = qml.Hermitian(get_random_observable(nqubits), wires=list(range(nqubits)))

    # a-axis resolution
    gran = 50
    # Finite difference spacing
    delta = 0.5

    # Fixed values of b for which to compute the gradient values
    b_lines = jnp.array([0.5, 1.0, 2.0])
    # grid for a-axis
    a_grid = jnp.linspace(0, np.pi + 0.001, gran)

    # Data generation/loading
    filename = f"data/sampled_grad_{num_samples}_{shots}.dill"
    if not os.path.exists(filename):
        grad_fns = {
            method: setup_grad_fn(method, observable, shots=None if method==ad else shots, delta=delta, num_samples=num_samples) for method in [ad, ps, fd, spsr]
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
        for method in [ad, ps, fd, spsr]:
            label = "Exact value" if method==ad else method 
            ax.plot(a_grid, grads[method][:, i], label=label, **plot_kwargs[method])

            if method!=ad:
                for sign in [1, -1]:
                    ax.plot(a_grid, grads[method][:, i] + sign * stds[method][:, i], **plot_kwargs[method])
                print(stds[method][:, i])
                ax.fill_between(
                    a_grid, 
                    grads[method][:, i] - sign * stds[method][:, i],
                    grads[method][:, i] + sign * stds[method][:, i],
                    color=plot_kwargs[method]["color"],
                    alpha=0.2, 
                    zorder=plot_kwargs[method]["zorder"],
                )
        ax.set_ylabel(r"$\frac{\partial C(\boldmath{\theta})}{\partial a}$")
        ax.text(
            0.08, 0.95, rf"$b = ${b_lines[i]:1.1f}", transform=ax.transAxes, verticalalignment="top"
        )
        ax.xaxis.set_ticks_position("bottom")
        if i<2:
            ax.set_xticks([])

    axs[0].legend(bbox_to_anchor=(0, 1), loc="lower left")
    ax.set_xlabel("$a$")
    plt.tight_layout()
    if not os.path.exists("figures"):
        os.makedirs("figures")
    fig.savefig("./figures/sampled_grad.pdf")
    plt.show()


if __name__ == "__main__":
    shots = 1000
    num_samples = 100
    #exact_grad(num_samples)
    sampled_grad(shots, num_samples)
