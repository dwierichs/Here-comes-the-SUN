import os
from itertools import product
import matplotlib.pyplot as plt
import scienceplots
import jax

import pennylane as qml
from pennylane import numpy as np

from single_qubit import (
    get_random_observable,
    circuit,
    circuit_with_opg,
    finite_diff_first,
    evaluate_on_grid,
)

jax.config.update("jax_enable_x64", True)

jnp = jax.numpy


def exact_grad():
    np.random.seed(21207)
    nqubits = 1
    dev = qml.device("default.qubit", wires=nqubits)
    observable = qml.Hermitian(get_random_observable(nqubits), wires=list(range(nqubits)))

    # a-axis resolution
    gran = 50
    # Finite difference spacing
    delta = 1e-5

    # QNode that takes parameters a and b, as well as the observable
    qnode = qml.QNode(circuit, dev, interface="jax")
    # QNode that takes parameters a and b, as well as the observable, and parameter t for the
    # exponential of the effective generator
    qnode_grad = qml.QNode(circuit_with_opg, dev, interface="jax")

    # Gradient computed by differentiating exp(Omega * t)
    grad_fn_generator = jax.jit(jax.grad(qnode_grad, argnums=2), static_argnums=3)
    # Gradient computed via automatic differentiation
    grad_fn_autodiff = jax.jit(jax.grad(qnode, argnums=0), static_argnums=2)
    # Gradient computed with finite difference
    grad_fn_finite_diff = finite_diff_first(qnode, dx=delta, argnums=0)

    qnode = jax.jit(qnode, static_argnums=2)

    # Fixed values of b for which to compute the gradient values
    b_lines = [0.5, 1.0, 2.0]
    # grid for a-axis
    a_grid = np.linspace(0, np.pi + 0.001, gran)

    grads_a_generator = evaluate_on_grid(
        grad_fn_generator, a_grid, b_lines, jnp.array(0.0), observable=observable
    )
    grads_a_exact = evaluate_on_grid(grad_fn_autodiff, a_grid, b_lines, observable=observable)
    grads_a_finite_diff = evaluate_on_grid(
        grad_fn_finite_diff, a_grid, b_lines, observable=observable
    )

    plt.style.use("science")
    plt.rcParams.update({"font.size": 15})
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(5, 4)
    for i in range(3):
        plot = axs.plot(a_grid, grads_a_generator[:, i], lw=2)[0]
        axs.plot(a_grid, grads_a_exact[:, i], ls="--", lw=2, c="k")
        axs.plot(
            a_grid,
            grads_a_finite_diff[:, i],
            lw=2,
            label=rf"$b = ${b_lines[i]:1.1f}",
            marker="o",
            ls="",
            markersize=4,
        )

    axs.legend()
    axs.set_xlabel("$a$")
    axs.set_ylabel(r"$\frac{\partial C(\boldmath{\theta})}{\partial a}$")
    plt.tight_layout()
    if not os.path.exists("figures"):
        os.makedirs("figures")
    fig.savefig("./figures/exact_grad.pdf")
    plt.show()


def sampled_grad(shots):
    np.random.seed(21207)
    nqubits = 1
    dev = qml.device("default.qubit", wires=nqubits)
    dev_sample = qml.device("default.qubit", wires=nqubits, shots=[1] * shots)
    observable = qml.Hermitian(get_random_observable(nqubits), wires=list(range(nqubits)))

    # a-axis resolution
    gran = 50

    # QNode that takes parameters a and b, as well as the observable, and parameter t for the
    # exponential of the effective generator
    qnode_grad = qml.QNode(circuit_with_opg, dev, interface="jax")

    qnode_grad_sample_ps = qml.QNode(
        circuit_with_opg, dev_sample, interface="jax", diff_method="parameter-shift"
    )

    # Gradient computed by differentiating exp(Omega * t)
    grad_fn_generator = jax.jit(jax.grad(qnode_grad, argnums=2), static_argnums=3)
    # Gradient computed via automatic differentiation
    grad_fn_sample_ps = jax.jacobian(qnode_grad_sample_ps, argnums=2)

    # Fixed values of b for which to compute the gradient values
    b_lines = [0.5, 1.0, 2.0]
    # grid for a-axis
    a_grid = np.linspace(0, np.pi + 0.001, gran)

    grads_a_exact = evaluate_on_grid(
        grad_fn_generator, a_grid, b_lines, jnp.array(0.0), observable=observable
    )
    grads_a_sample_ps_mean, grads_a_sample_ps_std = evaluate_on_grid(
        grad_fn_sample_ps, a_grid, b_lines, jnp.array(0.0), observable=observable, sampled=True
    )

    plt.style.use("science")
    plt.rcParams.update({"font.size": 15})
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(5, 4)
    for i in range(3):
        axs.plot(a_grid, grads_a_exact[:, i], lw=0.7, ls="--", zorder=4, alpha=1.0, c="k")[0]
        axs.errorbar(
            a_grid,
            grads_a_sample_ps_mean[:, i],
            yerr=grads_a_sample_ps_std[:, i],
            label=rf"$b = ${b_lines[i]:1.1f}",
            lw=2,
            alpha=0.8,
            zorder=0,
        )

    axs.legend()
    axs.set_xlabel("$a$")
    axs.set_ylabel(r"$\frac{\partial C(\boldmath{\theta})}{\partial a}$")
    plt.tight_layout()
    if not os.path.exists("figures"):
        os.makedirs("figures")
    fig.savefig("./figures/sampled_grad.pdf")
    plt.show()


def parshift_vs_finitediff(shots):
    np.random.seed(21207)
    nqubits = 1
    dev = qml.device("default.qubit", wires=nqubits)
    dev_sample = qml.device("default.qubit", wires=nqubits, shots=[1] * shots)
    observable = qml.Hermitian(get_random_observable(nqubits), wires=list(range(nqubits)))

    # a-axis resolution
    gran = 50
    # Finite difference spacing
    delta = 0.5

    # QNode that takes parameters a and b, as well as the observable, and parameter t for the
    # exponential of the effective generator
    qnode_grad = qml.QNode(circuit_with_opg, dev, interface="jax")

    qnode_grad_sample_ps = qml.QNode(
        circuit_with_opg, dev_sample, interface="jax", diff_method="parameter-shift"
    )
    # qnode_grad_sample_fd = qml.QNode(circuit_with_opg, dev_sample, interface="jax", diff_method="finite-diff", h=delta, approx_order=2)
    qnode_grad_sample_fd = qml.QNode(circuit, dev_sample, interface="jax", diff_method=None)

    # Gradient computed by differentiating exp(Omega * t)
    grad_fn_generator = jax.jit(jax.grad(qnode_grad, argnums=2), static_argnums=3)
    # Gradient computed via automatic differentiation
    grad_fn_sample_ps = jax.jacobian(qnode_grad_sample_ps, argnums=2)
    grad_fn_sample_fd = finite_diff_first(qnode_grad_sample_fd, dx=delta, argnums=0)

    # Fixed values of b for which to compute the gradient values
    b_lines = [0.5, 1.0, 2.0]
    # grid for a-axis
    a_grid = np.linspace(0, np.pi + 0.001, gran)

    grads_a_exact = evaluate_on_grid(
        grad_fn_generator, a_grid, b_lines, jnp.array(0.0), observable=observable
    )
    grads_a_sample_ps_mean, grads_a_sample_ps_std = evaluate_on_grid(
        grad_fn_sample_ps, a_grid, b_lines, jnp.array(0.0), observable=observable, sampled=True
    )
    grads_a_sample_fd_mean, grads_a_sample_fd_std = evaluate_on_grid(
        grad_fn_sample_fd, a_grid, b_lines, observable=observable, sampled=True
    )

    plt.style.use("science")
    plt.rcParams.update({"font.size": 13})
    fig, axs = plt.subplots(3, 1, figsize=(5, 8), gridspec_kw={"hspace": 0.0})
    labels = ["Parameter shift", "Finite difference"]
    colors = ["xkcd:bright blue", "xkcd:burnt orange"]
    for i, ax in enumerate(axs):
        plot = ax.plot(
            a_grid,
            grads_a_exact[:, i],
            lw=1.1,
            ls="-",
            zorder=3,
            alpha=0.7,
            c="k",
            label="" if i < 2 else "Exact",
        )[0]
        for j, (mean, std) in enumerate(
            [
                (grads_a_sample_ps_mean, grads_a_sample_ps_std),
                (grads_a_sample_fd_mean, grads_a_sample_fd_std),
            ]
        ):
            c = colors[j]
            mu, sigma = mean[:, i], std[:, i]
            ax.plot(a_grid, mu, lw=2.5, ls="--", zorder=1, label="" if i < 2 else labels[j], c=c)
            ax.plot(a_grid, mu + sigma, lw=1.0, ls="-", alpha=0.3, c=c, zorder=-1)
            ax.plot(a_grid, mu - sigma, lw=1.0, ls="-", alpha=0.3, c=c, zorder=-1)
            ax.fill_between(a_grid, mu - sigma, mu + sigma, color=c, alpha=0.2, zorder=-1)

        ax.text(
            0.05, 0.95, rf"$b = ${b_lines[i]:1.1f}", transform=ax.transAxes, verticalalignment="top"
        )
        ax.set_ylabel(r"$\frac{\partial C(\boldmath{\theta})}{\partial a}$")
        if i < 2:
            ax.set_xticks([])
        else:
            ax.legend(loc="lower left")
            ax.xaxis.set_ticks_position("bottom")
            ax.set_xlabel("$a$")
    plt.tight_layout()
    if not os.path.exists("figures"):
        os.makedirs("figures")
    fig.savefig("./figures/parshift_vs_finitediff.pdf")
    plt.show()


if __name__ == "__main__":
    shots = 100
    exact_grad()
    sampled_grad(shots)
    parshift_vs_finitediff(shots)
