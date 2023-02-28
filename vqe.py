"""Functions to run the VQE example in the paper and create the corresponding plots."""
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pennylane as qml
import numpy as np
from tqdm import tqdm

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


def apply_op_in_layers(theta, Op, depth, nqubits, fix_pars_per_layer=True):
    """Apply a (callable) operation in layers.

    Args:
        theta (tensor_like): The arguments passed to the operations. The expected shape
            is ``(depth, k, num_params_op)``, where ``k`` is determined by ``fix_pars_per_layer``
            and ``num_params_op`` is the number of paramters each operation takes.
        Op (callable): The operation to apply
        depth (int): The number of layers to apply
        fix_pars_per_layer (bool): Whether or not all operations applied in parallel share the
            same parameters. If ``True``, the dimension ``k`` in the shape of ``theta`` is set to
            two, otherwise it is set to ``nqubits``.

    """

    for d in range(depth):
        # Even-odd qubit pairs
        idx = 0
        for i in range(0, nqubits, 2):
            Op(theta[d, idx], wires=[i, i + 1])
            if i == nqubits - 2:
                idx += 1
            elif not fix_pars_per_layer:
                idx += 1
        # Odd-even qubit pairs
        for i in range(1, nqubits, 2):
            Op(theta[d, idx], wires=[i, (i + 1) % nqubits])
            if not fix_pars_per_layer:
                idx += 1


def gate_decomp_su4(params, wires):
    """Apply a sequence of 15 single-qubit rotations and three ``CNOT`` gates to
    compose an arbitrary SU(4) operation.

    Args:
        params (tensor_like): Parameters for single-qubit rotations, expected to have shape (15,)
        wires (list[int]): Wires on which to apply the gate sequence

    See Theo. 5 in https://arxiv.org/pdf/quant-ph/0308006.pdf for more details.
    """
    i, j = wires
    # Single U(2) parameterization on qubit 1
    qml.RY(params[0], wires=i)
    qml.RX(params[1], wires=i)
    qml.RY(params[2], wires=i)
    # Single U(2) parameterization on qubit 2
    qml.RY(params[3], wires=j)
    qml.RX(params[4], wires=j)
    qml.RY(params[5], wires=j)
    # CNOT with control on qubit 2
    qml.CNOT(wires=[j, i])
    # Rz and Ry gate
    qml.RZ(params[6], wires=i)
    qml.RY(params[7], wires=j)
    # CNOT with control on qubit 1
    qml.CNOT(wires=[i, j])
    # Ry gate on qubit 2
    qml.RY(params[8], wires=j)
    # CNOT with control on qubit 2
    qml.CNOT(wires=[j, i])
    # Single U(2) parameterization on qubit 1
    qml.RY(params[9], wires=i)
    qml.RX(params[10], wires=i)
    qml.RY(params[11], wires=i)
    # Single U(2) parameterization on qubit 2
    qml.RY(params[12], wires=j)
    qml.RX(params[13], wires=j)
    qml.RY(params[14], wires=j)


def make_observable(wires, seed):
    """Generate a random Hermitian matrix from the Gaussian Unitary Ensemble (GUE), as well
    as a ``qml.Hermitian`` observable.

    Args:
        wires (list[int]): Wires on which the observable should be measured.
        seed (int): Seed for random matrix sampling.

    Returns:
        Hermitian: The Hermitian observable
        tensor_like: The matrix of the observable

    For ``n`` entries in ``wires``, the returned observable matrix has size ``(2**n, 2**n)``.
    """
    np.random.seed(seed)
    num_wires = len(wires)
    d = 2**num_wires
    # Random normal complex-valued matrix
    mat = np.random.randn(d, d) + 1j * np.random.randn(d, d)
    # Make the random matrix Hermitian and divide by two to match the GUE.
    observable_matrix = (mat + mat.conj().T) / 2
    return qml.Hermitian(observable_matrix, wires=wires), observable_matrix


def vqe(
    seed_list,
    nqubits=None,
    depth=None,
    opname=None,
    fix_pars_per_layer=True,
    max_steps=None,
    learning_rate=None,
    data_header="",
):
    """Run the VQE numerical experiment.

    Args:
        seed_list (list[int]): Sequence of randomness seeds for the observable creation.
            One VQE experiment is run for each seed
        nqubits (int): Number of qubits
        depth (int): Number of layers of operations to use in the circuit ansatz
        opname (str): Which operation to use in the circuit ansatz, may be ``"su4"`` for
            out proposed SU(N) parametrization on two qubits, or ``"decomp"`` for the
            gate sequence created by ``gate_decomp_su4``.
        fix_pars_per_layer (bool): Whether or not all operations applied in parallel share the
            same parameters.
        max_steps (int): The number of steps for which to run the gradient descent
            optimizer of the VQE.
        learning_rate (float): The learning rate of the gradient descent optimizer.
        data_header (str): Subdirectory of ``./data`` to save data to

    This function executes the full VQE workflow, including

      - the generation of the observable and storage of its key energy information

      - the setup of the cost function by composing the chosen operation in a fabric of layers
        and measuring the observable expectation value afterwards

      - the optimization of the cost function for the indicated number of steps, using vanilla
        gradient descent with a fixed learning rate

      - the storage of the optimization curves on disk

    """
    data_path = f"./data/{data_header}/{nqubits}/{depth}/{opname}/"
    # Set the differentiation method to use backpropagation
    diff_method = "backprop"
    # Use the DefaultQubit PennyLane device
    dev_type = "default.qubit"
    # Choose the operation to use in the circuit ansatz
    if opname == "su4":
        Op = qml.SpecialUnitary
    elif opname == "decomp":
        Op = gate_decomp_su4
    elif opname is None:
        raise ValueError

    dev = qml.device(dev_type, wires=nqubits)

    for seed in tqdm(seed_list):
        observable, observable_matrix = make_observable(dev.wires, seed)

        @qml.qnode(dev, interface="jax", diff_method=diff_method)
        def cost_function(params):
            """Cost function of the VQE."""
            apply_op_in_layers(params, Op, depth, nqubits, fix_pars_per_layer)
            return qml.expval(observable)

        grad_function = jax.jit(jax.grad(cost_function))
        cost_function = jax.jit(cost_function)

        # Store the ground state and maximal energy of the created observable on disk
        if not os.path.exists(data_path + f"gs_energy_{seed}.npy"):
            energies = np.linalg.eigvalsh(observable_matrix)
            gs_energy = energies[0]
            max_energy = energies[-1]
            np.save(data_path + f"gs_energy_{seed}.npy", gs_energy)
            np.save(data_path + f"max_energy_{seed}.npy", max_energy)

        cost_path = data_path + f"cost_{seed}.npy"

        # Check whether the optimization curves already exist on disk
        if not os.path.exists(cost_path):
            if fix_pars_per_layer:
                # Parameter shape: depth x (2 layers) x (15 = 4**2-1)
                shape = (depth, 2, 15)
            else:
                # Parameter shape: depth x (nqubits operations) x (15 = 4**2-1)
                shape = (depth, nqubits, 15)
            # Create initial parameters
            params = jax.numpy.zeros(shape)

            cost_history = []
            for _ in tqdm(range(max_steps)):
                # Record old cost
                cost_history.append(cost_function(params))
                # Make step
                params = params - learning_rate * grad_function(params)

            # Record final cost
            cost_history.append(cost_function(params))
            # Save cost to disk
            np.save(cost_path, np.array(cost_history))
        else:
            # Load cost from disk
            cost_history = np.load(cost_path)
            print(f"{cost_path} exists, loaded data!")


def load_data(nqubits, depth, seed_list, max_steps, data_header):
    """Load the VQE optimization curve data and process it into
    relative energy errors.

    Args:
        nqubits (int): Number of qubits
        depth (list[int]): All depth to show
        seed_list (list[int]): The randomness seeds for all runs
        max_steps (int): The number of optimization steps
        data_header (str): Subdirectory of ``./data`` to load data from

    Returns:
        tensor_like: Data for gate sequence operation
        tensor_like: Data for SU(N) gate

    """

    data_path = f"./data/{data_header}/{nqubits}/{depth}/"
    decomp = np.zeros((len(seed_list), max_steps + 1))
    su4 = np.zeros((len(seed_list), max_steps + 1))

    for i, seed in enumerate(seed_list):
        gs_energy = np.load(f"{data_path}decomp/gs_energy_{seed}.npy")
        max_energy = np.load(f"{data_path}su4/max_energy_{seed}.npy")
        spectrum_width = max_energy - gs_energy
        try:
            decomp[i] = (np.load(f"{data_path}decomp/cost_{seed}.npy") - gs_energy) / spectrum_width
        except FileNotFoundError:
            # Just skip files that were not found
            print(f"File {data_path}decomp/cost_{seed}.npy not found, skipping...")
        try:
            su4[i] = (np.load(f"{data_path}su4/cost_{seed}.npy") - gs_energy) / spectrum_width
        except FileNotFoundError:
            # Just skip files that were not found
            print(f"File {data_path}su4/cost_{seed}.npy not found, skipping...")

    return decomp, su4


def plot_performance_diff(nqubits, depths, seed_list, max_steps, data_header):
    """Plot the difference in relative energy error between the VQE optimization curves
    for the SU(N) gate and the local gate sequence operation.

    Args:
        nqubits (int): Number of qubits
        depths (list[int]): All depth to show
        seed_list (list[int]): The randomness seeds for all runs
        max_steps (int): The number of optimization steps
        data_header (str): Subdirectory of ``./data`` to load data from

    """
    # Plot setup
    plt.style.use("science")
    plt.rcParams.update({"font.size": 15})
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(5, 4)

    # Add circuit diagram
    arr_image = mpimg.imread("./figures/brick.png", format="png")
    axin = fig.add_axes([0.3, 0.5, 0.4, 0.4])
    axin.matshow(arr_image)
    axin.axis("off")
    colors = ["xkcd:grass green", "xkcd:pinkish red", "xkcd:bright blue"]

    for depth, c in zip(depths, colors):
        # Load data
        decomp, su4 = load_data(nqubits, depth, seed_list, max_steps, data_header)
        # Plot the difference in relative energy error, averaged over all seeds
        axs.plot(np.mean(su4 - decomp, axis=0), label=rf"$\ell$ = {depth}", linewidth=2, c=c)

    axs.legend(loc="lower right")
    axs.set_xlabel("Step")
    axs.set_ylabel(r"$\Delta \bar{E}$")
    plt.tight_layout()

    # Save plot
    fig.savefig(f"./figures/{nqubits}_qubit_comparison.pdf", dpi=300)
    plt.show()


def plot_optim_curves(nqubits, depth, seed_list, max_steps, data_header):
    """Plot the relative energy error optimization curves of the VQE runs.

    Args:
        nqubits (int): Number of qubits
        depth (int): Depth of the circuit ansatz
        seed_list (list[int]): Randomness seeds for all VQE runs
        max_steps (int): Number of optimization steps
        data_header (str): Subdirectory of ``./data`` to load data from
    """
    plt.style.use("science")
    plt.rcParams.update({"font.size": 15})
    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(5, 4)

    # Load data
    decomp, su4 = load_data(nqubits, depth, seed_list, max_steps, data_header)

    reds = plt.cm.get_cmap("Reds")
    blues = plt.cm.get_cmap("Blues")
    num_seeds = len(seed_list)
    for i in range(num_seeds):
        color_red = reds((1 + i) / num_seeds)
        color_blue = blues((1 + i) / num_seeds)
        labels = ["Decomp.", r"$\mathrm{SU}(N)$"] if i == num_seeds // 2 else ["", ""]
        axs.plot(decomp[i], linewidth=1, color=color_red, label=labels[0])
        axs.plot(su4[i], linewidth=1, color=color_blue, label=labels[1])
    axs.legend()
    axs.set_xlabel("Step")
    axs.set_ylabel(r"$\bar{E}$")
    plt.tight_layout()

    fig.savefig(f"./figures/{nqubits}_qubit_{depth}_trajectories.pdf")
    plt.show()
