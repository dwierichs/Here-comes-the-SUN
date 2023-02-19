"""Run the VQE example in the paper and create the corresponding plots."""
import os
from functools import partial
from itertools import product
from multiprocessing import Pool
from dill import dump
import scienceplots

from vqe import run_vqe, plot_a, plot_b

# Choose the following settings

#  These are the settings used for the paper
nruns = 50  # Number of VQE runs
max_steps = 10000  # number of optimization steps in each VQE
learning_rate = 1e-3  # Learning rate for gradient descent in the optimization
nqubits = 10  # Number of qubits
depths = [1, 3, 5]  # Number of layers in the circuit ansatz. The VQE is run for each depth
fix_pars_per_layer = False  # Whether to reuse parameters for all operations in a layer
RUN = False  # Whether to run the computation. If results are present, computations are skipped
PLOT = True  # Whether to create plots of the results
num_workers = 1  # Number of threads to use in parallel. Needs to be set machine-dependently

#  These are some settings with much lower computational cost, for illustration and testing
nruns = 4  # Number of VQE runs
max_steps = 1000  # number of optimization steps in each VQE
learning_rate = 1e-3  # Learning rate for gradient descent in the optimization
nqubits = 4  # Number of qubits
depths = [1, 2, 3]  # Number of layers in the circuit ansatz. The VQE is run for each depth
fix_pars_per_layer = False  # Whether to reuse parameters for all operations in a layer
RUN = True  # Whether to run the computation. If results are present, computations are skipped
PLOT = True  # Whether to create plots of the results
num_workers = 4  # Number of threads to use in parallel. Needs to be set machine-dependently

if __name__ == "__main__":
    # Directory name to save results to. They will be in f"./data/{data_header}/"
    data_header = f"production_iter{max_steps}_lr{learning_rate}"

    # Generate seeds (deterministically)
    runs_per_worker = nruns // num_workers
    seed_lists = [
        [i * 37 for i in range(j * runs_per_worker, (j + 1) * runs_per_worker)]
        for j in range(num_workers)
    ]

    # Store the global variables to allow for later investigation of settings
    global_config_path = f"./data/{data_header}/"
    if not os.path.exists(global_config_path):
        os.makedirs(global_config_path)
    with open(global_config_path + "globals.dill", "wb") as file:
        dump(globals(), file)

    # Run computation if requested. If results are present already, computations are skipped
    if RUN:
        for depth, opname in product(depths, ["su4", "decomp"]):
            # Set up path and create missing directories
            data_path = f"./data/{data_header}/{nqubits}/{depth}/{opname}/"
            if not os.path.exists(data_path):
                os.makedirs(data_path)

            # Mappable version of ``run_vqe``.
            func = partial(
                run_vqe,
                nqubits=nqubits,
                depth=depth,
                opname=opname,
                max_steps=max_steps,
                fix_pars_per_layer=fix_pars_per_layer,
                learning_rate=learning_rate,
                data_header=data_header,
            )
            # Map ``run_vqe`` across the partial seed lists for parallel execution
            with Pool(num_workers) as p:
                p.map(func, seed_lists)

    # Create plots if requested
    if PLOT:
        seed_list = sum(seed_lists, start=[])
        plot_a(nqubits, depths, seed_list, max_steps, data_header)
        plot_b(nqubits, max(depths), seed_list, max_steps, data_header)
