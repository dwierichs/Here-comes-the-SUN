## Here comes the SU(N): multivariate quantum gates and gradients

This repository contains the code to reproduce all results and numerical figures/plots
of the paper
["Here comes the SU(N): multivariate quantum gates and gradients"](arxiv.org/abs/2303.?????)

The repository is structured as follows:

- `requirements.txt`: All required Python packages to run the programs. We used Python 3.10. The file can be used via `pip install -r requirements.txt`
- `run_single_qubit.py`: Execute this file to produce the data and plots for the toy example on a single qubit (Figs. 3, 4, F1)
- `run_vqe.py`: Execute this file to produce the data and plots for the 10-qubit VQE optimization example (Figs. 7, 8)
- `single_qubit.py`: Additional functions used in `run_single_qubit.py`
- `vqe.py`: Additional functions used in `run_vqe.py`
- `figures/`: All produced figures are stored here
- `data/`: All prodcued data is stored here

In general, the programs are written such that pre-computed data is loaded, in order to allow quick changes to the figures
and to save computation efforts. If you want to reproduce the data from scratch (randomness seeds are fixed almost everywhere),
you may move or delete the data locally and execute `run_vqe.py` or `run_single_qubit.py`.


