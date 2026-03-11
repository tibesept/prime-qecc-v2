# Prime-QECC Toy Model

This repository is a numerical sandbox connecting three topics:

- Weil explicit-formula computations (primes and zeta zeros)
- p-adic Bruhat-Tits tree geometry visualizations
- Finite-dimensional tensor deformation diagnostics

## Scope

This project is **not** a proof of the Riemann Hypothesis and does not derive an adelic holographic Hamiltonian.
It is a computational playground for testing whether a chosen coupling dictionary produces coherent numerical behavior.

Implemented components are intentionally separated:

1. Number-theory experiments:
- spectral-side Weil positivity scans under synthetic zero perturbations
- prime-side contributions in truncated explicit-formula sums

2. Geometry proxy:
- Bruhat-Tits trees with scalar edge-weight visualization
- no operator-level unitarity claim from tree edge signs

3. Intrinsic transfer-kernel test:
- prime-space transfer/Choi candidate built directly from the functional pairing `rho <-> 1-rho`
- healthy critical-line spectra produce an approximately Hermitian positive-type kernel
- synthetic off-line shifts can make that same kernel cease to be a valid CP candidate

4. Operator ansatz:
- a seeded isometric tensor `T0`
- deformed tensors `T_delta = exp(-alpha*delta*K) T0`
- diagnostics based on:
`||T_delta^dagger T_delta - I||`, Choi eigenvalues (CP check), trace-preservation defect

## What Is Not Implemented

- No full Bost-Connes C*-dynamical/KMS implementation
- No adelic Hilbert space construction
- No derivation that `beta = 1` is a QECC threshold
- No proof that Weil positivity is equivalent to tensor-network unitarity in this codebase
- No genuine holographic tensor network derived from number-theoretic dynamics

## Repository Layout

- `src/data_loader.py`: zero loading with source/precision metadata
- `src/weil_functional.py`: test functions and spectral-side evaluation
- `src/weil_primes.py`: prime-side contribution using first `n` primes
- `src/connection.py`: orchestrates separated experiments
- `src/tensor_network.py`: intrinsic transfer-kernel and tensor/channel diagnostics
- `src/bruhat_tits.py`: Bruhat-Tits tree construction and visualization helpers
- `src/dashboard.py`: Plotly dashboard output
- `src/main.py`: end-to-end run script
- `tests/`: unit tests

## Running

1. `python3 -m venv venv`
2. `source venv/bin/activate`
3. `pip install -r requirements.txt`
4. `cd src && python main.py`

Outputs are written in project root:

- `dashboard.html`
- `weil_components.html`
- `tree_healthy_p<prime>.html`
- `tree_broken_p<prime>.html`

## Theory Notes

`THEORY.md` describes the deformation model as an **ansatz**.
It is intentionally explicit about assumptions and limitations so that numerical outcomes are not over-interpreted.
