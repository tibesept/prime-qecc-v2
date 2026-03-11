#!/usr/bin/env python3

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import mpmath

from connection import WeilGraphConnection
from dashboard import create_dashboard, plot_weil_components
from data_loader import RiemannZerosLoader
from weil_functional import WeilFunctional


def main():
    print("=" * 70)
    print("PRIME-QECC TOY MODEL")
    print("=" * 70)

    mpmath.mp.dps = 50
    print(f"Precision: dps={mpmath.mp.dps}\n")

    print("[STEP 1] Loading zeta zeros...")
    loader = RiemannZerosLoader()
    gammas = loader.load_odlyzko(num_zeros=500, dps=50)
    if not gammas:
        print("No zeros available. Exiting.")
        return

    source_info = loader.get_last_load_info()
    verified = loader.verify_first_five(gammas, source_info=source_info)
    if not verified:
        print("Warning: first-five verification failed for current source precision.")
    else:
        print("First-five verification passed.")
    print(f"Loaded {len(gammas)} zeros from source={source_info.get('source')}.\n")

    print("[STEP 2] Baseline Weil decomposition...")
    _, components = WeilFunctional.compute(gammas, sigma=mpmath.mpf(0.2), num_primes=300, verbose=True)

    print("\n[STEP 3] Robustness scan...")
    results_weil = WeilGraphConnection.experiment_robustness(gammas, sigma_values=[0.1, 0.2, 0.3, 0.4, 0.5])

    print("\n[STEP 4] Synthetic prime resonance deltas...")
    results_graph = WeilGraphConnection.experiment_graph_weight_assignment(gammas, sigma=1.0, num_primes=300)
    resonance_prime = results_graph["resonance_prime"]

    print("\n[STEP 5] Weil positivity criterion...")
    results_positivity = WeilGraphConnection.experiment_weil_positivity_criterion(
        gammas, bandwidth=0.5, shift_values=[0.01, 0.05, 0.1, 0.2]
    )

    print("\n[STEP 6] Intrinsic transfer-operator test...")
    results_intrinsic = WeilGraphConnection.experiment_intrinsic_transfer_operator(
        gammas, bandwidth=0.25, num_primes=20, num_zeros=60, shift_values=[0.1, 0.2, 0.4, 0.8, 1.2]
    )

    print("\n[STEP 7] Tensor deformation ansatz...")
    results_tensor = WeilGraphConnection.experiment_tensor_deformation_ansatz(
        delta_values=[0.0, 0.01, 0.05, 0.1, 0.2], dim_in=2, dim_out=4, alpha=1.0, seed=7
    )

    print("\n[STEP 8] Coupled ansatz (imposed dictionary)...")
    results_coupled = WeilGraphConnection.experiment_coupled_ansatz(
        gammas, sigma=1.0, num_primes=300, top_k=10, alpha=1.0, seed=7
    )

    print("\n[STEP 9] Writing visual outputs...")
    create_dashboard(
        results_weil,
        results_graph,
        results_positivity,
        results_intrinsic=results_intrinsic,
        results_tensor=results_tensor,
        results_coupled=results_coupled,
        output_file="../dashboard.html",
    )
    plot_weil_components(components, output_file="../weil_components.html")

    results_graph["tree_healthy"].visualize(f"../tree_healthy_p{resonance_prime}.html")
    results_graph["tree_broken"].visualize(f"../tree_broken_p{resonance_prime}.html")

    print("\n" + "=" * 70)
    print("RUN COMPLETE")
    print("=" * 70)
    print("Generated files:")
    print("  - dashboard.html")
    print("  - weil_components.html")
    print(f"  - tree_healthy_p{resonance_prime}.html")
    print(f"  - tree_broken_p{resonance_prime}.html")
    print("\nSummary:")
    print(f"  Weil healthy W(F): {results_positivity['W_healthy']:.6e}")
    print(f"  Weil worst broken W(F): {results_positivity['worst_case']['W_broken']:.6e}")
    if results_positivity["any_violation"]:
        print("  Positivity violations detected for synthetic off-line shifts.")
    else:
        print("  No positivity violation detected for the tested shift set.")

    d0 = results_tensor["diagnostics"][0]
    d_last = results_tensor["diagnostics"][-1]
    first_failure = results_intrinsic["first_failure"]
    if first_failure:
        print(
            "  Intrinsic kernel first state-channel failure at "
            f"delta={first_failure['delta']:.2f} "
            f"({first_failure['failure_mode']}, "
            f"min eig={first_failure['min_hermitian_eigenvalue']:.3e})"
        )
    else:
        print("  Intrinsic kernel remained a valid unit-trace state candidate on the tested shift set.")
    print(f"  Tensor isometry defect delta=0:   {d0['isometry_defect']:.3e}")
    print(f"  Tensor isometry defect delta=max: {d_last['isometry_defect']:.3e}")
    print("=" * 70)


if __name__ == "__main__":
    main()
