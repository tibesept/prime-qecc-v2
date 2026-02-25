#!/usr/bin/env python3

import os
import sys

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import mpmath
from data_loader import RiemannZerosLoader
from weil_functional import WeilFunctional
from bruhat_tits import BruhatTitsTree
from connection import WeilGraphConnection
from dashboard import create_dashboard, plot_weil_components

def main():
    print("=" * 70)
    print("PRIME NUMBER QECC: TOY MODEL")
    print("=" * 70)

    # Set precision
    mpmath.mp.dps = 50
    print(f"Precision set to {mpmath.mp.dps} decimal places.\n")

    # Step 1: Load Zeros
    print("[STEP 1] Loading Riemann zeros...")
    loader = RiemannZerosLoader()
    gammas = loader.load_odlyzko(num_zeros=10000)
    if not gammas:
        print("Failed to load zeros. Exiting.")
        return
    loader.verify_first_five(gammas)
    print(f"✓ Loaded {len(gammas)} zeros.\n")

    # Step 2: Compute Weil Functional
    print("[STEP 2] Computing Weil functional...")
    # Using 500 primes for a fast demonstration
    w_total, components = WeilFunctional.compute(gammas, sigma=mpmath.mpf(0.1), num_primes=500, verbose=True)

    # Step 3 and 4: Run Experiments (Tree is built inside)
    print("\n[STEP 3 & 4] Running Zeta Resonance Experiments...")
    results_weil = WeilGraphConnection.experiment_robustness(
        gammas, None, sigma_values=[1.0, 1.5, 2.0]
    )

    results_graph = WeilGraphConnection.experiment_graph_weight_assignment(
        gammas, sigma=1.0, num_primes=20
    )

    tree_broken = results_graph['tree_broken']
    tree_healthy = results_graph['tree_healthy']
    resonance_prime = results_graph['resonance_prime']

    # Step 5: Visualizations
    print("\n[STEP 5] Creating visualizations...")
    create_dashboard(results_weil, results_graph, output_file="../dashboard.html")
    plot_weil_components(components, output_file="../weil_components.html")
    
    # Save trees
    tree_healthy.visualize(f"../tree_healthy_p{resonance_prime}.html")
    tree_broken.visualize(f"../tree_broken_p{resonance_prime}.html")

    print("\n" + "=" * 70)
    print("✓ EXPERIMENT COMPLETE")
    print("=" * 70)
    print("\nGenerated files (in project root):")
    print("  - dashboard.html")
    print("  - weil_components.html")
    print(f"  - tree_healthy_p{resonance_prime}.html")
    print(f"  - tree_broken_p{resonance_prime}.html")
    print("\nNext steps:")
    print("  1. Open dashboard.html in your browser")
    print("  2. Check if W(f) >= 0 for all test functions (RH verification)")
    print("=" * 70)

if __name__ == "__main__":
    main()
