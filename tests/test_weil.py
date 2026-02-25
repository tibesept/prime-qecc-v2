import os
import sys

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import mpmath
from data_loader import RiemannZerosLoader
from weil_functional import WeilFunctional

def test_baseline():
    """
    On true zeros, W(f*f*) >= 0 for all acceptable test functions.
    We test our Gaussian test function at a few sigmas.
    """
    mpmath.mp.dps = 50
    loader = RiemannZerosLoader()
    # Let's load fewer zeros for faster testing, but enough for convergence
    gammas = loader.load_odlyzko(num_zeros=10000)

    for sigma in [1.0, 1.5, 2.0]:
        w_total, components = WeilFunctional.compute(gammas, sigma=sigma, num_primes=500, verbose=False)
        w_float = float(w_total)
        
        # Output result for debugging
        print(f"sigma={sigma}, W={w_float:.6e}")
        
        # Weil criterion positivity check (with numeric tolerance)
        assert w_total >= -1e-10, f"Baseline failed at sigma={sigma}: W = {w_float}"
        print(f"✓ sigma={sigma}: W = {w_float:.6e} >= 0")


if __name__ == "__main__":
    test_baseline()
    print("✓ All Weil functional tests passed.")
