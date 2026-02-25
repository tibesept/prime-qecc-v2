import os
import sys

# Ensure src is in python path for tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import mpmath
from data_loader import RiemannZerosLoader


def test_zeros_loading():
    """
    Test loading zeros from the loader.
    """
    loader = RiemannZerosLoader()
    gammas = loader.load_odlyzko(num_zeros=10, dps=50)
    
    assert len(gammas) == 10
    
    # Check typing
    assert all(isinstance(g, mpmath.mpf) for g in gammas)
    
    # Check verification explicitly
    assert loader.verify_first_five(gammas)


if __name__ == "__main__":
    test_zeros_loading()
    print("✓ All data loader tests passed.")
