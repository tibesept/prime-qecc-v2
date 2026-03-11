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
    info = loader.get_last_load_info()
    
    assert len(gammas) == 10
    
    # Check typing
    assert all(isinstance(g, mpmath.mpf) for g in gammas)
    
    # Precision metadata should be available
    assert "precision_digits" in info
    assert int(info["precision_digits"]) >= 30
    assert str(info["source"]).startswith("mpmath")

    # Source-aware verification should pass
    assert loader.verify_first_five(gammas, source_info=info)

    # Very strict tolerance should fail for finite-precision reference constants
    assert not loader.verify_first_five(gammas, tolerance=mpmath.mpf("1e-40"), source_info=info)


if __name__ == "__main__":
    test_zeros_loading()
    print("✓ All data loader tests passed.")
