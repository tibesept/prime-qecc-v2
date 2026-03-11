import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import mpmath

from connection import WeilGraphConnection
from data_loader import RiemannZerosLoader
from weil_functional import WeilFunctional


def test_baseline_positive():
    mpmath.mp.dps = 50
    loader = RiemannZerosLoader()
    gammas = loader.load_odlyzko(num_zeros=200, dps=50)

    for sigma in [0.5, 1.0, 1.5]:
        _, components = WeilFunctional.compute(gammas, sigma=sigma, num_primes=120, verbose=False)
        assert float(components["W_zeros"]) >= -1e-10


def test_resonant_testfunc_nonnegative_on_reals():
    mpmath.mp.dps = 40
    gamma_1 = mpmath.im(mpmath.zetazero(1))
    _, f_hat, _ = WeilFunctional.resonant_testfunc(gamma_1, bandwidth=0.3)

    for t in [mpmath.mpf(v) for v in range(-50, 51)]:
        val = f_hat(t)
        assert mpmath.re(val) >= -1e-15


def test_broken_zero_quartet_matches_healthy_on_critical_line():
    mpmath.mp.dps = 50
    loader = RiemannZerosLoader()
    gammas = loader.load_odlyzko(num_zeros=200, dps=50)

    gamma_1 = mpmath.im(mpmath.zetazero(1))
    _, f_hat, f_hat_complex = WeilFunctional.resonant_testfunc(gamma_1, bandwidth=0.5)

    w_healthy = WeilFunctional.compute_spectral_sum(gammas, f_hat)
    rho_on_line = mpmath.mpc(0.5, float(gamma_1))
    w_quartet = WeilFunctional.compute_spectral_sum(gammas, f_hat_complex, broken_zeros={0: rho_on_line})

    assert abs(w_healthy - w_quartet) < mpmath.mpf("1e-8")


def test_weil_positivity_experiment_is_well_formed():
    mpmath.mp.dps = 50
    loader = RiemannZerosLoader()
    gammas = loader.load_odlyzko(num_zeros=200, dps=50)

    result = WeilGraphConnection.experiment_weil_positivity_criterion(
        gammas, bandwidth=0.5, shift_values=[0.01, 0.05, 0.1, 0.2]
    )
    assert result["W_healthy"] >= -1e-10
    assert len(result["scan_results"]) == 4
    deviations = [abs(item["W_broken"] - result["W_healthy"]) for item in result["scan_results"]]
    assert max(deviations) > 1e-6
