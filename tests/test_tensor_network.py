import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import mpmath

from connection import WeilGraphConnection
from data_loader import RiemannZerosLoader
from tensor_network import PrimeFunctionalTransfer, TensorDeformationAnsatz


def test_tensor_ansatz_monotone_defects_and_cp():
    model = TensorDeformationAnsatz(dim_in=2, dim_out=4, alpha=1.0, seed=11)
    deltas = [0.0, 0.01, 0.05, 0.1, 0.2]
    diagnostics = [model.diagnostics(d) for d in deltas]

    assert diagnostics[0]["isometry_defect"] < 1e-10
    assert diagnostics[0]["trace_preservation_defect"] < 1e-10

    iso_values = [d["isometry_defect"] for d in diagnostics]
    tp_values = [d["trace_preservation_defect"] for d in diagnostics]

    for i in range(1, len(iso_values)):
        assert iso_values[i] + 1e-12 >= iso_values[i - 1]
        assert tp_values[i] + 1e-12 >= tp_values[i - 1]

    for d in diagnostics:
        assert d["cp_ok"]
        assert d["choi_min_eigenvalue"] >= -1e-10
        assert d["contraction_norm"] <= 1.0 + 1e-10


def test_coupled_ansatz_outputs_operator_metrics():
    loader = RiemannZerosLoader()
    gammas = loader.load_odlyzko(num_zeros=200, dps=50)

    coupled = WeilGraphConnection.experiment_coupled_ansatz(
        gammas, sigma=1.0, num_primes=60, top_k=6, alpha=1.0, seed=5
    )
    rows = coupled["prime_diagnostics"]
    assert len(rows) == 6
    for row in rows:
        assert 0.0 <= row["normalized_delta"] <= 1.0
        assert row["tensor"]["cp_ok"]


def test_intrinsic_transfer_kernel_detects_invalid_cp_candidate():
    loader = RiemannZerosLoader()
    gammas = loader.load_odlyzko(num_zeros=120, dps=50)

    result = WeilGraphConnection.experiment_intrinsic_transfer_operator(
        gammas, bandwidth=0.25, num_primes=20, num_zeros=60, shift_values=[0.1, 0.2, 0.4, 0.8, 1.2]
    )

    assert result["healthy"]["cp_candidate_ok"]
    assert result["healthy"]["state_channel_candidate_ok"]
    assert result["healthy"]["hermiticity_defect"] < 1e-8

    first_broken = result["scan_results"][0]
    assert first_broken["hermiticity_defect"] < 1e-8
    assert first_broken["cp_candidate_ok"]
    assert first_broken["state_channel_candidate_ok"]

    worst = result["scan_results"][-1]
    assert worst["hermiticity_defect"] < 1e-8
    assert worst["min_hermitian_eigenvalue"] < 0
    assert worst["failure_mode"] == "negative_eigenvalue"
    assert not worst["state_channel_candidate_ok"]


def test_intrinsic_transfer_kernel_completes_upper_half_orbit():
    gamma_1 = mpmath.mpf("14.13472514173469379045725198356247027078")
    broken = WeilGraphConnection._positive_imaginary_rhos(
        [gamma_1, 21.02203963877155499262847959389690277733],
        num_zeros=2,
        broken_zeros={0: mpmath.mpc("0.3", gamma_1)},
    )

    assert len(broken) == 3
    assert any(abs(rho - mpmath.mpc("0.3", gamma_1)) < mpmath.mpf("1e-30") for rho in broken)
    assert any(abs(rho - mpmath.mpc("0.7", gamma_1)) < mpmath.mpf("1e-30") for rho in broken)


def test_intrinsic_upper_half_spectrum_retains_mpmath_precision():
    mpmath.mp.dps = 80
    gamma_1 = mpmath.mpf("14.134725141734693790457251983562470270784257115699243175685567")
    mpc_type = type(mpmath.mpc(0))

    healthy = WeilGraphConnection._positive_imaginary_rhos([gamma_1], num_zeros=1)
    broken = WeilGraphConnection._positive_imaginary_rhos(
        [gamma_1],
        num_zeros=1,
        broken_zeros={0: mpmath.mpc("0.3", gamma_1)},
    )

    assert all(isinstance(rho, mpc_type) for rho in healthy + broken)
    assert abs(mpmath.im(healthy[0]) - gamma_1) < mpmath.mpf("1e-40")
    assert abs(mpmath.re(broken[0]) - mpmath.mpf("0.3")) < mpmath.mpf("1e-40")
    assert abs(mpmath.im(broken[0]) - gamma_1) < mpmath.mpf("1e-40")
    assert abs(mpmath.re(broken[1]) - mpmath.mpf("0.7")) < mpmath.mpf("1e-40")
    assert abs(mpmath.im(broken[1]) - gamma_1) < mpmath.mpf("1e-40")


def test_prime_functional_transfer_healthy_kernel_is_gram_like():
    transfer = PrimeFunctionalTransfer(primes=[2, 3, 5, 7, 11, 13], bandwidth=0.25)
    rhos = [0.5 + 1j * g for g in [14.1347251417, 21.0220396388, 25.0108575801]]
    diag = transfer.diagnostics_from_rhos(rhos)
    assert diag["cp_candidate_ok"]
    assert diag["state_channel_candidate_ok"]
    assert diag["unit_trace_error"] < 1e-12
