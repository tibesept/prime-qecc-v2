import math
from typing import Dict, List, Optional

import mpmath

from bruhat_tits import BruhatTitsTree
from tensor_network import PrimeFunctionalTransfer, TensorDeformationAnsatz
from weil_functional import WeilFunctional
from weil_primes import PrimesContribution


class WeilGraphConnection:
    """
    Separates number-theory experiments from operator-level tensor diagnostics.
    """

    @staticmethod
    def calculate_required_primes(sigma: float, target_epsilon: float = 1e-12) -> int:
        ln_eps = math.log(target_epsilon)
        x = -(sigma**2) + math.sqrt((sigma**4) - 4 * (sigma**2) * ln_eps)
        p_max = math.exp(x)
        approx_count = int(1.2 * (p_max / math.log(p_max))) if p_max > 2 else 10
        return max(10, min(approx_count, 3000))

    @staticmethod
    def _positive_imaginary_rhos(
        gammas: List[mpmath.mpf], num_zeros: int, broken_zeros: Optional[Dict[int, complex]] = None
    ) -> List[complex]:
        if broken_zeros is None:
            broken_zeros = {}

        rhos: List[complex] = []
        for i, gamma in enumerate(gammas[:num_zeros]):
            if i in broken_zeros:
                rhos.append(complex(broken_zeros[i]))
            else:
                rhos.append(complex(0.5 + 1j * float(gamma)))
        return rhos

    @staticmethod
    def experiment_robustness(
        gammas: List[mpmath.mpf], sigma_values: Optional[List[float]] = None
    ) -> Dict:
        """
        Experiment 1 (number theory): evaluate spectral positivity on healthy zeros.
        """
        if sigma_values is None:
            sigma_values = [0.3, 0.5, 0.8, 1.0, 1.2]

        results = {
            "sigma_values": [],
            "w_values": [],
            "prime_counts": [],
            "all_positive": True,
            "min_w": None,
            "negative_sigmas": [],
        }

        min_w = None
        for sigma in sigma_values:
            allocated_primes = WeilGraphConnection.calculate_required_primes(sigma)
            w_total, components = WeilFunctional.compute(
                gammas,
                sigma=mpmath.mpf(sigma),
                num_primes=allocated_primes,
                verbose=False,
            )
            w_float = float(components["W_zeros"])

            results["sigma_values"].append(float(sigma))
            results["w_values"].append(w_float)
            results["prime_counts"].append(int(allocated_primes))

            if w_float < -1e-10:
                results["all_positive"] = False
                results["negative_sigmas"].append(float(sigma))

            if min_w is None or w_float < min_w:
                min_w = w_float

        results["min_w"] = min_w
        return results

    @staticmethod
    def _compute_prime_resonance_deltas(
        gammas: List[mpmath.mpf], sigma: float = 1.0, num_primes: int = 200, zero_shift: complex = 0.1j
    ) -> Dict[int, Dict[str, float]]:
        _, components_ideal = WeilFunctional.compute(
            gammas, sigma=sigma, num_primes=num_primes, verbose=False
        )
        p_ideal = components_ideal["p_contributions"]

        gamma_1_ideal = mpmath.mpf(gammas[0])
        gamma_1_broken = gamma_1_ideal + mpmath.mpc(zero_shift)

        sigma_mpf = mpmath.mpf(sigma)

        def f(u):
            u = mpmath.mpf(u)
            return mpmath.exp(-(u / (2 * sigma_mpf)) ** 2)

        results = {}
        for p, w_ideal in p_ideal.items():
            p_mpf = mpmath.mpf(p)
            log_p = mpmath.log(p_mpf)

            wave_ideal = p_mpf ** (mpmath.j * gamma_1_ideal)
            wave_broken = p_mpf ** (mpmath.j * gamma_1_broken)
            delta_p_complex = (wave_broken - wave_ideal) * (log_p / mpmath.sqrt(p_mpf)) * f(log_p)
            delta_p = float(mpmath.re(delta_p_complex))
            w_broken = float(w_ideal + delta_p)

            results[p] = {
                "w_ideal": float(w_ideal),
                "w_broken": w_broken,
                "delta": delta_p,
            }

        return results

    @staticmethod
    def experiment_graph_weight_assignment(
        gammas: List[mpmath.mpf], sigma: float = 1.0, num_primes: int = 200
    ) -> Dict:
        """
        Experiment 2 (geometry proxy): map synthetic per-prime perturbation to edge weights.
        """
        prime_data = WeilGraphConnection._compute_prime_resonance_deltas(
            gammas, sigma=sigma, num_primes=num_primes
        )

        resonance_prime = max(prime_data, key=lambda p: abs(prime_data[p]["delta"]))
        healthy_value = prime_data[resonance_prime]["w_ideal"]
        broken_value = prime_data[resonance_prime]["w_broken"]

        tree_healthy = BruhatTitsTree(p=resonance_prime, depth=3)
        tree_healthy.assign_edge_weights_from_weil(healthy_value)

        tree_broken = BruhatTitsTree(p=resonance_prime, depth=3)
        tree_broken.assign_edge_weights_from_weil(broken_value)

        return {
            "prime_data": prime_data,
            "resonance_prime": resonance_prime,
            "tree_healthy": tree_healthy,
            "tree_broken": tree_broken,
            "healthy_negative_edge_fraction": tree_healthy.measure_negative_edge_fraction(),
            "broken_negative_edge_fraction": tree_broken.measure_negative_edge_fraction(),
        }

    @staticmethod
    def experiment_weil_positivity_criterion(
        gammas: List[mpmath.mpf], bandwidth: float = 0.5, shift_values: Optional[List[float]] = None
    ) -> Dict:
        """
        Experiment 3 (number theory): probes Weil positivity under synthetic RH violations.
        """
        if shift_values is None:
            shift_values = [0.01, 0.05, 0.1, 0.2]

        gamma_1 = mpmath.im(mpmath.zetazero(1))
        _, f_hat, f_hat_complex = WeilFunctional.resonant_testfunc(gamma_1, bandwidth)

        w_healthy = float(WeilFunctional.compute_spectral_sum(gammas, f_hat))
        scan_results = []
        for delta in shift_values:
            rho_broken = mpmath.mpc(0.5 - delta, float(gamma_1))
            w_broken = float(
                WeilFunctional.compute_spectral_sum(
                    gammas, f_hat_complex, broken_zeros={0: rho_broken}
                )
            )
            scan_results.append(
                {
                    "delta": float(delta),
                    "rho_broken": str(rho_broken),
                    "W_broken": w_broken,
                    "is_violation": bool(w_broken < -1e-10),
                }
            )

        worst = min(scan_results, key=lambda x: x["W_broken"])
        return {
            "gamma_target": float(gamma_1),
            "bandwidth": float(bandwidth),
            "W_healthy": w_healthy,
            "scan_results": scan_results,
            "worst_case": worst,
            "any_violation": any(item["is_violation"] for item in scan_results),
        }

    @staticmethod
    def experiment_tensor_deformation_ansatz(
        delta_values: Optional[List[float]] = None,
        dim_in: int = 2,
        dim_out: int = 4,
        alpha: float = 1.0,
        seed: int = 7,
    ) -> Dict:
        """
        Experiment 4 (operator model): diagnostics for T_delta = exp(-alpha*delta*K) T0.
        """
        if delta_values is None:
            delta_values = [0.0, 0.01, 0.05, 0.1, 0.2]

        ansatz = TensorDeformationAnsatz(
            dim_in=dim_in, dim_out=dim_out, alpha=alpha, seed=seed
        )
        diagnostics = [ansatz.diagnostics(delta) for delta in delta_values]

        return {
            "dim_in": dim_in,
            "dim_out": dim_out,
            "alpha": float(alpha),
            "seed": seed,
            "diagnostics": diagnostics,
        }

    @staticmethod
    def experiment_intrinsic_transfer_operator(
        gammas: List[mpmath.mpf],
        bandwidth: float = 0.25,
        num_primes: int = 20,
        num_zeros: int = 60,
        shift_values: Optional[List[float]] = None,
    ) -> Dict:
        """
        Experiment 4 (intrinsic operator test): builds a prime-space transfer kernel
        directly from the functional-equation pairing rho <-> 1-rho and checks whether
        the resulting Choi candidate remains Hermitian PSD under synthetic RH violation.
        """
        if shift_values is None:
            shift_values = [0.1, 0.2, 0.4, 0.8, 1.2]

        primes = PrimesContribution.first_n_primes(num_primes)
        transfer = PrimeFunctionalTransfer(primes, bandwidth=bandwidth)

        healthy_rhos = WeilGraphConnection._positive_imaginary_rhos(gammas, num_zeros=num_zeros)
        healthy = transfer.diagnostics_from_rhos(healthy_rhos)

        gamma_1 = float(mpmath.im(mpmath.zetazero(1)))
        scan_results = []
        for delta in shift_values:
            broken_rhos = WeilGraphConnection._positive_imaginary_rhos(
                gammas,
                num_zeros=num_zeros,
                broken_zeros={0: complex(0.5 - delta + 1j * gamma_1)},
            )
            diagnostics = transfer.diagnostics_from_rhos(broken_rhos)
            diagnostics["delta"] = float(delta)
            diagnostics["rho_broken"] = str(complex(0.5 - delta + 1j * gamma_1))
            scan_results.append(diagnostics)

        return {
            "bandwidth": float(bandwidth),
            "num_primes": int(num_primes),
            "num_zeros": int(num_zeros),
            "primes": primes,
            "healthy": healthy,
            "scan_results": scan_results,
            "first_failure": next((r for r in scan_results if not r["cp_candidate_ok"]), None),
        }

    @staticmethod
    def experiment_coupled_ansatz(
        gammas: List[mpmath.mpf],
        sigma: float = 1.0,
        num_primes: int = 200,
        top_k: int = 10,
        alpha: float = 1.0,
        seed: int = 7,
    ) -> Dict:
        """
        Experiment 5 (coupled ansatz): maps normalized prime anomaly magnitude
        to tensor deformation parameters. This is an imposed dictionary, not a derivation.
        """
        prime_data = WeilGraphConnection._compute_prime_resonance_deltas(
            gammas, sigma=sigma, num_primes=num_primes
        )
        sorted_primes = sorted(prime_data.keys(), key=lambda p: abs(prime_data[p]["delta"]), reverse=True)
        selected = sorted_primes[:top_k]

        max_abs_delta = max(abs(prime_data[p]["delta"]) for p in selected) if selected else 1.0
        max_abs_delta = max(max_abs_delta, 1e-14)

        ansatz = TensorDeformationAnsatz(dim_in=2, dim_out=4, alpha=alpha, seed=seed)

        diagnostics = []
        for p in selected:
            raw_delta = prime_data[p]["delta"]
            normalized_delta = abs(raw_delta) / max_abs_delta
            tensor_metrics = ansatz.diagnostics(normalized_delta)
            diagnostics.append(
                {
                    "prime": int(p),
                    "raw_delta": float(raw_delta),
                    "normalized_delta": float(normalized_delta),
                    "w_ideal": float(prime_data[p]["w_ideal"]),
                    "w_broken": float(prime_data[p]["w_broken"]),
                    "tensor": tensor_metrics,
                }
            )

        return {
            "sigma": float(sigma),
            "num_primes": int(num_primes),
            "top_k": int(top_k),
            "alpha": float(alpha),
            "prime_diagnostics": diagnostics,
        }
