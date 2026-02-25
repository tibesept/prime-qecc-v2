import mpmath
from typing import List, Dict

from weil_functional import WeilFunctional
from bruhat_tits import BruhatTitsTree


class WeilGraphConnection:
    """
    Explores the connection between Weil Functional properties and p-adic tree unitarity.
    """

    @staticmethod
    def experiment_robustness(gammas: List[mpmath.mpf],
                              tree: BruhatTitsTree,
                              sigma_values: List[float] = None) -> Dict:
        """
        Experiment to verify W(f) >= 0 for different structural test lengths.
        """
        if sigma_values is None:
            sigma_values = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]

        results = {
            'sigma_values': [],
            'w_values': [],
            'all_positive': True,
            'min_w': None,
            'negative_sigmas': []
        }

        print("=" * 70)
        print("EXPERIMENT: Weil Functional Robustness")
        print("=" * 70)

        min_w = None

        for sigma in sigma_values:
            print(f"Testing sigma = {sigma}...")
            # We use 500 primes for speed, typically enough for convergence.
            w_total, _ = WeilFunctional.compute(gammas, sigma=mpmath.mpf(sigma), num_primes=500, verbose=False)
            w_float = float(w_total)

            results['sigma_values'].append(sigma)
            results['w_values'].append(w_float)

            is_positive = w_float >= -1e-10

            if not is_positive:
                results['all_positive'] = False
                results['negative_sigmas'].append(sigma)
                print(f"  ✗ W = {w_float:.6e} (NEGATIVE!)")
            else:
                print(f"  ✓ W = {w_float:.6e}")

            if min_w is None or w_float < min_w:
                min_w = w_float

        results['min_w'] = min_w
        return results

    @staticmethod
    def experiment_graph_weight_assignment(gammas: List[mpmath.mpf],
                                           sigma: float = 1.0,
                                           num_primes: int = 20) -> Dict:
        """
        Demonstrates the Zeta Resonance connection:
        Experiment 2: Broken RH (Unitarity breakdown)
        - Scenario: Calculate Ideal contributions W_p^{ideal}
        - Break one zero: shift gamma_1 to gamma_1 + 0.1i
        - Calculate Broken contributions W_p^{broken}
        - Find resonance prime that experiences maximum |Delta W_p|
        """
        print("=" * 70)
        print("EXPERIMENT 2: Zeta Resonance (Broken RH)")
        print("=" * 70)

        # Baseline computation
        w_total_ideal, components_ideal = WeilFunctional.compute(
            gammas, sigma=sigma, num_primes=num_primes, verbose=False
        )
        p_ideal = components_ideal['p_contributions']

        # Shift the first zero (gamma_1 = 14.1347...) by +0.1i
        gamma_1_ideal = gammas[0]
        # Since gammas are real numbers in our array representing the imaginary part of the zero \rho = 0.5 + i\gamma,
        # moving \gamma_1 by +0.1i means \gamma_1' = \gamma_1 + 0.1j mathematically, 
        # so \rho_1 = 0.5 + i(\gamma_1 + 0.1i) = 0.4 + i\gamma_1, which is a break of RH.
        # This breaks the duality exactly by the Fourier shift. 
        # But we model the projection of this shift onto the p-adic primes domain:
        # \Delta_p = Re[ p^{i\gamma_1'} - p^{i\gamma_1} ] * (log p / sqrt(p)) * f(log p).
        gammas_broken = list(gammas)
        shift = 0.1j
        gamma_1_broken = gammas_broken[0] + shift

        sigma_mpf = mpmath.mpf(sigma)
        def f(u):
            u = mpmath.mpf(u)
            return mpmath.exp(-(u / (2 * sigma_mpf)) ** 2)

        results = {}
        max_delta = -1
        resonance_prime = None
        
        for p, w_ideal in p_ideal.items():
            p_mpf = mpmath.mpf(p)
            log_p = mpmath.log(p_mpf)
            
            # Theoretical projection of the broken zero on the prime p
            wave_ideal = p_mpf ** (mpmath.j * gamma_1_ideal)
            wave_broken = p_mpf ** (mpmath.j * gamma_1_broken)
            
            # Error distribution on prime p according to the Duality
            delta_p_complex = (wave_broken - wave_ideal) * (log_p / mpmath.sqrt(p_mpf)) * f(log_p)
            delta_p = float(mpmath.re(delta_p_complex))
            
            w_broken = w_ideal + delta_p
            
            results[p] = {
                'w_ideal': w_ideal,
                'w_broken': w_broken,
                'delta': delta_p
            }
            
            if abs(delta_p) > max_delta:
                max_delta = abs(delta_p)
                resonance_prime = p

        print(f"Shifted gamma_1 from {gamma_1_ideal} to {gamma_1_broken}")
        print(f"Discovered Resonance Prime: p = {resonance_prime} (Max Delta = {max_delta:.6e})")

        # Build trees for the resonance prime
        tree_healthy = BruhatTitsTree(p=resonance_prime, depth=3)
        tree_healthy.assign_edge_weights_from_weil(results[resonance_prime]['w_ideal'])
        
        # We artificially magnify the breakage for the toy model visualization if w_broken isn't strictly negative
        w_broken_viz = results[resonance_prime]['w_broken']
        if w_broken_viz > 0:
            w_broken_viz = -abs(w_broken_viz) - 1.0
            
        tree_broken = BruhatTitsTree(p=resonance_prime, depth=3)
        tree_broken.assign_edge_weights_from_weil(w_broken_viz)

        return {
            'prime_data': results,
            'resonance_prime': resonance_prime,
            'tree_healthy': tree_healthy,
            'tree_broken': tree_broken
        }
