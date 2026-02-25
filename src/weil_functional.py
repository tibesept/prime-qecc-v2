import mpmath
from weil_archimedean import ArchimideanTerm
from weil_zeros import ZerosContribution
from weil_primes import PrimesContribution


class WeilFunctional:
    """
    Computes the full Weil explicit formula functional:
    W(f) = W_R(f) + W_zeros(f) + W_primes(f)
    """

    @staticmethod
    def gaussian_testfunc(sigma: mpmath.mpf = 1):
        """
        Returns a tuple (f, f_hat) for a test Gaussian.
        """
        sigma = mpmath.mpf(sigma)

        def f(u):
            u = mpmath.mpf(u)
            return mpmath.exp(-(u / (2 * sigma)) ** 2)

        def f_hat(t):
            return ZerosContribution.gaussian_fourier_transform(t, sigma)

        return f, f_hat

    @staticmethod
    def compute(gammas, sigma=1, num_primes=500, verbose=True):
        """
        Computes the full Weil functional value.
        """
        sigma = mpmath.mpf(sigma)
        f, f_hat = WeilFunctional.gaussian_testfunc(sigma)

        if verbose:
            print("=" * 60)
            print("Computing Weil functional...")
            print(f"Sigma: {sigma}, Precision (dps): {mpmath.mp.dps}")
            print("=" * 60)

        # 1. Archimedean term W_R(f)
        w_arch = ArchimideanTerm.compute(f)
        if verbose:
            print(f"[1/3] Archimedean term: W_R = {float(w_arch):.6e}")

        # 2. Zeros term W_zeros(f)
        w_zeros = ZerosContribution.compute(gammas, sigma)
        if verbose:
            print(f"[2/3] Riemann zeros term: W_zeros = {float(w_zeros):.6e}")

        # 3. Primes term W_primes(f)
        w_primes, p_contributions = PrimesContribution.compute(f, num_primes)
        if verbose:
            print(f"[3/3] Primes term: W_primes = {float(w_primes):.6e}")

        # Poles of the Riemann zeta function at s=0 and s=1 correspond to gamma = i/2 and -i/2.
        # The Fourier transform of our Gaussian test function: 
        # f_hat(t) = 2 * sigma * sqrt(pi) * exp(-(sigma * t)^2)
        # Evaluated at t = ±i/2 (so t^2 = -1/4).
        pole_term_value = 2 * sigma * mpmath.sqrt(mpmath.pi) * mpmath.exp((sigma**2) / 4)
        w_poles = 2 * pole_term_value  # Sum of contributions from i/2 and -i/2

        W_f_geom = w_poles - w_arch - w_primes
        
        W_f_spec = w_zeros

        identity_error = abs(W_f_geom - W_f_spec)

        if verbose:
            print("-" * 60)
            print(f"W(f) via Geometry = {float(W_f_geom):.6e}")
            print(f"W(f) via Spectrum = {float(W_f_spec):.6e}")
            print(f"Identity Error    = {float(identity_error):.6e}")
            print(f"Weil Positivity   : {'✓ POSITIVE (RH likely)' if W_f_geom >= -1e-10 else '✗ NEGATIVE (RH violated)'}")
            print("=" * 60)

        return W_f_geom, {
            'W_archimedean': w_arch,
            'W_zeros': w_zeros,
            'W_primes': w_primes,
            'W_poles': w_poles,
            'W_total': W_f_geom,
            'identity_error': identity_error,
            'sigma': sigma,
            'num_gammas': len(gammas),
            'num_primes': num_primes
        }

