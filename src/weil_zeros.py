import mpmath
from typing import List


class ZerosContribution:
    """
    Computes the contribution from non-trivial Riemann zeros to the Weil functional.
    W_zeros(f) = sum_n f_hat(gamma_n)
    where f_hat is the Mellin transform of f.
    """

    @staticmethod
    def gaussian_fourier_transform(t: mpmath.mpf, sigma: mpmath.mpf = 1) -> mpmath.mpf:
        """
        Analytic Mellin-Fourier transform of the test Gaussian:
        f(u) = exp(-(u / 2*sigma)^2)
        f_hat(t) = 2 * sigma * sqrt(pi) * exp(-(sigma * t)^2)
        """
        # Convert inputs to precise mpf values
        t = mpmath.mpf(t)
        sigma = mpmath.mpf(sigma)
        
        return 2 * sigma * mpmath.sqrt(mpmath.pi) * mpmath.exp(-(sigma * t) ** 2)

    @staticmethod
    def compute(gammas: List[mpmath.mpf], sigma: mpmath.mpf = 1) -> mpmath.mpf:
        """
        Sums the contribution of all loaded gamma_n.
        
        Args:
            gammas: List of imaginary parts of Riemann zeros (gamma_n)
            sigma: Width parameter for the Gaussian test function
            
        Returns:
            The sum of f_hat(gamma_n) over all n mapped.
        """
        total = mpmath.mpf(0)
        sigma = mpmath.mpf(sigma)
        
        for i, gamma_n in enumerate(gammas):
            f_hat_val = ZerosContribution.gaussian_fourier_transform(gamma_n, sigma)
            total += f_hat_val
            
            # The test Gaussian decays EXTREMELY fast (exp(-sigma^2 t^2)).
            # Thus, we can safely break early if f_hat_val underflows.
            if f_hat_val < mpmath.mpf('1e-50'):
                break

        # Assuming symmetry of Riemann zeros: gamma_{-n} = -gamma_n.
        # For our test function f_hat(t) = f_hat(-t), so the total sum includes both positive and negative zeros.
        # The gammas list only has positive zeros.
        return 2 * total
