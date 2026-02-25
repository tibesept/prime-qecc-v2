import mpmath
from typing import Callable


class ArchimideanTerm:
    """
    Computes the Archimedean contribution to the Weil explicit functional W_R(f).
    W_R(f) = (ln(4pi) + gamma_E) f(1) + PV int_0^infty [f(u) + f(1/u) - 2f(1)] / (u - 1) * du/u
    Where gamma_E is the Euler-Mascheroni constant and PV is the Cauchy principal value.
    """

    @staticmethod
    def compute(f: Callable[[mpmath.mpf], mpmath.mpf],
                f_of_0: mpmath.mpf = None) -> mpmath.mpf:
        """
        Args:
            f: Test function, assumed to be smooth and rapidly decaying
            f_of_0: Precomputed value of f(0) if available
            
        Returns:
            The Archimedean term W_R(f) as mpmath.mpf
        """
        if f_of_0 is None:
            f_of_0 = f(mpmath.mpf(0))

        # Part 1: (ln(4pi) + gamma_E) * f(0)
        coeff = mpmath.log(4 * mpmath.pi) + mpmath.euler
        part1 = coeff * f_of_0

        # Part 2: Principal value integral
        def integrand(x):
            x = mpmath.mpf(x)
            if abs(x) < mpmath.mpf('1e-15'):
                return mpmath.mpf(0)
            
            # Correct additive formulation for Weil's Archimedean integral:
            # f(x) is evaluated additively, and the denominator (exp(x) - 1) 
            # is compensated by exp(x/2) factors to handle singularities.
            num = f(x) * mpmath.exp(x/2) + f(-x) * mpmath.exp(-x/2) - 2 * f_of_0
            den = abs(mpmath.exp(x) - 1)
            return num / den

        part2 = mpmath.quad(integrand, [0, mpmath.mpf('1e-5'), 1, 10, mpmath.inf])

        return part1 + part2
