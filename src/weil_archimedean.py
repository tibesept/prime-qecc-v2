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
                f_of_1: mpmath.mpf = None) -> mpmath.mpf:
        """
        Args:
            f: Test function, assumed to be smooth and rapidly decaying
            f_of_1: Precomputed value of f(1) if available
            
        Returns:
            The Archimedean term W_R(f) as mpmath.mpf
        """
        if f_of_1 is None:
            f_of_1 = f(mpmath.mpf(1))

        # Part 1: (ln(4pi) + gamma_E) * f(1)
        coeff = mpmath.log(4 * mpmath.pi) + mpmath.euler
        part1 = coeff * f_of_1

        # Part 2: Principal value integral
        # PV int_0^infty [f(u) + f(1/u) - 2f(1)] / (u - 1) * du/u
        # Changed variable u = exp(x), du/u = dx, integral goes from -inf to inf
        def integrand(x):
            x = mpmath.mpf(x)
            # Limit as x -> 0 is 0 because f(exp(x)) + f(exp(-x)) is even 
            # and its derivative at x=0 is 0, resolving the 0/0 removable singularity cleanly.
            if abs(x) < mpmath.mpf('1e-15'):
                return mpmath.mpf(0)
            
            num = f(mpmath.exp(x)) + f(mpmath.exp(-x)) - 2 * f_of_1
            den = mpmath.exp(x) - mpmath.mpf(1)
            return num / den

        # Integrate over an effective finite symmetric interval to prevent mpmath.quad from hanging. 
        # The integral theoretically diverges, but evaluating to x ~ +/- 120 matches the 
        # effective integration domain of dps=50 (since e^-120 ~ 1e-52).
        part2 = mpmath.quad(integrand, [-120, 120])

        return part1 + part2
