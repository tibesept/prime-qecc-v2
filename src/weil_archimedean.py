import mpmath
from typing import Callable

class ArchimideanTerm:
    @staticmethod
    def compute(f_hat: Callable[[mpmath.mpf], mpmath.mpf],
                integration_bound: float = 30.0) -> mpmath.mpf:
        
        def integrand(t):
            t = mpmath.mpf(t)
            s_half = mpmath.mpf('0.25') + mpmath.mpc(0, t/2)
            psi = mpmath.re(mpmath.digamma(s_half))
            return f_hat(t) * (mpmath.log(mpmath.pi) - psi)

        result = mpmath.quad(integrand, [-integration_bound, 0, integration_bound])
        return result / (2 * mpmath.pi)
