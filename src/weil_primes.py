import mpmath
from sympy import primerange
from typing import Callable, Tuple, Dict

class PrimesContribution:
    @staticmethod
    def compute(f: Callable[[mpmath.mpf], mpmath.mpf], 
                num_primes: int = 500) -> Tuple[mpmath.mpf, Dict[int, float]]:
        
        primes = list(primerange(2, num_primes))
        w_primes = mpmath.mpf(0)
        p_contributions = {}

        for p in primes:
            p_mp = mpmath.mpf(p)
            log_p = mpmath.log(p_mp)
            inner_sum = mpmath.mpf(0)

            for m in range(1, 100):
                p_power = p_mp ** (-mpmath.mpf(m) / 2)
                u = mpmath.mpf(m) * log_p
                term = p_power * (f(u) + f(-u))
                inner_sum += term
                
                if abs(term) < mpmath.mpf(10) ** (-mpmath.mp.dps):
                    break

            # Делим на 2, чтобы убрать двойной учет симметрии f(u)+f(-u)
            prime_term = log_p * (inner_sum / 2)
            w_primes += prime_term
            p_contributions[p] = float(prime_term)

        return w_primes, p_contributions
