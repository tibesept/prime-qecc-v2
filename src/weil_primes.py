import mpmath
from sympy import prime, primerange
from typing import Callable, Tuple, Dict


class PrimesContribution:
    @staticmethod
    def first_n_primes(n: int):
        if n <= 0:
            return []
        upper = int(prime(n))
        return list(primerange(2, upper + 1))

    @staticmethod
    def compute(f: Callable[[mpmath.mpf], mpmath.mpf],
                num_primes: int = 500) -> Tuple[mpmath.mpf, Dict[int, float]]:
        """
        Computes the primes contribution using the first num_primes primes.
        """
        primes = PrimesContribution.first_n_primes(int(num_primes))
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

            # Divide by 2 to compensate for symmetric evaluation f(u) + f(-u)
            prime_term = log_p * (inner_sum / 2)
            w_primes += prime_term
            p_contributions[p] = float(prime_term)

        return w_primes, p_contributions
