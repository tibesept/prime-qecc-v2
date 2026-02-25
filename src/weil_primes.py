import mpmath
from typing import Callable, List


class PrimesContribution:
    """
    Computes the contribution from prime numbers to the Weil functional.
    W_primes(f) = sum_p log(p) * sum_{m=1}^infty p^{-m/2} * [f(m*log_p) + f(-m*log_p)]
    CRITICAL: The factor p^{-m/2} is mandatory for convergence.
    """

    @staticmethod
    def first_primes(n: int = 1000) -> List[int]:
        """Returns the first n prime numbers using Sieve of Eratosthenes."""
        limit = max(15000, n * 20)  # Safe heuristic limit
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
                    
        primes = [i for i in range(2, limit + 1) if is_prime[i]]
        return primes[:n]

    @staticmethod
    def compute(f: Callable[[mpmath.mpf], mpmath.mpf], num_primes: int = 1000) -> tuple[mpmath.mpf, dict[int, float]]:
        """
        Computes W_primes(f).
        
        Args:
            f: The test function (evaluated on the multiplicative group u = exp(m*log_p))
            num_primes: Count of primes to sum over.
            
        Returns:
            Tuple of (total_sum as mpmath.mpf, dictionary of {prime: float_contribution})
        """
        primes = PrimesContribution.first_primes(num_primes)
        total = mpmath.mpf(0)
        p_contributions = {}
        
        for p in primes:
            p_mp = mpmath.mpf(p)
            log_p = mpmath.log(p_mp)
            
            inner_sum = mpmath.mpf(0)
            
            # The sum usually converges quickly because of p^{-m/2} and the Gaussian decay of f
            for m in range(1, 100):
                p_power = p_mp ** (-mpmath.mpf(m) / mpmath.mpf(2))
                
                if p_power < mpmath.mpf(10) ** (-mpmath.mp.dps):
                    break
                    
                u_pos = mpmath.mpf(m) * log_p
                u_neg = -mpmath.mpf(m) * log_p
                
                f_pos = f(u_pos)
                f_neg = f(u_neg)
                
                term = p_power * (f_pos + f_neg)
                inner_sum += term
                
                if abs(term) < mpmath.mpf(10) ** (-mpmath.mp.dps):
                    break
                    
            prime_term = log_p * inner_sum
            total += prime_term
            p_contributions[int(p)] = float(prime_term)
            
            # Since f has exponential decay, outer sum also converges. Break early if contribution vanishes.
            if abs(prime_term) < mpmath.mpf(10) ** (-mpmath.mp.dps):
                break
                
        return total, p_contributions
