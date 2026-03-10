import mpmath
from weil_archimedean import ArchimideanTerm
from weil_zeros import ZerosContribution
from weil_primes import PrimesContribution

class WeilFunctional:
    """
    Computes the full Weil explicit formula functional:
    W_zeros(f) = W_poles(f) - W_arch(f) - W_primes(f)
    """

    @staticmethod
    def gaussian_testfunc(sigma: mpmath.mpf = 1):
        sigma = mpmath.mpf(sigma)

        def f(u):
            u = mpmath.mpf(u)
            return mpmath.exp(-(u / (2 * sigma)) ** 2)

        def f_hat(t):
            t = mpmath.mpf(t)
            # ПРАВИЛЬНЫЙ Фурье-образ с множителем 2
            return 2 * sigma * mpmath.sqrt(mpmath.pi) * mpmath.exp(-(sigma * t) ** 2)

        return f, f_hat

    @staticmethod
    def compute(gammas, sigma=1, num_primes=500, verbose=True):
        sigma = mpmath.mpf(sigma)

        if verbose:
            print("=" * 60)
            print("Computing Weil functional...")
            print(f"Sigma: {sigma}, Precision (dps): {mpmath.mp.dps}")
            print("=" * 60)

        f, f_hat = WeilFunctional.gaussian_testfunc(sigma)

        # 1. Zeros (Спектр) - теперь f_hat правильный, поэтому внутри Zeros просто суммируем f_hat
        w_zeros = mpmath.mpf(0)
        for g in gammas:
            gamma = mpmath.mpf(g)
            # Сумма по +gamma и -gamma
            w_zeros += 2 * f_hat(gamma)

        # 2. Archimedean (Передаем f_hat)
        w_arch = ArchimideanTerm.compute(f_hat)

        # 3. Primes (Ожидаем кортеж)
        w_primes, p_contributions = PrimesContribution.compute(f, num_primes)

        # 4. Poles: f_hat(i/2) + f_hat(-i/2)
        # f_hat(i/2) = 2 * sigma * sqrt(pi) * exp(sigma^2 / 4)
        # Сумма для i/2 и -i/2 дает удвоение:
        w_poles = 4 * sigma * mpmath.sqrt(mpmath.pi) * mpmath.exp((sigma**2) / 4)

        # 5. Баланс (Геометрия)
        w_geom = w_poles - w_arch - w_primes

        # 6. Ошибка тождества
        identity_error = abs(w_geom - w_zeros)

        if verbose:
            print("-" * 60)
            print(f"W_poles          = {float(w_poles):.10e}")
            print(f"W_arch           = {float(w_arch):.10e}")
            print(f"W_primes         = {float(w_primes):.10e}")
            print("-" * 60)
            print(f"W(f) Zeros (LHS) = {float(w_zeros):.10e}")
            print(f"W(f) Geom  (RHS) = {float(w_geom):.10e}")
            print(f"Identity Error   = {float(identity_error):.10e}")
            print("=" * 60)

        w_total = w_geom

        return w_total, {
            'W_archimedean': w_arch,
            'W_zeros': w_zeros,
            'W_primes': w_primes,
            'W_poles': w_poles,
            'p_contributions': p_contributions,
            'W_total': w_total,
            'identity_error': identity_error,
            'sigma': float(sigma),
            'num_gammas': len(gammas),
            'num_primes': num_primes
        }
