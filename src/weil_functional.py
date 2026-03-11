import mpmath
from weil_archimedean import ArchimideanTerm
from weil_zeros import ZerosContribution
from weil_primes import PrimesContribution

class WeilFunctional:
    """
    Computes the full Weil explicit formula functional:
    W_zeros(f) = W_poles(f) - W_arch(f) - W_primes(f)

    Also provides tools for the Weil Positivity Criterion:
    RH ⟺ W(F) ≥ 0 for all F = f * f̃  (self-convolution test functions)
    """

    @staticmethod
    def gaussian_testfunc(sigma: mpmath.mpf = 1):
        """
        Standard Gaussian test function.
        f(u) = exp(-(u / 2σ)²)
        f̂(t) = 2σ√π · exp(-(σt)²)

        This automatically satisfies the positivity condition because
        f̂(t) > 0 ∀t, which is equivalent to F = f * f̃.
        """
        sigma = mpmath.mpf(sigma)

        def f(u):
            u = mpmath.mpf(u)
            return mpmath.exp(-(u / (2 * sigma)) ** 2)

        def f_hat(t):
            t = mpmath.mpf(t)
            return 2 * sigma * mpmath.sqrt(mpmath.pi) * mpmath.exp(-(sigma * t) ** 2)

        return f, f_hat

    @staticmethod
    def resonant_testfunc(gamma_target, bandwidth=0.5):
        """
        Resonant test function whose Fourier transform peaks at t = ±γ_target.

        We construct g(u) = exp(-(u/2σ)²) · cos(γ₁·u)
        so that ĝ(t) ∝ exp(-σ²(t-γ₁)²) + exp(-σ²(t+γ₁)²)

        Then F = g * g̃ has Fourier transform |ĝ(t)|² ≥ 0,
        satisfying the admissibility requirement for Weil's criterion.

        The bandwidth σ controls how sharply the peak is localized at γ_target.
        Smaller σ → sharper peak → better detection of anomalies near γ_target.

        Args:
            gamma_target: the imaginary part of the zero to resonate with (e.g. 14.1347...)
            bandwidth: σ parameter controlling peak width (smaller = sharper)

        Returns:
            (F, F_hat) where F_hat(t) = |ĝ(t)|² is manifestly non-negative on the real line.
        """
        gamma_target = mpmath.mpf(gamma_target)
        sigma = mpmath.mpf(bandwidth)

        # ĝ(t) — Fourier transform of g(u) = Gaussian × cos
        # ĝ(t) = σ√π [ exp(-σ²(t-γ₁)²) + exp(-σ²(t+γ₁)²) ]
        def g_hat(t):
            t_val = mpmath.mpf(t) if isinstance(t, (int, float, str)) else t
            peak_plus = mpmath.exp(-(sigma * (t_val - gamma_target)) ** 2)
            peak_minus = mpmath.exp(-(sigma * (t_val + gamma_target)) ** 2)
            return sigma * mpmath.sqrt(mpmath.pi) * (peak_plus + peak_minus)

        # F̂(t) = [ĝ(t)]²  — manifestly non-negative on the real line
        # since ĝ(t) is real-valued for real t.
        def F_hat(t):
            val = g_hat(t)
            return val ** 2

        # F̂(z) for complex argument z (needed for off-critical-line zeros)
        def F_hat_complex(z):
            """
            Analytic continuation of F̂(t) = [ĝ(t)]² to complex z.

            Since g(u) is real-valued and even, ĝ has real Taylor coefficients.
            The analytic continuation of |ĝ|² (which equals ĝ² on the reals)
            is simply [ĝ(z)]².

            CRITICAL: this is NOT ĝ(z)·ĝ(z̄). That would give |ĝ(z)|² ≥ 0 always,
            making the positivity criterion vacuous. The correct analytic continuation
            [ĝ(z)]² CAN be negative for complex z — this is what detects RH violations.
            """
            z_val = mpmath.mpc(z)
            peak_plus = mpmath.exp(-(sigma * (z_val - gamma_target)) ** 2)
            peak_minus = mpmath.exp(-(sigma * (z_val + gamma_target)) ** 2)
            g_val = sigma * mpmath.sqrt(mpmath.pi) * (peak_plus + peak_minus)
            return g_val ** 2

        # F(u) in position space — inverse Fourier of |ĝ|²
        # For the primes-side computation we need f(u).
        # F = g * g̃ in position space. Since g is real and even, g̃ = g,
        # so F(u) = ∫ g(v) g(u-v) dv = (autocorrelation of g).
        # Compute numerically via quadrature.
        def F(u):
            u = mpmath.mpf(u)
            def integrand(v):
                v = mpmath.mpf(v)
                gauss_v = mpmath.exp(-(v / (2 * sigma)) ** 2) * mpmath.cos(gamma_target * v)
                gauss_uv = mpmath.exp(-((u - v) / (2 * sigma)) ** 2) * mpmath.cos(gamma_target * (u - v))
                return gauss_v * gauss_uv
            # Integration range: g decays as Gaussian with width ~2σ
            bound = 8 * float(sigma)
            result = mpmath.quad(integrand, [-bound, 0, bound])
            return result

        return F, F_hat, F_hat_complex

    @staticmethod
    def compute(gammas, sigma=1, num_primes=500, verbose=True):
        """
        Standard Weil explicit formula computation (identity check).
        Uses the basic Gaussian test function.
        """
        sigma = mpmath.mpf(sigma)

        if verbose:
            print("=" * 60)
            print("Computing Weil functional...")
            print(f"Sigma: {sigma}, Precision (dps): {mpmath.mp.dps}")
            print("=" * 60)

        f, f_hat = WeilFunctional.gaussian_testfunc(sigma)

        # 1. Zeros (Спектр) — sum f̂ over all zeros
        w_zeros = mpmath.mpf(0)
        for g in gammas:
            gamma = mpmath.mpf(g)
            w_zeros += 2 * f_hat(gamma)

        # 2. Archimedean
        w_arch = ArchimideanTerm.compute(f_hat)

        # 3. Primes
        w_primes, p_contributions = PrimesContribution.compute(f, num_primes)

        # 4. Poles: f̂(i/2) + f̂(-i/2)
        w_poles = 4 * sigma * mpmath.sqrt(mpmath.pi) * mpmath.exp((sigma**2) / 4)

        # 5. Balance (Geometry)
        w_geom = w_poles - w_arch - w_primes

        # 6. Identity error
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

    @staticmethod
    def compute_spectral_sum(gammas, F_hat_func, broken_zeros=None):
        """
        Compute the spectral side of the Weil functional:
        W_spec(F) = Σ_ρ Re[F̂( -i (ρ - 1/2) )]

        For broken zeros (off the critical line), we use:
        W_spec(F) with a symmetry-complete replacement orbit
        {ρ, 1-ρ, ρ̄, 1-ρ̄} for each modified zero.

        Args:
            gammas: list of γ_n (imaginary parts of zeros on the critical line)
            F_hat_func: callable F̂(t), valid for real or complex t
            broken_zeros: dict {index: complex_rho} for zeros moved off the line.
                          e.g. {0: mpc(0.4, 14.1347)} means ρ₁ = 0.4 + 14.1347i

        Returns:
            W_spec: the spectral sum (real-valued)
        """
        if broken_zeros is None:
            broken_zeros = {}

        def spectral_arg(rho):
            # This map agrees with the healthy case:
            # rho = 1/2 + i*gamma  ->  -i*(rho-1/2) = gamma (real).
            return -mpmath.j * (rho - mpmath.mpf("0.5"))

        def canonical_key(z, ndigits=30):
            return (
                mpmath.nstr(mpmath.re(z), ndigits),
                mpmath.nstr(mpmath.im(z), ndigits),
            )

        w_spec = mpmath.mpf(0)

        for i, gamma in enumerate(gammas):
            gamma = mpmath.mpf(gamma)

            if i in broken_zeros:
                rho = mpmath.mpc(broken_zeros[i])

                orbit = [
                    rho,
                    1 - rho,
                    mpmath.conj(rho),
                    1 - mpmath.conj(rho),
                ]

                seen = set()
                for orbit_rho in orbit:
                    arg = spectral_arg(orbit_rho)
                    key = canonical_key(arg)
                    if key in seen:
                        continue
                    seen.add(key)
                    w_spec += mpmath.re(F_hat_func(arg))
            else:
                val = F_hat_func(gamma)
                w_spec += 2 * mpmath.re(val)

            # Early termination for fast-decaying F̂
            if i not in broken_zeros and abs(F_hat_func(gamma)) < mpmath.mpf('1e-50'):
                break

        return w_spec
