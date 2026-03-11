import mpmath
import numpy as np
from typing import Dict, List, Sequence


class TensorDeformationAnsatz:
    """
    Finite-dimensional operator ansatz used to probe how an externally supplied
    anomaly parameter delta deforms an isometric tensor.
    """

    def __init__(self, dim_in: int = 2, dim_out: int = 4, alpha: float = 1.0, seed: int = 7):
        if dim_out < dim_in:
            raise ValueError("dim_out must be >= dim_in so the baseline tensor can be an isometry.")

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.alpha = float(alpha)
        self.seed = int(seed)

        self.t0 = self._build_seeded_isometry(dim_out, dim_in, seed)
        self.k = self._build_positive_operator(dim_out, seed + 1)

    @staticmethod
    def _build_seeded_isometry(dim_out: int, dim_in: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        real_part = rng.normal(size=(dim_out, dim_in))
        imag_part = rng.normal(size=(dim_out, dim_in))
        matrix = real_part + 1j * imag_part
        q, _ = np.linalg.qr(matrix)
        return q[:, :dim_in]

    @staticmethod
    def _build_positive_operator(dim: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        real_part = rng.normal(size=(dim, dim))
        imag_part = rng.normal(size=(dim, dim))
        matrix = real_part + 1j * imag_part
        hermitian_psd = matrix.conj().T @ matrix

        eigvals = np.linalg.eigvalsh(hermitian_psd)
        max_eval = float(np.max(eigvals))
        if max_eval <= 0:
            return np.eye(dim, dtype=np.complex128)
        return hermitian_psd / max_eval

    @staticmethod
    def _choi_matrix_from_kraus(kraus_ops: List[np.ndarray]) -> np.ndarray:
        if not kraus_ops:
            raise ValueError("At least one Kraus operator is required.")

        dim_out, dim_in = kraus_ops[0].shape
        choi = np.zeros((dim_out * dim_in, dim_out * dim_in), dtype=np.complex128)

        for i in range(dim_in):
            for j in range(dim_in):
                basis = np.zeros((dim_in, dim_in), dtype=np.complex128)
                basis[i, j] = 1.0
                mapped = np.zeros((dim_out, dim_out), dtype=np.complex128)
                for k_op in kraus_ops:
                    mapped += k_op @ basis @ k_op.conj().T

                row_slice = slice(i * dim_out, (i + 1) * dim_out)
                col_slice = slice(j * dim_out, (j + 1) * dim_out)
                choi[row_slice, col_slice] = mapped

        return choi

    def deformation_operator(self, delta: float) -> np.ndarray:
        if delta < 0:
            raise ValueError("delta must be non-negative for this attenuation ansatz.")

        eigvals, eigvecs = np.linalg.eigh(self.k)
        attenuated = np.exp(-self.alpha * float(delta) * eigvals)
        return eigvecs @ np.diag(attenuated) @ eigvecs.conj().T

    def deformed_tensor(self, delta: float) -> np.ndarray:
        return self.deformation_operator(delta) @ self.t0

    def diagnostics(self, delta: float, tol: float = 1e-10) -> Dict[str, float]:
        tensor = self.deformed_tensor(delta)
        gram = tensor.conj().T @ tensor
        identity_in = np.eye(self.dim_in, dtype=np.complex128)

        isometry_defect = float(np.linalg.norm(gram - identity_in, ord="fro"))
        singular_values = np.linalg.svd(tensor, compute_uv=False)
        contraction_norm = float(np.max(singular_values))

        # Single-Kraus map E(rho) = T rho T^\dagger
        kraus_ops = [tensor]
        choi = self._choi_matrix_from_kraus(kraus_ops)
        choi_eigs = np.linalg.eigvalsh((choi + choi.conj().T) / 2)
        choi_min_eig = float(np.min(choi_eigs))
        cp_ok = bool(choi_min_eig >= -tol)

        trace_op = tensor.conj().T @ tensor
        tp_defect = float(np.linalg.norm(trace_op - identity_in, ord="fro"))

        return {
            "delta": float(delta),
            "isometry_defect": isometry_defect,
            "trace_preservation_defect": tp_defect,
            "contraction_norm": contraction_norm,
            "choi_min_eigenvalue": choi_min_eig,
            "cp_ok": cp_ok,
            "tp_ok": bool(tp_defect <= tol),
        }


class PrimeFunctionalTransfer:
    """
    Builds a prime-space kernel directly from prime modes and the functional-equation
    pairing rho <-> 1-rho.

    On the critical line, 1-rho = conjugate(rho), so the kernel is of Gram type.
    After unit-trace normalization, that gives a valid state / 1->N channel candidate.

    Off the line, the same construction can stop being positive semidefinite even if
    Hermiticity is preserved. The kernel is accumulated with mpmath precision so the
    input zero precision is not immediately collapsed to float64.
    """

    def __init__(self, primes: Sequence[int], bandwidth: float = 0.25):
        if not primes:
            raise ValueError("At least one prime is required.")

        self.primes = [int(p) for p in primes]
        self.bandwidth = mpmath.mpf(bandwidth)
        self.logs = [mpmath.log(p) for p in self.primes]
        self.weights = [
            (log_p / mpmath.sqrt(p)) * mpmath.exp(-(self.bandwidth * log_p) ** 2)
            for p, log_p in zip(self.primes, self.logs)
        ]

    @staticmethod
    def spectral_parameter(rho: complex) -> mpmath.mpc:
        return -mpmath.j * (mpmath.mpc(rho) - mpmath.mpf("0.5"))

    def prime_mode_vector(self, rho: complex) -> List[mpmath.mpc]:
        tau = self.spectral_parameter(rho)
        return [
            weight * mpmath.exp(-mpmath.j * tau * log_p)
            for weight, log_p in zip(self.weights, self.logs)
        ]

    @staticmethod
    def _trace(kernel: mpmath.matrix) -> mpmath.mpc:
        return sum(kernel[i, i] for i in range(kernel.rows))

    @staticmethod
    def _adjoint(kernel: mpmath.matrix) -> mpmath.matrix:
        adjoint = mpmath.matrix(kernel.cols, kernel.rows)
        for i in range(kernel.rows):
            for j in range(kernel.cols):
                adjoint[j, i] = mpmath.conj(kernel[i, j])
        return adjoint

    @staticmethod
    def _frobenius_norm(kernel: mpmath.matrix) -> mpmath.mpf:
        total = mpmath.mpf("0")
        for i in range(kernel.rows):
            for j in range(kernel.cols):
                total += abs(kernel[i, j]) ** 2
        return mpmath.sqrt(total)

    @staticmethod
    def _scale(kernel: mpmath.matrix, scalar: mpmath.mpf) -> mpmath.matrix:
        scaled = mpmath.matrix(kernel.rows, kernel.cols)
        for i in range(kernel.rows):
            for j in range(kernel.cols):
                scaled[i, j] = kernel[i, j] / scalar
        return scaled

    def transfer_kernel(self, rhos: Sequence[complex]) -> mpmath.matrix:
        n_primes = len(self.primes)
        kernel = mpmath.matrix(n_primes, n_primes)
        for rho in rhos:
            v_rho = self.prime_mode_vector(rho)
            v_pair = self.prime_mode_vector(1 - mpmath.mpc(rho))
            for i in range(n_primes):
                for j in range(n_primes):
                    kernel[i, j] += v_rho[i] * v_pair[j]
        return kernel

    @staticmethod
    def normalize_by_trace(kernel: mpmath.matrix) -> mpmath.matrix:
        trace_val = PrimeFunctionalTransfer._trace(kernel)
        scale = mpmath.re(trace_val)
        if abs(scale) < mpmath.mpf("1e-30"):
            return kernel
        return PrimeFunctionalTransfer._scale(kernel, scale)

    def diagnostics_from_rhos(self, rhos: Sequence[complex], tol: float = 1e-10) -> Dict[str, float]:
        tol_mp = mpmath.mpf(tol)
        kernel = self.transfer_kernel(rhos)
        kernel_norm = self.normalize_by_trace(kernel)
        adjoint = self._adjoint(kernel_norm)
        hermiticity_defect = self._frobenius_norm(kernel_norm - adjoint)

        hermitian_part = (kernel_norm + adjoint) / 2
        hermitian_eigs = mpmath.eig(hermitian_part, left=False, right=False)
        min_hermitian_eig = min(mpmath.re(val) for val in hermitian_eigs)
        max_hermitian_eig = max(mpmath.re(val) for val in hermitian_eigs)

        trace_val = self._trace(kernel)
        normalized_trace = self._trace(kernel_norm)
        trace_real = mpmath.re(trace_val)
        trace_imag = mpmath.im(trace_val)
        unit_trace_error = abs(normalized_trace - 1)

        trace_normalization_ok = (
            trace_real > tol_mp
            and abs(trace_imag) <= tol_mp * max(1, abs(trace_real))
            and unit_trace_error <= tol_mp
        )
        cp_candidate_ok = hermiticity_defect <= tol_mp and min_hermitian_eig >= -tol_mp
        state_channel_candidate_ok = bool(trace_normalization_ok and cp_candidate_ok)

        failure_mode = "ok"
        if not trace_normalization_ok:
            failure_mode = "trace_normalization"
        elif hermiticity_defect > tol_mp:
            failure_mode = "non_hermitian"
        elif min_hermitian_eig < -tol_mp:
            failure_mode = "negative_eigenvalue"

        return {
            "num_primes": len(self.primes),
            "num_rhos": len(rhos),
            "trace_real": float(trace_real),
            "trace_imag": float(trace_imag),
            "unit_trace_error": float(unit_trace_error),
            "trace_normalization_ok": bool(trace_normalization_ok),
            "hermiticity_defect": float(hermiticity_defect),
            "min_hermitian_eigenvalue": float(min_hermitian_eig),
            "max_hermitian_eigenvalue": float(max_hermitian_eig),
            "cp_candidate_ok": bool(cp_candidate_ok),
            "state_channel_candidate_ok": state_channel_candidate_ok,
            "failure_mode": failure_mode,
            "kernel": kernel_norm,
        }
