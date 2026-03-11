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
    Builds a prime-space transfer/Choi candidate directly from prime modes and the
    functional-equation pairing rho <-> 1-rho.

    On the critical line, 1-rho = conjugate(rho), so the kernel is of Gram type.
    Off the line, that identity fails and the same construction can stop being a
    valid Hermitian PSD Choi candidate.
    """

    def __init__(self, primes: Sequence[int], bandwidth: float = 0.25):
        if not primes:
            raise ValueError("At least one prime is required.")

        self.primes = [int(p) for p in primes]
        self.bandwidth = float(bandwidth)
        self.logs = np.array([float(mpmath.log(p)) for p in self.primes], dtype=np.float64)
        self.weights = np.array(
            [
                float(
                    (mpmath.log(p) / mpmath.sqrt(p))
                    * mpmath.exp(-(self.bandwidth * mpmath.log(p)) ** 2)
                )
                for p in self.primes
            ],
            dtype=np.complex128,
        )

    @staticmethod
    def spectral_parameter(rho: complex) -> complex:
        return complex(-1j * (complex(rho) - 0.5))

    def prime_mode_vector(self, rho: complex) -> np.ndarray:
        tau = self.spectral_parameter(rho)
        return self.weights * np.exp(-1j * tau * self.logs)

    def transfer_kernel(self, rhos: Sequence[complex]) -> np.ndarray:
        kernel = np.zeros((len(self.primes), len(self.primes)), dtype=np.complex128)
        for rho in rhos:
            v_rho = self.prime_mode_vector(rho)
            v_pair = self.prime_mode_vector(1 - complex(rho))
            kernel += np.outer(v_rho, v_pair)
        return kernel

    @staticmethod
    def normalize_by_trace(kernel: np.ndarray) -> np.ndarray:
        trace_val = np.trace(kernel)
        scale = float(np.real(trace_val))
        if abs(scale) < 1e-14:
            return kernel
        return kernel / scale

    def diagnostics_from_rhos(self, rhos: Sequence[complex], tol: float = 1e-10) -> Dict[str, float]:
        kernel = self.transfer_kernel(rhos)
        kernel_norm = self.normalize_by_trace(kernel)
        hermiticity_defect = float(np.linalg.norm(kernel_norm - kernel_norm.conj().T, ord="fro"))
        hermitian_part = (kernel_norm + kernel_norm.conj().T) / 2
        hermitian_eigs = np.linalg.eigvalsh(hermitian_part)
        min_hermitian_eig = float(np.min(hermitian_eigs))
        max_singular_value = float(np.linalg.norm(kernel_norm, 2))

        return {
            "num_primes": len(self.primes),
            "num_rhos": len(rhos),
            "trace_real": float(np.real(np.trace(kernel))),
            "trace_imag": float(np.imag(np.trace(kernel))),
            "hermiticity_defect": hermiticity_defect,
            "min_hermitian_eigenvalue": min_hermitian_eig,
            "max_singular_value": max_singular_value,
            "cp_candidate_ok": bool(hermiticity_defect <= tol and min_hermitian_eig >= -tol),
            "contractive_candidate_ok": bool(max_singular_value <= 1.0 + tol),
            "kernel": kernel_norm,
        }
