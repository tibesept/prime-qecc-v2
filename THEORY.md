# Toy Deformation Ansatz (Not a Derivation)

This document records the modeling assumptions used in the codebase.
It does not claim a rigorous derivation from an adelic Hamiltonian.

## Established Inputs

1. Weil positivity criterion (number theory):
- for admissible test functions of the form `F = g * g_tilde`, the RH statement can be phrased via positivity of a Weil functional.

2. p-adic geometry motif:
- Bruhat-Tits trees provide a discrete geometry associated to a fixed prime `p`.

3. Tensor-network motif:
- finite tensors can be tested for exact/isometric embedding properties using operator norms and channel diagnostics.

## Intrinsic Transfer-Kernel Construction

To answer the strongest version of the toy-model challenge, the repository now builds a prime-space transfer kernel directly from the zero data rather than feeding a separate deformation scalar into an attenuator.

For a prime mode basis indexed by primes `p`, define

`psi_p(rho) = w_p exp(-i tau(rho) log p)`

with

- `tau(rho) = -i (rho - 1/2)`
- `w_p = (log p / sqrt(p)) exp(-(sigma log p)^2)`

Then form the functional-pairing kernel

`K_{pq} = sum_rho psi_p(rho) psi_q(1-rho)`

where the sum runs over upper-half-plane zeros.

For a synthetic off-line zero `rho = beta + i gamma`, the upper-half spectrum must also contain
`1-conj(rho) = 1-beta + i gamma`.
The code now completes that upper-half orbit explicitly before building the kernel.

Why this matters:

- if `rho = 1/2 + i gamma`, then `1-rho = conjugate(rho)`, so `K` is a Gram-type kernel and is Hermitian positive-type
- if an upper-half pair `{rho, 1-conj(rho)}` is shifted off the critical line, the normalized kernel can stay Hermitian but still cease to be positive semidefinite
- after unit-trace normalization, a Hermitian PSD kernel defines a valid state / `1 -> N` channel candidate

This is the closest thing in the repository to an intrinsic answer to the objection "does the transfer operator itself fail, rather than a hand-imposed dictionary?"

## Ansatz Used in Code

We introduce an external scalar anomaly parameter `delta >= 0` and define

`T_delta = exp(-alpha * delta * K) T0`

with:

- `T0`: a seeded exact isometry (`T0^dagger T0 = I`) in finite dimensions
- `K`: positive semidefinite Hermitian matrix
- `alpha > 0`: coupling constant

From this, one obtains:

- isometry defect: `||T_delta^dagger T_delta - I||`
- trace-preservation defect for map `E(rho)=T_delta rho T_delta^dagger`
- complete positivity check via Choi matrix eigenvalues

This structure is mathematically well-defined, but the link
`(number-theory anomaly) -> delta` is imposed as a dictionary in the coupled experiment.

## Coupling Dictionary in This Repository

For a synthetic per-prime perturbation score, the code normalizes `|delta_p|` to `[0,1]` and feeds it as the ansatz parameter `delta`.
This is a modeling choice, not a theorem.

## Explicit Limitations

- No adelic Hamiltonian is built.
- No full Bost-Connes C*-algebraic/KMS dynamics are simulated.
- No claim is made that `beta = 1` is a QECC threshold in this implementation.
- A single Bruhat-Tits tree at fixed `p` is not an adelic bulk.
- Tree edge signs are visualization only; operator diagnostics are the unitarity-related signal.

## What the Code Can Legitimately Show

- How Weil spectral positivity behaves under synthetic off-line zero perturbations.
- How a directly derived, symmetry-complete functional-pairing kernel can stop being a valid unit-trace Hermitian PSD state / `1 -> N` channel candidate under synthetic off-line zero shifts.
- How a chosen attenuation ansatz affects isometry/trace defects while remaining CP.
- Whether an imposed coupling map produces monotone or non-monotone trends.

## What It Cannot Show

- A proof of RH.
- A derivation that Weil positivity equals physical unitarity of an actual holographic QECC.
- A formal equivalence between Bost-Connes phase structure and error-correction thresholds.
