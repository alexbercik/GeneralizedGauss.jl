
"""
    gauss_legendre(n, ::Type{T}=BigFloat) where T

Compute `n`-point Gauss-Legendre quadrature nodes and weights on `[-1, 1]`
in arithmetic type `T`.  Uses Newton iteration on the three-term Legendre
recurrence, which works natively with `BigFloat` (or any `AbstractFloat`).
"""
function gauss_legendre(n::Int, ::Type{T}=BigFloat) where T
    nodes   = zeros(T, n)
    weights = zeros(T, n)
    m = (n + 1) ÷ 2          # number of non-negative roots (symmetry)

    for i in 1:m
        # Tricomi initial guess
        x = cos(T(π) * (4i - 1) / (4n + 2))

        # Newton iterations to find the root of P_n
        for _ in 1:200
            P_prev, P_curr = one(T), x
            for k in 2:n
                P_next = ((2k - 1) * x * P_curr - (k - 1) * P_prev) / T(k)
                P_prev, P_curr = P_curr, P_next
            end
            # P_curr = P_n(x),  P_prev = P_{n-1}(x)
            dP = T(n) * (x * P_curr - P_prev) / (x^2 - 1)
            δ  = P_curr / dP
            x -= δ
            abs(δ) <= 10 * eps(T) && break
        end

        # Recompute P'_n at the converged root for the weight formula
        P_prev, P_curr = one(T), x
        for k in 2:n
            P_next = ((2k - 1) * x * P_curr - (k - 1) * P_prev) / T(k)
            P_prev, P_curr = P_curr, P_next
        end
        dP = T(n) * (x * P_curr - P_prev) / (x^2 - 1)

        w = T(2) / ((1 - x^2) * dP^2)

        nodes[n + 1 - i] =  x;   weights[n + 1 - i] = w
        nodes[i]          = -x;   weights[i]          = w
    end

    nodes, weights
end


"""
    gram_matrix(basis::Dictionary, quad_order::Int; measure=nothing)

Gram matrix ``G_{ij} = \\int_a^b \\varphi_i(x)\\,\\varphi_j(x)\\,w(x)\\,dx``
evaluated via `quad_order`-point Gauss-Legendre quadrature.  Arithmetic
precision is inherited from the support endpoints of `basis`.

`measure` is an optional callable ``w(x)``; when `nothing` the Lebesgue
measure is used.
"""
function gram_matrix(basis::Dictionary, quad_order::Int; measure=nothing)
    n = length(basis)
    a = leftendpoint(support(basis))
    b = rightendpoint(support(basis))
    T = typeof(a)

    # GL nodes/weights on [-1,1], mapped to [a,b]
    ref_nodes, ref_weights = gauss_legendre(quad_order, T)
    scale = (b - a) / 2
    shift = (a + b) / 2
    qnodes   = scale .* ref_nodes  .+ shift
    qweights = scale .* ref_weights

    # Incorporate the measure weight into the quadrature weights
    if measure !== nothing
        for k in 1:quad_order
            qweights[k] *= measure(qnodes[k])
        end
    end

    # Evaluate all basis functions at quadrature points
    Φ = zeros(T, n, quad_order)
    for i in 1:n, k in 1:quad_order
        Φ[i, k] = funeval(basis, i, qnodes[k])
    end

    # G = Φ diag(qweights) Φᵀ,  symmetrized to eliminate rounding asymmetry
    G = Φ * Diagonal(qweights) * Φ'
    G = (G + G') / 2

    G
end


"""
    _weighted_eval_matrix(basis, quad_order, a, b, ::Type{FT}; measure=nothing)

Build the weighted evaluation matrix ``A_{qi} = \\sqrt{w_q}\\,\\varphi_i(x_q)``
on `quad_order`-point Gauss-Legendre quadrature mapped to `[a, b]`, with
an optional callable `measure` incorporated into the quadrature weights.

The matrix has size `quad_order x n` (rows = quadrature points, columns =
basis functions).
"""
function _weighted_eval_matrix(basis, quad_order::Int, a, b, ::Type{FT};
                               measure=nothing) where FT
    n = length(basis)
    ref_nodes, ref_weights = gauss_legendre(quad_order, FT)
    scale = (b - a) / 2
    shift = (a + b) / 2
    qnodes   = scale .* ref_nodes  .+ shift
    qweights = scale .* ref_weights

    if measure !== nothing
        for k in 1:quad_order
            qweights[k] *= measure(qnodes[k])
        end
    end

    sqw = sqrt.(qweights)
    A = zeros(FT, quad_order, n)
    for q in 1:quad_order, j in 1:n
        A[q, j] = sqw[q] * funeval(basis, j, qnodes[q])
    end
    A
end


"""
    orthogonalize_basis(basis::Dictionary;
                        measure=nothing, quad_order=nothing)

Orthonormalize `basis` with respect to the inner product
``\\langle f,g\\rangle = \\int f(x)\\,g(x)\\,w(x)\\,dx``
using unpivoted Householder QR factorization of the weighted evaluation
matrix.

The algorithm forms ``A_{qi} = \\sqrt{w_q}\\,\\varphi_i(x_q)`` (where
``x_q, w_q`` are Gauss-Legendre quadrature nodes and weights
incorporating the measure weight) and computes the thin QR factorization
``A = QR``.  Since ``A^\\top A = R^\\top R = G`` (the Gram matrix), the
upper-triangular ``R`` is the Cholesky factor of ``G`` and the
lower-triangular transformation ``T = R^{-\\top}`` satisfies
``T G T^\\top = I``.

**Why QR of ``A`` instead of Cholesky of ``G``?**  Forming
``G = A^\\top A`` squares the condition number
(``\\kappa(G) = \\kappa(A)^2``), so the Cholesky factor ``L`` of ``G``
carries errors at ``O(\\varepsilon\\,\\kappa(A)^2)``.  Householder QR
applied directly to ``A`` produces ``R`` with errors at
``O(\\varepsilon\\,\\kappa(A))`` — half the digit loss.

**Why unpivoted (not column-pivoted)?**  The continuation algorithm
slices the basis as `dict[1:k]` at each step and requires every prefix
``\\{\\psi_1, \\ldots, \\psi_k\\}`` to be a Chebyshev system.  A
lower-triangular ``T`` guarantees ``\\psi_i`` depends only on
``\\varphi_1, \\ldots, \\varphi_i``, preserving this structure.
Column pivoting would permute the basis ordering, destroying the
prefix-Chebyshev property.

Julia's `qr()` uses a generic Householder implementation that works
natively with `BigFloat`.

By default, the quadrature order is set to ``\\max(2n, 8)`` points,
which is more than sufficient for polynomial and smooth bases.  Pass
`quad_order` explicitly to override this (e.g. for oscillatory or
singular weight functions that need more points).

Returns `(new_basis, T_mat)` where

- `new_basis` is a [`GenericFunctionSet`](@ref) whose functions and
  derivatives are the orthonormalized linear combinations
  ``\\psi_i = \\sum_{j=1}^{i} T_{ij}\\,\\varphi_j``.
- `T_mat` is the lower-triangular transformation matrix ``R^{-\\top}``.

Pre-computed moments transform by the same matrix:
`new_moments = T_mat * old_moments`.

A warning is printed when the diagonal entries of ``R`` indicate that
the original basis is ill-conditioned, together with an estimate of the
number of decimal digits lost.

# Keyword arguments
- `measure`: callable ``w(x)`` for the measure ``d\\mu = w(x)\\,dx``.
  Default `nothing` (Lebesgue measure).
- `quad_order`: number of GL quadrature points.  Default
  ``\\max(2n, 8)``.
"""
function orthogonalize_basis(basis::Dictionary; measure=nothing, quad_order=nothing)
    n = length(basis)

    a = leftendpoint(support(basis))
    b = rightendpoint(support(basis))
    FT = typeof(a)                      # arithmetic type (e.g. BigFloat)

    q = quad_order !== nothing ? quad_order : max(2n, 8)
    A = _weighted_eval_matrix(basis, q, a, b, FT; measure)

    # Unpivoted Householder QR: A = Q R.
    # Julia's generic qr() works natively with BigFloat.
    # Since A is q×n with q ≥ n, R is n×n upper-triangular and satisfies
    # RᵀR = AᵀA = G (the Gram matrix).
    F = qr(A)
    R = Matrix(F.R)      # n×n upper-triangular

    # Ensure positive diagonal (Householder QR may negate rows of R).
    # This makes R agree with the Cholesky factor of G and keeps the
    # orthonormalized basis functions oriented consistently.
    for i in 1:n
        if R[i, i] < 0
            R[i, :] .*= -one(FT)
        end
    end

    # Transformation: T = R⁻ᵀ  (lower-triangular)
    # Since R is upper-triangular, Rᵀ is lower-triangular, and T = (Rᵀ)⁻¹
    # is also lower-triangular.  We solve Rᵀ T' = I via forward substitution.
    R_dense = Matrix(R)
    Id = zeros(FT, n, n)
    for i in 1:n; Id[i,i] = one(FT); end
    T_mat = R_dense' \ Id
    T_mat = LowerTriangular(T_mat) # enforce exact triangular structure

    # --- Conditioning diagnostics ---
    # The old diagonal-only estimate misses cases where the transformation
    # itself develops very large coefficients even though the diagonal spread
    # looks harmless.  Use the actual triangular change-of-basis map instead.
    T_dense = Matrix(T_mat)
    cond_T = cond(R_dense)                    # cond(T) == cond(R)
    digits_lost = log10(cond_T)
    working_digits = precision(FT) * log10(FT(2))  # bits → decimal digits
    reliable_digits = working_digits - digits_lost
    T_norm_inf = opnorm(T_dense, Inf)
    max_abs_T = maximum(abs.(T_dense))

    _sigfig(x; sigdigits::Integer=3) = round(Float64(x); sigdigits)
    if reliable_digits < 1
        @warn("Basis is nearly linearly dependent: orthogonalization is " *
              "expected to lose about $(_sigfig(digits_lost)) " *
              "decimal digits out of $(_sigfig(working_digits)) " *
              "available. cond(T) ≈ $(_sigfig(cond_T)), ‖T‖∞ ≈ $(_sigfig(T_norm_inf)), " *
              "max|Tᵢⱼ| ≈ $(_sigfig(max_abs_T)). The orthogonalized basis may have " *
              "essentially no reliable digits, and downstream routines such as " *
              "`compute_moments` or `compute_gauss_rule` may fail in $(FT) due " *
              "to severe cancellation. Consider increasing BigFloat precision " *
              "or using a better-conditioned basis.")
    elseif digits_lost > working_digits / 3
        @warn("Basis is ill-conditioned: orthogonalization is expected to lose " *
              "about $(_sigfig(digits_lost)) decimal digits " *
              "out of $(_sigfig(working_digits)) available. " *
              "cond(T) ≈ $(_sigfig(cond_T)), ‖T‖∞ ≈ $(_sigfig(T_norm_inf)), max|Tᵢⱼ| ≈ " *
              "$(_sigfig(max_abs_T)). Approximately $(_sigfig(reliable_digits)) " *
              "digits remain. The orthogonalized basis can be much harder to " *
              "evaluate stably than the original basis, and downstream routines " *
              "such as `compute_gauss_rule` may require higher precision.")
    end

    # Build orthonormalized functions: ψ_i(x) = Σ_{j=1}^{i} T_mat[i,j] φ_j(x)
    new_funs = [let coeffs = T_mat[i, 1:i]
        x -> sum(coeffs[j] * funeval(basis, j, x) for j in 1:length(coeffs))
    end for i in 1:n]

    # Build transformed derivatives: ψ'_i(x) = Σ_{j=1}^{i} T_mat[i,j] φ'_j(x)
    # (skip when the original basis has no derivative information)
    x_probe = (a + b) / 2
    max_deriv_order = 0
    while max_deriv_order < n &&
          all(i -> maybe_funeval_deriv(basis, i, x_probe, max_deriv_order + 1) !== nothing,
              1:n)
        max_deriv_order += 1
    end

    has_derivs = max_deriv_order >= 1
    new_derivs = if has_derivs
        [let coeffs = T_mat[i, 1:i]
            [let order = order
                x -> sum(coeffs[j] * maybe_funeval_deriv(basis, j, x, order)
                         for j in 1:length(coeffs))
             end for order in 1:max_deriv_order]
        end for i in 1:n]
    else
        nothing
    end

    new_basis = GenericFunctionSet(new_funs, new_derivs, a, b, digits_lost)
    new_basis, T_mat
end
