
# ============================================================================
# Chebyshev-system diagnostics (T-system and extended complete T-system)
#
# Extended Complete Chebyshev system diagnostics (`check_ECT_system`):
#
# Derivative computation strategy:
#
#   1. PRIMARY — Use analytic derivatives whenever they are available.
#
#   2. SECONDARY — For any missing derivative order, build a Chebyshev
#      interpolant on the highest available lower-order analytic derivative
#      (or on the function itself if no derivatives are available).  Missing
#      higher derivatives are then obtained from that interpolant via the
#      Chebyshev coefficient recurrence.
#
#   3. FALLBACK — When the Chebyshev coefficients for that quantity fail to
#      converge (typically because of an endpoint singularity such as √x at
#      x = 0), the code issues a warning and falls back to computing the
#      missing derivatives pointwise via Taylor arithmetic.  Evaluation always
#      occurs at interior points of (a, b), so the Taylor expansion is
#      well-defined whenever the function implementation supports it.
#
# The Wronskian criterion:
#   {f_1, ..., f_n} ⊂ Cⁿ⁻¹(a,b) is a positive Extended Complete Chebyshev
#   (ECT) system on [a, b] if and only if
#       W(f_1, ..., f_k)(x) > 0   for all x ∈ (a, b),  k = 1, ..., n.
#
# An ECT-system is in particular a T-system (Chebyshev system), which is the
# property required by the continuation algorithm for existence and uniqueness
# of generalized Gaussian quadrature rules.
#
# If some Wronskians are negative but of constant sign, the basis is still an
# ECT-system after flipping signs of individual basis functions (which does
# not change the span).  The critical failure is a Wronskian that changes
# sign, i.e. crosses zero on (a, b).
#
#
# Collocation / T-system sampling (`check_T_system`):
#   A simpler diagnostic for the ordinary Chebyshev (T-) system property only:
#   sample many ordered tuples x_1 < ⋯ < x_m ⊂ (a, b) and track the normalized
#   collocation determinant
#       det([f_i(x_j)]_{i,j=1}^m) / ∏_{i<j} (x_j − x_i).
#   Only values of the basis functions are needed (no derivatives).  For a
#   T-system this determinant keeps a fixed nonzero sign across increasing
#   m-tuples; observed sign changes or persistent near-zeros argue against the
#   property.  Like `check_ECT_system`, this is numerical evidence from random
#   sampling, not a mathematical certificate.
# ============================================================================


# ────────────────────────────────────────────────────────────────────────────
# Section 1: Taylor arithmetic (internal)
#
# _TaylorCoeffs{T} stores Taylor coefficients [a₀, a₁, …, aₙ] where
#     f(x₀ + t) = Σₖ aₖ tᵏ,    aₖ = f⁽ᵏ⁾(x₀) / k!.
#
# All standard arithmetic and elementary functions propagate these
# coefficients exactly (to floating-point precision), giving all
# derivatives of f at x₀ in a single evaluation.
# ────────────────────────────────────────────────────────────────────────────

struct _TaylorCoeffs{T}
    c::Vector{T}
end

_torder(t::_TaylorCoeffs) = length(t.c) - 1

# Construct the Taylor variable  x₀ + t  truncated at order n.
function _tvar(x0::T, n::Int) where {T}
    c = zeros(T, n + 1)
    c[1] = x0
    if n >= 1
        c[2] = one(T)
    end
    _TaylorCoeffs{T}(c)
end

# Constant (scalar lifted into Taylor arithmetic).
_tconst(x::T, n::Int) where {T} = _TaylorCoeffs{T}([x; zeros(T, n)])

# Extract actual derivatives  f(x₀), f′(x₀), …, f⁽ᵐ⁾(x₀).
function _extract_derivs(t::_TaylorCoeffs{T}, max_order::Int) where T
    derivs = zeros(T, max_order + 1)
    for k in 0:min(max_order, _torder(t))
        derivs[k+1] = t.c[k+1] * T(factorial(big(k)))
    end
    derivs
end

# --- Utilities ---------------------------------------------------------------
Base.one(t::_TaylorCoeffs{T})  where T = _tconst(one(T),  _torder(t))
Base.zero(t::_TaylorCoeffs{T}) where T = _tconst(zero(T), _torder(t))
Base.conj(t::_TaylorCoeffs) = t
Base.abs(t::_TaylorCoeffs)  = t.c[1] >= zero(eltype(t.c)) ? t : -t

Base.isnan(t::_TaylorCoeffs)  = isnan(t.c[1])
Base.isinf(t::_TaylorCoeffs)  = isinf(t.c[1])
Base.iszero(t::_TaylorCoeffs) = all(iszero, t.c)

Base.show(io::IO, t::_TaylorCoeffs{T}) where T =
    print(io, "_TaylorCoeffs{$T}(order=$(_torder(t)), a₀=$(t.c[1]))")

# --- Addition ----------------------------------------------------------------
function Base.:+(a::_TaylorCoeffs{T}, b::_TaylorCoeffs{T}) where T
    _TaylorCoeffs{T}(a.c .+ b.c)
end
function Base.:+(a::_TaylorCoeffs{T}, b::Number) where T
    a + _tconst(T(b), _torder(a))
end
function Base.:+(a::Number, b::_TaylorCoeffs{T}) where T
    _tconst(T(a), _torder(b)) + b
end

# --- Subtraction -------------------------------------------------------------
Base.:-(a::_TaylorCoeffs{T}) where T = _TaylorCoeffs{T}(.- a.c)

function Base.:-(a::_TaylorCoeffs{T}, b::_TaylorCoeffs{T}) where T
    _TaylorCoeffs{T}(a.c .- b.c)
end
function Base.:-(a::_TaylorCoeffs{T}, b::Number) where T
    a - _tconst(T(b), _torder(a))
end
function Base.:-(a::Number, b::_TaylorCoeffs{T}) where T
    _tconst(T(a), _torder(b)) - b
end

# --- Multiplication (Cauchy product) -----------------------------------------
function Base.:*(a::_TaylorCoeffs{T}, b::_TaylorCoeffs{T}) where T
    n = _torder(a)
    c = zeros(T, n + 1)
    for k in 0:n
        @inbounds for j in 0:k
            c[k+1] += a.c[j+1] * b.c[k-j+1]
        end
    end
    _TaylorCoeffs{T}(c)
end
Base.:*(a::_TaylorCoeffs{T}, b::Number) where T = _TaylorCoeffs{T}(a.c .* T(b))
Base.:*(a::Number, b::_TaylorCoeffs{T}) where T = _TaylorCoeffs{T}(T(a) .* b.c)

# --- Division ----------------------------------------------------------------
function Base.:/(a::_TaylorCoeffs{T}, b::_TaylorCoeffs{T}) where T
    n = _torder(a)
    c = zeros(T, n + 1)
    @assert !iszero(b.c[1]) "Taylor division: denominator has zero constant term"
    inv_b0 = inv(b.c[1])
    for k in 0:n
        s = a.c[k+1]
        @inbounds for j in 1:k
            s -= b.c[j+1] * c[k-j+1]
        end
        c[k+1] = s * inv_b0
    end
    _TaylorCoeffs{T}(c)
end
Base.:/(a::_TaylorCoeffs{T}, b::Number) where T = _TaylorCoeffs{T}(a.c ./ T(b))
function Base.:/(a::Number, b::_TaylorCoeffs{T}) where T
    _tconst(T(a), _torder(b)) / b
end

# --- Integer power -----------------------------------------------------------
function Base.:^(a::_TaylorCoeffs{T}, p::Integer) where T
    p == 0 && return one(a)
    p == 1 && return a
    p == 2 && return a * a
    p < 0  && return one(a) / a^(-p)
    # Binary exponentiation
    iseven(p) && (half = a^(p ÷ 2); return half * half)
    return a * a^(p - 1)
end

# --- Real power (via exp/log) ------------------------------------------------
function Base.:^(a::_TaylorCoeffs, p::Real)
    pi = round(Int, p)
    p == pi && return a^pi
    return exp(p * log(a))
end
Base.:^(a::_TaylorCoeffs, b::_TaylorCoeffs) = exp(b * log(a))

# --- Exponential -------------------------------------------------------------
#   g = exp(f):  g₀ = exp(f₀),  gₘ = (1/m) Σₖ₌₁ᵐ k fₖ gₘ₋ₖ
function Base.exp(a::_TaylorCoeffs{T}) where T
    n = _torder(a)
    g = zeros(T, n + 1)
    g[1] = exp(a.c[1])
    for m in 1:n
        s = zero(T)
        @inbounds for k in 1:m
            s += T(k) * a.c[k+1] * g[m-k+1]
        end
        g[m+1] = s / T(m)
    end
    _TaylorCoeffs{T}(g)
end

# --- Logarithm --------------------------------------------------------------
#   g = log(f):  g₀ = log(f₀),
#   gₘ = (1/f₀)[ fₘ − (1/m) Σₖ₌₁ᵐ⁻¹ k gₖ fₘ₋ₖ ]
function Base.log(a::_TaylorCoeffs{T}) where T
    n = _torder(a)
    @assert a.c[1] > zero(T) "Taylor log: leading coefficient must be positive"
    g = zeros(T, n + 1)
    g[1] = log(a.c[1])
    inv_a0 = inv(a.c[1])
    for m in 1:n
        s = zero(T)
        @inbounds for k in 1:m-1
            s += T(k) * g[k+1] * a.c[m-k+1]
        end
        g[m+1] = (a.c[m+1] - s / T(m)) * inv_a0
    end
    _TaylorCoeffs{T}(g)
end

# --- Square root -------------------------------------------------------------
#   g = √f:  g₀ = √f₀,  gₘ = (fₘ − Σₖ₌₁ᵐ⁻¹ gₖ gₘ₋ₖ) / (2g₀)
function Base.sqrt(a::_TaylorCoeffs{T}) where T
    n = _torder(a)
    @assert a.c[1] > zero(T) "Taylor sqrt: leading coefficient must be positive"
    g = zeros(T, n + 1)
    g[1] = sqrt(a.c[1])
    inv_2g0 = inv(T(2) * g[1])
    for m in 1:n
        s = zero(T)
        @inbounds for k in 1:m-1
            s += g[k+1] * g[m-k+1]
        end
        g[m+1] = (a.c[m+1] - s) * inv_2g0
    end
    _TaylorCoeffs{T}(g)
end

# --- Sin / Cos (computed together) -------------------------------------------
#   g = sin(f), h = cos(f):
#   gₘ₊₁ = (1/(m+1)) Σₖ₌₀ᵐ (k+1) fₖ₊₁ hₘ₋ₖ
#   hₘ₊₁ = −(1/(m+1)) Σₖ₌₀ᵐ (k+1) fₖ₊₁ gₘ₋ₖ
function _sincos_taylor(a::_TaylorCoeffs{T}) where T
    n = _torder(a)
    s = zeros(T, n + 1)
    c = zeros(T, n + 1)
    s[1] = sin(a.c[1])
    c[1] = cos(a.c[1])
    for m in 0:n-1
        ss = zero(T)
        sc = zero(T)
        @inbounds for k in 0:m
            fk1 = a.c[k+2]
            ss += T(k + 1) * fk1 * c[m-k+1]
            sc += T(k + 1) * fk1 * s[m-k+1]
        end
        s[m+2] =  ss / T(m + 1)
        c[m+2] = -sc / T(m + 1)
    end
    _TaylorCoeffs{T}(s), _TaylorCoeffs{T}(c)
end

Base.sin(a::_TaylorCoeffs) = _sincos_taylor(a)[1]
Base.cos(a::_TaylorCoeffs) = _sincos_taylor(a)[2]
Base.tan(a::_TaylorCoeffs) = sin(a) / cos(a)

# --- Comparisons (operate on constant term, for control flow) ----------------
Base.:<(a::_TaylorCoeffs, b::Number) = a.c[1] < b
Base.:<(a::Number, b::_TaylorCoeffs) = a < b.c[1]
Base.:<(a::_TaylorCoeffs, b::_TaylorCoeffs) = a.c[1] < b.c[1]
Base.:>(a::_TaylorCoeffs, b::Number) = a.c[1] > b
Base.:>(a::Number, b::_TaylorCoeffs) = a > b.c[1]
Base.isless(a::_TaylorCoeffs, b::_TaylorCoeffs) = isless(a.c[1], b.c[1])
Base.isless(a::_TaylorCoeffs, b::Number) = isless(a.c[1], b)
Base.isless(a::Number, b::_TaylorCoeffs) = isless(a, b.c[1])
Base.:(==)(a::_TaylorCoeffs, b::Number) = a.c[1] == b
Base.:(==)(a::Number, b::_TaylorCoeffs) = a == b.c[1]
Base.:>=(a::_TaylorCoeffs, b::Number) = a.c[1] >= b
Base.:>=(a::Number, b::_TaylorCoeffs) = a >= b.c[1]
Base.:<=(a::_TaylorCoeffs, b::Number) = a.c[1] <= b
Base.:<=(a::Number, b::_TaylorCoeffs) = a <= b.c[1]


# ────────────────────────────────────────────────────────────────────────────
# Section 2: Chebyshev approximation
# ────────────────────────────────────────────────────────────────────────────

"""
    _cheb_nodes(N, ::Type{T})

Chebyshev-I (root) nodes on `[-1, 1]`:  xⱼ = cos(π(2j−1)/(2N)),  j = 1…N.
"""
function _cheb_nodes(N::Int, ::Type{T}) where T
    [cos(T(π) * T(2j - 1) / T(2N)) for j in 1:N]
end


"""
    _cheb_coefficients(fvals::Vector{T}) where T

Compute Chebyshev coefficients from function values at Chebyshev-I nodes.
Direct O(N²) computation (BigFloat-compatible, no FFT required).

Returns coefficients `c` such that  f(x) ≈ Σₖ c[k+1] Tₖ(x)  on [-1, 1],
using the convention where c₀ already absorbs the standard 1/2 factor
(i.e. no halving in the expansion).
"""
function _cheb_coefficients(fvals::Vector{T}) where T
    N = length(fvals)
    coeffs = Vector{T}(undef, N)
    for k in 0:N-1
        s = zero(T)
        for j in 1:N
            s += fvals[j] * cos(T(k) * T(π) * T(2j - 1) / T(2N))
        end
        coeffs[k+1] = (k == 0 ? one(T) : T(2)) * s / T(N)
    end
    coeffs
end


struct _ChebApproxInfo{T}
    coeffs::Vector{T}
    converged::Bool
    truncation_bound::T
end


"""
    _cheb_approximate(f, a::T, b::T; tol, N_init, N_max) where T

Adaptively approximate `f` on `[a, b]` by a Chebyshev series on `[-1, 1]`.
Doubles the number of nodes until the tail coefficients decay below `tol`
(relative to the maximum coefficient).

Returns `(coeffs, converged::Bool)`.
"""
function _cheb_approximate_info(f, a::T, b::T;
                                tol::T  = T(4) * eps(T),
                                N_init::Int = 32,
                                N_max::Int  = 2048) where T
    mid  = (a + b) / 2
    half = (b - a) / 2

    N = N_init
    while N <= N_max
        nodes = _cheb_nodes(N, T)
        fvals = T[f(mid + half * x) for x in nodes]
        coeffs = _cheb_coefficients(fvals)

        scale = maximum(abs, coeffs)
        if scale == zero(T)
            return _ChebApproxInfo(coeffs[1:1], true, zero(T))
        end

        # Convergence: last third of coefficients negligible.
        # The O(N²) DCT accumulates roundoff ~ N * eps(T), so use an
        # N-dependent noise floor to avoid false non-convergence.
        tail_start = max(1, 2N ÷ 3)
        tail_max   = maximum(abs, @view coeffs[tail_start:end])
        noise_floor = T(10 * N) * eps(T) * scale
        threshold   = max(tol * scale, noise_floor)

        if tail_max < threshold
            last_sig = something(
                findlast(i -> abs(coeffs[i]) > threshold, 1:N), 1)
            last_sig = max(last_sig, 2)         # keep ≥ 2 for differentiation
            tail_bound = last_sig < N ? sum(abs, @view coeffs[last_sig+1:end]) :
                                        zero(T)
            return _ChebApproxInfo(coeffs[1:last_sig], true, tail_bound)
        end

        N *= 2
    end

    # Did not converge
    _ChebApproxInfo(T[], false, T(Inf))
end

function _cheb_approximate(f, a::T, b::T;
                           tol::T  = T(4) * eps(T),
                           N_init::Int = 32,
                           N_max::Int  = 2048) where T
    info = _cheb_approximate_info(f, a, b; tol=tol, N_init=N_init, N_max=N_max)
    info.coeffs, info.converged
end


"""
    _cheb_differentiate(c::Vector{T}) where T

Differentiate Chebyshev coefficients on `[-1, 1]`.

Given `c` with  f = Σₖ c[k+1] Tₖ,  returns `d` with  f′ = Σₖ d[k+1] Tₖ.
Uses the standard recurrence:
    d_N = d_{N+1} = 0
    dₖ = dₖ₊₂ + 2(k+1) cₖ₊₁   for k = N−1, …, 1
    d₀ = d₂/2 + c₁
"""
function _cheb_differentiate(c::Vector{T}) where T
    N = length(c) - 1                   # degree of input
    N <= 0 && return T[zero(T)]

    d = zeros(T, N)                     # d[k+1] stores dₖ, k = 0…N-1

    # k = N-1 (start of recurrence; d_{N+1} = d_N = 0)
    # Guard: for N=1, d[1] is the d₀ special case handled below.
    if N >= 2
        d[N] = T(2N) * c[N+1]
    end

    # k = N-2 down to 1
    for k in (N-2):-1:1
        dk2 = (k + 3 <= N) ? d[k+3] : zero(T)      # d_{k+2}
        d[k+1] = dk2 + T(2) * T(k + 1) * c[k+2]
    end

    # k = 0 (special)
    d2 = (N >= 3) ? d[3] : zero(T)
    d[1] = d2 / 2 + (length(c) >= 2 ? c[2] : zero(T))

    d
end


"""
    _cheb_evaluate(c::Vector{T}, x::T) where T

Evaluate a Chebyshev series  f(x) = Σₖ c[k+1] Tₖ(x)  via Clenshaw recurrence.
"""
function _cheb_evaluate(c::Vector{T}, x::T) where T
    N = length(c)
    N == 0 && return zero(T)
    N == 1 && return c[1]

    b_next1 = zero(T)                   # b_{k+1}
    b_next2 = zero(T)                   # b_{k+2}

    for k in (N-1):-1:1
        bk = c[k+1] + T(2) * x * b_next1 - b_next2
        b_next2 = b_next1
        b_next1 = bk
    end

    c[1] + x * b_next1 - b_next2
end


# ────────────────────────────────────────────────────────────────────────────
# Section 3: Derivative engine
# ────────────────────────────────────────────────────────────────────────────

struct _DerivativeSurrogate{T}
    mode::Symbol
    base_order::Int
    max_shift::Int
    deriv_coeffs::Vector{Vector{T}}
end

"""
    _FuncDerivInfo{T}

Per-function derivative recovery plan on `[a, b]`.
"""
struct _FuncDerivInfo{T}
    exact_available::BitVector
    missing_base_order::Vector{Int}
    surrogates::Vector{Union{Nothing,_DerivativeSurrogate{T}}}
end

function _derivative_callable(basis, j::Int, order::Int)
    if order == 0
        return x -> funeval(basis, j, x)
    end
    return x -> begin
        deriv = maybe_funeval_deriv(basis, j, x, order)
        deriv === nothing &&
            error("Lost analytic derivative of order $order for basis function $j")
        deriv
    end
end

function _highest_contiguous_exact_order(exact_available::AbstractVector{Bool})
    max_order = length(exact_available) - 1
    order = 0
    while order < max_order && exact_available[order + 2]
        order += 1
    end
    order
end

_missing_orders(exact_available::AbstractVector{Bool}) =
    [order for order in 1:length(exact_available)-1 if !exact_available[order+1]]

function _warn_numeric_derivatives(exact_available::BitVector, a, b)
    missing_orders = _missing_orders(exact_available)
    isempty(missing_orders) && return

    highest_contig = _highest_contiguous_exact_order(exact_available)
    trailing_missing = collect(highest_contig+1:length(exact_available)-1)

    if missing_orders == trailing_missing
        if highest_contig == 0
            @warn("No analytic derivatives are available " *
                  "on [$(_fmt_short(a)), $(_fmt_short(b))].  Missing " *
                  "derivatives will be approximated numerically, so any " *
                  "Wronskian certification using them is numerical.")
        else
            @warn("Analytic derivatives are available " *
                  "through order $highest_contig on [$(_fmt_short(a)), " *
                  "$(_fmt_short(b))], but higher orders are missing.  " *
                  "Missing derivatives will be approximated numerically, so " *
                  "any Wronskian certification using them is numerical.")
        end
    else
        orders_str = join(string.(missing_orders), ", ")
        @warn("Analytic derivatives are missing for " *
              "orders $orders_str on [$(_fmt_short(a)), $(_fmt_short(b))].  " *
              "Missing derivatives will be approximated numerically, so any " *
              "Wronskian certification using them is numerical.")
    end
end

function _supports_taylor_callable(g, x_probe::T; order::Int = 1) where T
    try
        g(_tvar(x_probe, max(order, 1)))
        true
    catch e
        e isa MethodError || rethrow(e)
        false
    end
end

function _prepare_single_function_derivs(basis, j::Int, a::T, b::T, max_order::Int;
                                         tol::T = T(4) * eps(T),
                                         N_init::Int = 32,
                                         N_max::Int = 2048,
                                         emit_warnings::Bool = true) where T
    x_probe = (a + b) / 2

    exact_available = falses(max_order + 1)
    exact_available[1] = true
    for order in 1:max_order
        exact_available[order+1] =
            maybe_funeval_deriv(basis, j, x_probe, order) !== nothing
    end

    missing_base_order = fill(-1, max_order + 1)
    surrogates = Vector{Union{Nothing,_DerivativeSurrogate{T}}}(undef, max_order + 1)
    fill!(surrogates, nothing)

    emit_warnings && _warn_numeric_derivatives(exact_available, a, b)

    missing_orders = _missing_orders(exact_available)
    isempty(missing_orders) &&
        return _FuncDerivInfo{T}(exact_available, missing_base_order, surrogates)

    candidate_modes = fill(:unknown, max_order + 1)
    candidate_coeffs = Vector{Union{Nothing,Vector{T}}}(undef, max_order + 1)
    fill!(candidate_coeffs, nothing)

    function _candidate_capability(base_order::Int)
        idx = base_order + 1
        candidate_modes[idx] != :unknown &&
            return candidate_modes[idx], candidate_coeffs[idx]

        g = _derivative_callable(basis, j, base_order)
        info = _cheb_approximate_info(g, a, b; tol=tol, N_init=N_init, N_max=N_max)
        if info.converged
            candidate_modes[idx] = :chebyshev
            candidate_coeffs[idx] = info.coeffs
        elseif _supports_taylor_callable(g, x_probe)
            candidate_modes[idx] = :taylor
        else
            candidate_modes[idx] = :none
        end
        candidate_modes[idx], candidate_coeffs[idx]
    end

    preferred_base_order = fill(-1, max_order + 1)
    for order in missing_orders
        preferred_idx = something(findlast(identity, @view exact_available[1:order]), 1)
        preferred_base = preferred_idx - 1
        preferred_base_order[order+1] = preferred_base

        chosen_base = -1
        for base_order in preferred_base:-1:0
            exact_available[base_order+1] || continue
            mode, _ = _candidate_capability(base_order)
            if mode != :none
                chosen_base = base_order
                break
            end
        end

        chosen_base >= 0 ||
            error("Unable to recover derivative order $order for basis " *
                  "function $j on [$(_fmt_short(a)), $(_fmt_short(b))].")
        missing_base_order[order+1] = chosen_base
    end

    needed_bases = sort(unique([missing_base_order[order+1] for order in missing_orders]))
    for base_order in needed_bases
        max_shift = maximum(order - base_order for order in missing_orders
                            if missing_base_order[order+1] == base_order)
        mode, coeffs = _candidate_capability(base_order)

        if mode == :chebyshev
            coeffs === nothing &&
                error("Missing Chebyshev coefficients for basis function $j " *
                      "derivative order $base_order.")
            dc = Vector{Vector{T}}(undef, max_shift + 1)
            dc[1] = coeffs
            c = coeffs
            for shift in 1:max_shift
                c = _cheb_differentiate(c)
                dc[shift+1] = c
            end
            surrogates[base_order+1] =
                _DerivativeSurrogate{T}(:chebyshev, base_order, max_shift, dc)
        elseif mode == :taylor
            if emit_warnings
                fallback_orders = [order for order in missing_orders
                                   if missing_base_order[order+1] == base_order]
                preferred_orders = unique([preferred_base_order[order+1]
                                           for order in fallback_orders])
                preferred_str = join(string.(preferred_orders), ", ")
                @warn("Chebyshev approximation did not converge for basis " *
                      "function $j on the preferred recovery derivative " *
                      "order(s) $preferred_str over [$(_fmt_short(a)), " *
                      "$(_fmt_short(b))].  Falling back to Taylor-mode AD " *
                      "from derivative order $base_order.  This fallback " *
                      "requires the function or derivative implementation " *
                      "to accept `_TaylorCoeffs` inputs.")
            end
            surrogates[base_order+1] =
                _DerivativeSurrogate{T}(:taylor, base_order, max_shift,
                                        Vector{Vector{T}}())
        else
            error("Unable to build a recovery surrogate for basis function " *
                  "$j derivative order $base_order on " *
                  "[$(_fmt_short(a)), $(_fmt_short(b))].")
        end
    end

    _FuncDerivInfo{T}(exact_available, missing_base_order, surrogates)
end

"""
    _prepare_basis_derivs(basis, n, a, b, max_order; tol, N_init, N_max)

For each of the first `n` basis functions, record exactly which derivative
orders are available analytically up to `max_order`.  Any missing order is
recovered numerically from the highest available lower-order derivative:
Chebyshev interpolation first, Taylor-mode AD only if that interpolation
does not converge.

Returns `Vector{_FuncDerivInfo{T}}` of length `n`.
"""
function _prepare_basis_derivs(basis, n::Int, a::T, b::T, max_order::Int;
                               tol::T   = T(4) * eps(T),
                               N_init::Int = 32,
                               N_max::Int  = 2048,
                               emit_warnings::Bool = true) where T
    func_data = Vector{_FuncDerivInfo{T}}(undef, n)

    for j in 1:n
        func_data[j] = _prepare_single_function_derivs(basis, j, a, b, max_order;
                                                       tol=tol,
                                                       N_init=N_init,
                                                       N_max=N_max,
                                                       emit_warnings=emit_warnings && j == 1)
    end

    func_data
end

function _fill_derivatives_row!(row::AbstractVector{T}, basis, j::Int, x::T,
                                max_order::Int, info::_FuncDerivInfo{T},
                                a::T, b::T, scale::T) where T
    row[1] = funeval(basis, j, x)
    x_ref = (T(2) * x - (a + b)) / (b - a)

    taylor_cache = Vector{Union{Nothing,Vector{T}}}(undef, length(info.surrogates))
    fill!(taylor_cache, nothing)

    for order in 1:max_order
        if info.exact_available[order+1]
            deriv = maybe_funeval_deriv(basis, j, x, order)
            deriv === nothing &&
                error("Lost analytic derivative of order $order for basis function $j")
            row[order+1] = deriv
            continue
        end

        base_order = info.missing_base_order[order+1]
        base_order < 0 &&
            error("No derivative recovery path for basis function $j order $order")
        surrogate = info.surrogates[base_order+1]
        surrogate === nothing &&
            error("No derivative surrogate for basis function $j order $order")
        shift = order - base_order

        if surrogate.mode == :chebyshev
            row[order+1] =
                _cheb_evaluate(surrogate.deriv_coeffs[shift+1], x_ref) * scale^shift
        else
            cached = taylor_cache[base_order+1]
            if cached === nothing
                g = _derivative_callable(basis, j, base_order)
                t = _tvar(x, surrogate.max_shift)
                result = try
                    g(t)
                catch e
                    if e isa MethodError
                        error("Taylor-mode AD could not evaluate basis " *
                              "function $j derivative order $base_order at " *
                              "x = $(_fmt_short(x)).  This fallback requires " *
                              "methods that accept `_TaylorCoeffs` inputs.")
                    end
                    rethrow(e)
                end
                cached = _extract_derivs(result, surrogate.max_shift)
                taylor_cache[base_order+1] = cached
            end
            row[order+1] = cached[shift+1]
        end
    end

    row
end


"""
    _derivative_table(basis, x, n, max_order, func_data, a, b, scale)

Build the `n × (max_order + 1)` derivative table
`D[j, m+1] = f_j^{(m)}(x)` using analytic derivatives, Chebyshev data,
or Taylor-mode AD.
"""
function _derivative_table(basis, x::T, n::Int, max_order::Int,
                           func_data::Vector{_FuncDerivInfo{T}},
                           a::T, b::T, scale::T) where T
    D = zeros(T, n, max_order + 1)

    for j in 1:n
        _fill_derivatives_row!(view(D, j, :), basis, j, x, max_order,
                               func_data[j], a, b, scale)
    end

    D
end


# ────────────────────────────────────────────────────────────────────────────
# Section 4: Wronskian computation
# ────────────────────────────────────────────────────────────────────────────

"""
    wronskian(basis, k, x)

Compute the Wronskian determinant of the first `k` basis functions at `x`:

    W(f_1, …, f_k)(x) = det [f_j^{(i-1)}(x)]_{i,j=1}^{k}

Uses analytic derivatives whenever available and recovers only the missing
orders numerically from the highest available lower-order derivative.
"""
function wronskian(basis, k::Int, x)
    T = typeof(x)
    max_order = k - 1
    a = T(leftendpoint(support(basis)))
    b = T(rightendpoint(support(basis)))
    scale = T(2) / (b - a)
    func_data = _prepare_basis_derivs(basis, k, a, b, max_order;
                                      emit_warnings=false)
    D = _derivative_table(basis, x, k, max_order, func_data, a, b, scale)

    _wronskian_from_table(D, k)
end

_wronskian_from_table(D, k::Int) = det(@view D[1:k, 1:k])

function _wronskian_prime_from_table(D, k::Int)
    M = Matrix(@view D[1:k, 1:k])
    total = zero(eltype(D))
    for col in 1:k
        Mc = copy(M)
        Mc[:, col] .= @view D[1:k, col+1]
        total += det(Mc)
    end
    total
end


"""
    _function_derivs(basis, j, x, max_order)

Compute [f_j(x), f_j′(x), …, f_j^{(max_order)}(x)] using analytic
derivatives whenever available and recovering only the missing orders
numerically.
"""
function _function_derivs(basis, j::Int, x::T, max_order::Int) where T
    a = T(leftendpoint(support(basis)))
    b = T(rightendpoint(support(basis)))
    scale = T(2) / (b - a)
    derivs = zeros(T, max_order + 1)
    info = _prepare_single_function_derivs(basis, j, a, b, max_order;
                                           emit_warnings=false)
    _fill_derivatives_row!(derivs, basis, j, x, max_order, info, a, b, scale)
    derivs
end


# ────────────────────────────────────────────────────────────────────────────
# Section 5: Sampling and minimization
# ────────────────────────────────────────────────────────────────────────────

"""
    _chebyshev_points(a, b, n, ::Type{T})

Generate `n` Chebyshev nodes of the first kind on the open interval `(a, b)`.
These cluster near the endpoints (good for detecting boundary issues) and
avoid the exact endpoints.
"""
function _chebyshev_points(a, b, n::Int, ::Type{T}) where T
    mid  = (a + b) / 2
    half = (b - a) / 2
    [mid + half * cos(T(π) * (2k - 1) / (2n)) for k in 1:n]
end


"""
    _golden_section_min(f, a, b; tol, maxiter=200)

Find the minimum of a unimodal function `f` on `[a, b]` via golden-section
search.  Returns `(x_min, f_min)`.
"""
function _golden_section_min(f, a::T, b::T;
                             tol::T = (b - a) * eps(T)^(one(T)/3),
                             maxiter::Int = 200) where T
    φ = (sqrt(T(5)) - 1) / 2

    x1 = b - φ * (b - a)
    x2 = a + φ * (b - a)
    f1 = f(x1)
    f2 = f(x2)

    for _ in 1:maxiter
        (b - a) <= tol && break

        if f1 < f2
            b  = x2
            x2 = x1;  f2 = f1
            x1 = b - φ * (b - a)
            f1 = f(x1)
        else
            a  = x1
            x1 = x2;  f1 = f2
            x2 = a + φ * (b - a)
            f2 = f(x2)
        end
    end

    xm = (a + b) / 2
    xm, f(xm)
end


"""
    _refine_minimum(f, pts, vals; n_intervals=5)

Given sample points `pts` and values `vals = f.(pts)`, identify the
`n_intervals` intervals around the smallest values and resample each interval
on a denser Chebyshev grid.  Returns `(x_min, f_min)`.
"""
function _refine_minimum(f, pts::Vector{T}, vals::Vector{T};
                         n_intervals::Int = 5,
                         n_subsamples::Int = 33) where T
    n = length(pts)

    abs_vals = abs.(vals)
    sorted_idx = sortperm(abs_vals)

    seen_intervals = Set{Int}()
    best_x = pts[sorted_idx[1]]
    best_f = abs_vals[sorted_idx[1]]

    for idx in sorted_idx
        for ii in max(1, idx-1):min(n-1, idx)
            ii in seen_intervals && continue
            length(seen_intervals) >= n_intervals && break
            push!(seen_intervals, ii)

            for x in _chebyshev_points(pts[ii], pts[ii+1], n_subsamples, T)
                f_val = abs(f(x))
                if f_val < best_f
                    best_f = f_val
                    best_x = x
                end
            end
        end
        length(seen_intervals) >= n_intervals && break
    end

    best_x, best_f
end


function _lipschitz_upper_bound(fprime, a::T, b::T;
                                tol::T = T(4) * eps(T),
                                N_init::Int = 64,
                                N_max::Int = 4096) where T
    info = _cheb_approximate_info(fprime, a, b; tol=tol, N_init=N_init,
                                  N_max=N_max)
    info.converged || return T(Inf), false
    sum(abs, info.coeffs) + info.truncation_bound, true
end

function _func_deriv_method_summary(info::_FuncDerivInfo)
    max_order = length(info.exact_available) - 1
    missing_orders = _missing_orders(info.exact_available)
    isempty(missing_orders) && return "Analytic"

    exact_orders = [order for order in 1:max_order if info.exact_available[order+1]]
    pieces = String[]
    if isempty(exact_orders)
        push!(pieces, "Function values only")
    else
        push!(pieces, "Analytic orders " * join(string.(exact_orders), ", "))
    end

    for surrogate in info.surrogates
        surrogate === nothing && continue
        mode = surrogate.mode == :chebyshev ? "Chebyshev" : "Taylor AD"
        base_str = surrogate.base_order == 0 ? "values" :
                   "order $(surrogate.base_order)"
        lo = surrogate.base_order + 1
        hi = surrogate.base_order + surrogate.max_shift
        range_str = lo == hi ? "order $lo" : "orders $lo:$hi"
        detail = surrogate.mode == :chebyshev ?
                 " (N=$(length(surrogate.deriv_coeffs[1])))" : ""
        push!(pieces, "$mode from $base_str for $range_str$detail")
    end

    join(pieces, "; ")
end


# ────────────────────────────────────────────────────────────────────────────
# Section 6: Main diagnostic
# ────────────────────────────────────────────────────────────────────────────

"""
    WronskianInfo{T}

Diagnostic result for a single initial-segment Wronskian ``W_k``.
"""
struct WronskianInfo{T}
    "Number of basis functions in this Wronskian."
    k::Int
    "Constant sign on sampled points: +1, -1, or 0 (sign change detected)."
    sign::Int
    "Minimum absolute value of W_k found (after adaptive refinement)."
    min_abs::T
    "Location where the minimum absolute value occurs."
    min_loc::T
    "Maximum absolute value of W_k (for relative scale)."
    max_abs::T
    "Chebyshev-coefficient upper bound for |W_k'| on the sampled interval."
    lipschitz_upper_bound::T
    "Estimated minimum of W_k between sample points via Lipschitz bound."
    lipschitz_lower_bound::T
    "True when the sign is constant on samples and the Lipschitz bound proves no crossing."
    certified::Bool
end


"""
    BasisCheckResult{T}

Result of `check_ECT_system`: collects per-Wronskian diagnostics and an overall
verdict.  `is_ect` is conservative: it is true only when every Wronskian has
constant sampled sign and the Lipschitz certificate proves that no zero
crossing occurs between adjacent sample points.
"""
struct BasisCheckResult{T}
    "Number of basis functions."
    n::Int
    "Support interval."
    a::T
    b::T
    "Per-Wronskian diagnostics, indexed by k = 1, …, n."
    wronskians::Vector{WronskianInfo{T}}
    "True if all sampled Wronskians have constant sign."
    sampled_constant_sign::Bool
    "Overall verdict: true if all Wronskians are certified to be of constant sign."
    is_ect::Bool
end

function Base.show(io::IO, r::BasisCheckResult)
    status = if r.is_ect
        "PASS (ECT-system)"
    elseif r.sampled_constant_sign
        "FAIL (uncertified)"
    else
        "FAIL (not ECT)"
    end
    print(io, "BasisCheckResult(n=$(r.n), " *
          "[$(_fmt_short(r.a)), $(_fmt_short(r.b))], $status)")
end


"""
    check_ECT_system(basis; n_points=200, verbose=true)

Numerically verify whether `basis` forms an Extended Complete Chebyshev
(ECT) system on its support interval `(a, b)` by evaluating all
initial-segment Wronskians.

By the Wronskian criterion for ECT systems, the basis is a
positive ECT-system if and only if

    W(f_1, …, f_k)(x) > 0   for all x ∈ (a, b),   k = 1, …, n.

If a Wronskian is negative but of constant sign, the basis is still an
ECT-system (the sign can be corrected by flipping a basis function without
changing the span).  A sign change (zero crossing) means the basis is
**not** an ECT-system.

# Derivative computation

1. **Analytic derivatives**: every derivative order supplied by the basis is
   used directly.
2. **Chebyshev interpolation**: any missing derivative order is recovered
   from a Chebyshev interpolant built on the highest available lower-order
   analytic derivative (or on the function itself if no derivatives are
   available).
3. **Taylor-mode AD fallback**: when that Chebyshev interpolant does not
   converge (e.g. endpoint singularities), the missing derivatives are
   computed pointwise via Taylor arithmetic at interior points.

# Sampling strategy

1. **Chebyshev nodes** on `(a, b)`: cluster near endpoints where Wronskians
   often become small.
2. **Adaptive refinement**: resample the most suspicious intervals to lower
   the observed minimum of `|W_k|`.
3. **Lipschitz certificate**: build a Chebyshev surrogate for `W_k'`; if
   `min |W_k| > L × (max spacing)/2`, no zero crossing can exist between
   sample points.

# Arguments

- `basis`: a `Dictionary` (BasisFunctions) or `GenericFunctionSet`.
- `n_points::Int = 200`: number of Chebyshev sample points.
- `verbose::Bool = true`: print a diagnostic table.

# Returns

A [`BasisCheckResult`](@ref) containing per-Wronskian diagnostics.
"""
function check_ECT_system(basis; n_points::Int = 200, verbose::Bool = true)
    n = length(basis)
    wronskian_order = max(n - 1, 0)
    certification_order = n

    endpoint_a = leftendpoint(support(basis))
    endpoint_b = rightendpoint(support(basis))
    FT = promote_type(codomaintype(basis), typeof(endpoint_a), typeof(endpoint_b))
    if !(FT <: AbstractFloat)
        FT = BigFloat          # safe fallback
    end
    a = FT(endpoint_a)
    b = FT(endpoint_b)

    # --- Precompute Chebyshev / Taylor derivative data for each function ---
    func_data = _prepare_basis_derivs(basis, n, a, b, certification_order)
    scale = FT(2) / (b - a)

    # Margin: keep sample points slightly inside (a, b)
    margin = (b - a) * eps(FT)^(one(FT) / 3)
    a_inner = a + margin
    b_inner = b - margin

    if a_inner >= b_inner
        a_inner = a + (b - a) / 100
        b_inner = b - (b - a) / 100
    end

    # --- Generate sample points ---
    pts = sort(_chebyshev_points(a_inner, b_inner, n_points, FT))
    max_spacing = maximum(pts[qi+1] - pts[qi] for qi in 1:n_points-1)

    # --- Precompute derivative tables at all sample points ---
    D_tables = [_derivative_table(basis, pts[q], n, wronskian_order,
                                  func_data, a, b, scale)
                for q in 1:n_points]

    # --- Evaluate all Wronskians ---
    results = WronskianInfo{FT}[]

    if verbose
        println()
        println("  Wronskian criterion check for ECT-system property")
        println("  Basis: $n functions on [$(_fmt_short(a)), $(_fmt_short(b))]")

        # Report derivative method per function
        for j in 1:n
            println("    f_$j: $(_func_deriv_method_summary(func_data[j]))")
        end

        println("  Sample points: $n_points Chebyshev nodes on " *
                "[$(_fmt_short(a_inner)), $(_fmt_short(b_inner))]")
        println()
        println("  ┌──────┬──────────┬────────────────┬────────────────┬──────────────────┬──────────────┐")
        println("  │   k  │   sign   │    min |W_k|   │    max |W_k|   │   min location   │   Lipschitz  │")
        println("  ├──────┼──────────┼────────────────┼────────────────┼──────────────────┼──────────────┤")
    end

    overall_sampled_pass = true
    overall_pass = true

    for k in 1:n
        # Extract Wronskian values from precomputed derivative tables
        wvals = Vector{FT}(undef, n_points)
        for q in 1:n_points
            wvals[q] = _wronskian_from_table(D_tables[q], k)
        end

        # Sign analysis
        pos_count = count(w -> w > 0, wvals)
        neg_count = count(w -> w < 0, wvals)

        if pos_count == n_points
            wsign = +1
        elseif neg_count == n_points
            wsign = -1
        else
            wsign = 0
            overall_sampled_pass = false
            overall_pass = false
        end

        # Absolute values for minimum-finding
        abs_wvals = abs.(wvals)
        max_abs = maximum(abs_wvals)

        # Adaptive refinement to find the true minimum of |W_k|
        eval_abs_wk = let kk = k
            x -> begin
                D = _derivative_table(basis, x, kk, kk, func_data, a, b, scale)
                abs(_wronskian_from_table(D, kk))
            end
        end

        min_loc, min_abs = _refine_minimum(eval_abs_wk, pts, abs_wvals)

        eval_wkprime = let kk = k
            x -> begin
                D = _derivative_table(basis, x, kk, kk, func_data, a, b, scale)
                _wronskian_prime_from_table(D, kk)
            end
        end
        lip_upper, lip_available = _lipschitz_upper_bound(eval_wkprime,
                                                          a_inner, b_inner;
                                                          N_max=max(4096, 16 * n_points))
        lip_lower = lip_available ? min_abs - lip_upper * max_spacing / 2 :
                                    -FT(Inf)
        certified = wsign != 0 && lip_available && lip_lower > 0
        certified || (overall_pass = false)

        info = WronskianInfo{FT}(k, wsign, min_abs, min_loc, max_abs,
                                 lip_upper, lip_lower, certified)
        push!(results, info)

        if verbose
            sign_str = wsign == 1  ? "   +    " :
                       wsign == -1 ? "   -    " : " CHANGE "
            lip_str  = certified ? "  certified  " :
                       lip_available ? " uncertified " :
                                       "   no bound   "
            min_str = _fmt_sci(min_abs)
            max_str = _fmt_sci(max_abs)
            loc_str = _fmt_loc(min_loc)

            println("  │ $(lpad(k, 4)) │ $sign_str │ $(rpad(min_str, 14)) │ " *
                    "$(rpad(max_str, 14)) │ $(rpad(loc_str, 16)) │ $lip_str │")
        end

        # Warnings
        if wsign == 0
            @warn("Wronskian W_$k changes sign on ($(_fmt_short(a)), " *
                  "$(_fmt_short(b))): the basis is NOT an ECT-system.")
        elseif !lip_available
            @warn("Could not certify Wronskian W_$k on ($(_fmt_short(a)), " *
                  "$(_fmt_short(b))): the sampled values had constant sign, " *
                  "but the Chebyshev surrogate for W_$k' did not converge, " *
                  "so no Lipschitz no-zero-crossing bound was obtained.")
        elseif !certified
            @warn("Wronskian W_$k has constant sampled sign but is not " *
                  "certified on ($(_fmt_short(a)), $(_fmt_short(b))): " *
                  "the minimum sampled magnitude is too small relative to " *
                  "the Lipschitz bound to rule out a zero crossing between " *
                  "sample points.")
        elseif min_abs > 0 && max_abs > 0
            rel = min_abs / max_abs
            if rel < eps(FT)^(one(FT)/3)
                @warn("Wronskian W_$k is very small near x = " *
                      "$(_fmt_short(min_loc)): min/max ratio ≈ " *
                      "$(Float64(rel)).  The basis may be " *
                      "ill-conditioned or close to losing the " *
                      "Chebyshev property.")
            end
        end
    end

    if verbose
        println("  └──────┴──────────┴────────────────┴────────────────┴──────────────────┴──────────────┘")
        println()
        if overall_pass
            println("  Result: Basis forms a certified ECT-system " *
                    "(all Wronskians are sign-definite and certified).")
        elseif overall_sampled_pass
            println("  Result: All sampled Wronskians have constant sign, " *
                    "but the basis is not certified as an ECT-system.")
        else
            println("  Result: Basis does NOT form an ECT-system " *
                    "(sign change detected in at least one Wronskian).")
        end
        println()
    end

    BasisCheckResult{FT}(n, a, b, results, overall_sampled_pass, overall_pass)
end


# ────────────────────────────────────────────────────────────────────────────
# Section 7: User-supplied derivative diagnostic
# ────────────────────────────────────────────────────────────────────────────

"""
    check_basis_derivs(basis; step_size=nothing, tol=nothing, num_samples=10, verbose=true, rng=Random.default_rng())

Check the supplied first derivative for every basis function by comparing it
against a centered finite-difference approximation at random interior points.

The tolerance is a combined absolute-relative tolerance:

    abs(finite_difference - supplied_derivative) <= tol * max(1, abs(finite_difference), abs(supplied_derivative))

# Arguments

- `basis`: a `Dictionary` (BasisFunctions) or `GenericFunctionSet`.
- `step_size`: finite-difference spacing.  The default is based on machine
  precision and capped by the interval width.
- `tol`: comparison tolerance.  The default is `sqrt(eps(T))`, where `T` is
  the floating-point type used for the support interval.
- `num_samples::Int = 10`: random samples per basis function.
- `verbose::Bool = true`: print a success message on pass and warn on failure.
- `rng::AbstractRNG = Random.default_rng()`: random number generator.

# Returns

`true` when all sampled derivative values match the finite-difference
approximation, and `false` otherwise.
"""
function check_basis_derivs(basis;
                            step_size = nothing,
                            tol = nothing,
                            num_samples::Int = 10,
                            verbose::Bool = true,
                            rng::AbstractRNG = Random.default_rng())
    num_samples >= 1 ||
        throw(ArgumentError("num_samples must be positive."))

    endpoint_a = leftendpoint(support(basis))
    endpoint_b = rightendpoint(support(basis))
    FT = promote_type(codomaintype(basis), typeof(endpoint_a), typeof(endpoint_b))
    if !(FT <: AbstractFloat)
        FT = BigFloat
    end
    a = FT(endpoint_a)
    b = FT(endpoint_b)

    isfinite(a) && isfinite(b) ||
        throw(ArgumentError("check_basis_derivs requires a finite support interval."))

    width = b - a
    width > zero(FT) ||
        throw(ArgumentError("basis support must have positive width."))

    # Keep the random samples far enough from the endpoints that the centered
    # stencil stays inside the support interval.
    default_step = min(eps(FT)^(one(FT) / 3) * max(one(FT), width), width / 100)
    h = step_size === nothing ? default_step : FT(step_size)
    h > zero(FT) ||
        throw(ArgumentError("step_size must be positive."))
    2h < width ||
        throw(ArgumentError("step_size must be less than half the support width."))

    rel_tol = tol === nothing ? sqrt(eps(FT)) : FT(tol)
    rel_tol > zero(FT) ||
        throw(ArgumentError("tol must be positive."))

    n = length(basis)
    worst_i = 0
    worst_x = a
    worst_error = zero(FT)
    worst_allowed = rel_tol
    worst_fd = zero(FT)
    worst_deriv = zero(FT)
    worst_ratio = -one(FT)
    all_ok = true

    for i in 1:n
        for _ in 1:num_samples
            x = a + h + (width - 2h) * FT(rand(rng))
            supplied_deriv = maybe_funeval_deriv(basis, i, x, 1)

            if supplied_deriv === nothing
                if verbose
                    @warn("Basis derivative check failed: no supplied first " *
                          "derivative for basis function $i at x = $(_fmt_short(x)).")
                end
                return false
            end

            fd = (funeval(basis, i, x + h) - funeval(basis, i, x - h)) / (2h)
            err = abs(fd - supplied_deriv)
            allowed = rel_tol * max(one(FT), abs(fd), abs(supplied_deriv))

            if !isfinite(err) || err > allowed
                all_ok = false
                ratio = isfinite(err) ? err / allowed : FT(Inf)
                if ratio > worst_ratio
                    worst_i = i
                    worst_x = x
                    worst_error = err
                    worst_allowed = allowed
                    worst_fd = fd
                    worst_deriv = supplied_deriv
                    worst_ratio = ratio
                end
            end
        end
    end

    if all_ok
        verbose && println("  Success: supplied basis derivatives match " *
                           "finite differences for $n basis functions " *
                           "($num_samples samples each).")
        return true
    end

    if verbose
        @warn("Basis derivative check failed for basis function $worst_i at " *
              "x = $(_fmt_short(worst_x)): finite difference = " *
              "$(_fmt_short(worst_fd)), supplied derivative = " *
              "$(_fmt_short(worst_deriv)), error = $(_fmt_sci(worst_error)), " *
              "allowed = $(_fmt_sci(worst_allowed)).")
    end

    false
end


# ────────────────────────────────────────────────────────────────────────────
# Section 8: Collocation-based T-system diagnostic
# ────────────────────────────────────────────────────────────────────────────

struct ESystemCheckResult{T}
    "Tuple size, i.e. number of basis functions tested."
    m::Int
    "Number of sampled ordered tuples."
    num_tuples::Int
    "Support interval."
    a::T
    b::T
    "Reference sign inferred from a reliable sample: +1, -1, or 0 if unavailable."
    reference_sign::Int
    "Smallest normalized determinant after aligning by the reference sign."
    min_signed_normalized_det::T
    "Smallest absolute normalized determinant encountered."
    min_abs_normalized_det::T
    "Largest absolute normalized determinant encountered."
    max_abs_normalized_det::T
    "Threshold below which a normalized determinant is considered near-zero."
    near_zero_threshold::T
    "Sampled tuple that gave the worst signed normalized determinant."
    worst_tuple::Vector{T}
    "True when a sampled tuple produced the opposite sign."
    sign_change_detected::Bool
    "True when a sampled tuple produced a near-zero normalized determinant."
    near_zero_detected::Bool
    "True when a sampled tuple produced a non-finite normalized determinant."
    nonfinite_detected::Bool
    "Overall sampled diagnostic verdict."
    sampled_pass::Bool
end

function Base.show(io::IO, r::ESystemCheckResult)
    status = r.sampled_pass ? "PASS (sampled T-system)" : "FAIL (sampled T-system)"
    print(io, "ESystemCheckResult(m=$(r.m), tuples=$(r.num_tuples), " *
          "[$(_fmt_short(r.a)), $(_fmt_short(r.b))], $status)")
end

function _interior_interval(a::T, b::T) where T
    margin = (b - a) * eps(T)^(one(T) / 3)
    a_inner = a + margin
    b_inner = b - margin
    if a_inner >= b_inner
        a_inner = a + (b - a) / 100
        b_inner = b - (b - a) / 100
    end
    a_inner, b_inner
end

_unit_equispaced_tuple(m::Int, ::Type{T}) where T =
    [T(k) / T(m + 1) for k in 1:m]

function _unit_chebyshev_tuple(m::Int, ::Type{T}) where T
    sort(_chebyshev_points(zero(T), one(T), m, T))
end

function _unit_random_sorted_tuple(rng::AbstractRNG, m::Int, ::Type{T}) where T
    m == 1 && return T[T(rand(rng))]
    sort!(T.(rand(rng, m)))
end

function _unit_clustered_tuple(rng::AbstractRNG, m::Int, ::Type{T}) where T
    m == 1 && return T[T(rand(rng))]

    raw = sort!(T.(rand(rng, m)))
    span = raw[end] - raw[1]
    if span == zero(T)
        return _unit_equispaced_tuple(m, T)
    end

    z = (raw .- raw[1]) ./ span
    width = T(10.0^(-1 - 5 * rand(rng)))
    center = width / 2 + (one(T) - width) * T(rand(rng))
    center .+ width .* (z .- T(0.5))
end

function _affine_tuple(a::T, b::T, z::AbstractVector{T}) where T
    width = b - a
    [a + width * zi for zi in z]
end

function _sample_e_system_tuple(rng::AbstractRNG, a::T, b::T,
                                m::Int, idx::Int, ::Type{T}) where T
    u = _unit_equispaced_tuple(m, T)
    strategy = idx <= 4 ? idx : 5 + mod(idx - 5, 4)

    z = if strategy == 1
        u
    elseif strategy == 2
        _unit_chebyshev_tuple(m, T)
    elseif strategy == 3
        u .^ T(3)
    elseif strategy == 4
        one(T) .- reverse(u .^ T(3))
    elseif strategy == 5
        _unit_random_sorted_tuple(rng, m, T)
    elseif strategy == 6
        p = T(2 + 4 * rand(rng))
        sort!(T.(rand(rng, m)) .^ p)
    elseif strategy == 7
        p = T(2 + 4 * rand(rng))
        sort!(one(T) .- T.(rand(rng, m)) .^ p)
    else
        _unit_clustered_tuple(rng, m, T)
    end

    _affine_tuple(a, b, z)
end

function _divided_difference_collocation_det(basis, xs::AbstractVector{T},
                                             m::Int) where T
    A = Matrix{T}(undef, m, m)
    for j in 1:m
        xj = xs[j]
        for i in 1:m
            A[i, j] = funeval(basis, i, xj)
        end
    end

    # Convert value columns to Newton divided-difference columns in place.
    # The transformation determinant is exactly inv(prod_{i<j}(x_j - x_i)),
    # so det(A) after this loop equals the normalized collocation determinant.
    for order in 2:m
        for j in m:-1:order
            gap = xs[j] - xs[j - order + 1]
            gap > zero(T) ||
                error("Ordered tuple with strictly increasing points required.")
            for i in 1:m
                A[i, j] = (A[i, j] - A[i, j - 1]) / gap
            end
        end
    end

    det(A)
end

function _cluster_guard_bits(xs::AbstractVector{T}, interval_width::T,
                             m::Int) where T
    m <= 1 && return 0

    min_gap = minimum(xs[j] - xs[j - 1] for j in 2:m)
    min_gap > zero(T) ||
        error("Ordered tuple with strictly increasing points required.")

    scale = max(abs(interval_width), one(T))
    rel_gap = min(one(T), min_gap / scale)
    rel_gap > zero(T) || return typemax(Int) ÷ 4

    # Divided differences of order m-1 lose roughly log2(1/h) bits per order
    # when the nodes are tightly clustered.  Add guard bits only for that local
    # calculation instead of forcing users to raise global BigFloat precision.
    ceil(Int, (m - 1) * max(0.0, -Float64(log2(rel_gap))))
end

function _normalized_collocation_det(basis, xs::AbstractVector{T}, m::Int,
                                     interval_width::T) where T
    if T === BigFloat
        base_bits = precision(BigFloat)
        guard_bits = _cluster_guard_bits(xs, interval_width, m)
        extra_bits = max(0, guard_bits - base_bits ÷ 2)
        work_bits = max(base_bits, min(base_bits + extra_bits + 64,
                                      base_bits + 2048))

        if work_bits > base_bits
            return setprecision(BigFloat, work_bits) do
                xs_work = BigFloat.(xs)
                _divided_difference_collocation_det(basis, xs_work, m)
            end
        end
    end

    _divided_difference_collocation_det(basis, xs, m)
end

"""
    check_T_system(basis; num_tuples=5000, tuple_size=length(basis), verbose=true, rng=Random.default_rng(), near_zero_rel_tol=eps(T)^(1/3))

Numerically test the Chebyshev-system (T-system) property by sampling ordered
tuples `x_1 < ⋯ < x_m` and checking whether the normalized collocation
determinant

    det([f_i(x_j)]_{i,j=1}^m) / prod_{i<j} (x_j - x_i)

keeps a fixed nonzero sign.

Internally, the normalized determinant is evaluated as a determinant of Newton
divided differences rather than by forming `det([f_i(x_j)])` and dividing by a
Vandermonde product.  For tightly clustered `BigFloat` tuples the local working
precision is raised automatically for this determinant calculation only.

This is a numerical diagnostic, not a proof.  A detected sign change or
near-zero value is strong evidence against the T-system property.  Passing the
sampled test is evidence in favor of the property, but does not certify it.
"""
function check_T_system(basis;
                        num_tuples::Int = 5000,
                        tuple_size::Int = length(basis),
                        verbose::Bool = true,
                        rng::AbstractRNG = Random.default_rng(),
                        near_zero_rel_tol = nothing)
    n = length(basis)
    1 <= tuple_size <= n || throw(ArgumentError("tuple_size must satisfy 1 <= tuple_size <= length(basis)."))
    num_tuples >= 1 || throw(ArgumentError("num_tuples must be positive."))

    endpoint_a = leftendpoint(support(basis))
    endpoint_b = rightendpoint(support(basis))
    FT = promote_type(codomaintype(basis), typeof(endpoint_a), typeof(endpoint_b))
    if !(FT <: AbstractFloat)
        FT = BigFloat
    end
    a = FT(endpoint_a)
    b = FT(endpoint_b)
    a_inner, b_inner = _interior_interval(a, b)

    rel_tol = near_zero_rel_tol === nothing ?
              eps(FT)^(one(FT) / 3) : FT(near_zero_rel_tol)

    warmup_count = min(num_tuples, 8)
    warmup_tuples = Vector{Vector{FT}}(undef, warmup_count)
    warmup_vals = Vector{FT}(undef, warmup_count)

    for idx in 1:warmup_count
        xs = _sample_e_system_tuple(rng, a_inner, b_inner, tuple_size, idx, FT)
        warmup_tuples[idx] = xs
        warmup_vals[idx] = _normalized_collocation_det(basis, xs, tuple_size,
                                                       b_inner - a_inner)
    end

    ref_idx = 0
    ref_abs = zero(FT)
    for idx in 1:warmup_count
        val = warmup_vals[idx]
        if isfinite(val) && val != zero(FT)
            abs_val = abs(val)
            if abs_val > ref_abs
                ref_abs = abs_val
                ref_idx = idx
            end
        end
    end
    reference_sign = ref_idx == 0 ? 0 : (warmup_vals[ref_idx] > 0 ? 1 : -1)

    min_signed = FT(Inf)
    min_abs = FT(Inf)
    max_abs = zero(FT)
    worst_tuple = Vector{FT}()
    sign_change_detected = false
    nonfinite_detected = false

    function _update_state!(xs, val)
        if !isfinite(val)
            nonfinite_detected = true
            isempty(worst_tuple) && append!(worst_tuple, xs)
            return
        end

        abs_val = abs(val)
        max_abs = max(max_abs, abs_val)
        if abs_val < min_abs
            min_abs = abs_val
            if reference_sign == 0
                empty!(worst_tuple)
                append!(worst_tuple, xs)
            end
        end

        if reference_sign != 0
            signed_val = reference_sign == 1 ? val : -val
            if signed_val < min_signed
                min_signed = signed_val
                empty!(worst_tuple)
                append!(worst_tuple, xs)
            end
            signed_val < zero(FT) && (sign_change_detected = true)
        end
    end

    for idx in 1:warmup_count
        _update_state!(warmup_tuples[idx], warmup_vals[idx])
    end

    for idx in warmup_count+1:num_tuples
        xs = _sample_e_system_tuple(rng, a_inner, b_inner, tuple_size, idx, FT)
        val = _normalized_collocation_det(basis, xs, tuple_size,
                                          b_inner - a_inner)
        _update_state!(xs, val)
    end

    if reference_sign == 0
        min_signed = zero(FT)
        min_abs = isfinite(min_abs) ? min_abs : zero(FT)
    end

    near_zero_threshold = rel_tol * max_abs
    near_zero_detected = reference_sign == 0 ? true : (min_abs <= near_zero_threshold)
    sampled_pass = reference_sign != 0 && !sign_change_detected &&
                   !near_zero_detected && !nonfinite_detected

    result = ESystemCheckResult{FT}(tuple_size, num_tuples, a, b, reference_sign,
                                    min_signed, min_abs, max_abs,
                                    near_zero_threshold, worst_tuple,
                                    sign_change_detected, near_zero_detected,
                                    nonfinite_detected, sampled_pass)

    if verbose
        println()
        println("  Collocation determinant check for T-system property")
        println("  Basis: first $tuple_size functions on [$(_fmt_short(a)), $(_fmt_short(b))]")
        println("  Sampled tuples: $num_tuples")
        println("  Tuple strategies: equispaced, Chebyshev-like, endpoint-heavy, random, clustered")
        println("  Reference sign: " *
                (reference_sign == 1 ? "+" :
                 reference_sign == -1 ? "-" : "unavailable"))
        println("  Smallest signed normalized determinant: $(_fmt_sci(min_signed))")
        println("  Smallest |normalized determinant|:      $(_fmt_sci(min_abs))")
        println("  Largest |normalized determinant|:       $(_fmt_sci(max_abs))")
        println("  Near-zero threshold:                    $(_fmt_sci(near_zero_threshold))")
        println("  Worst tuple:                            $(_fmt_tuple(worst_tuple))")
        println("  Sign change detected:                   $(sign_change_detected ? "yes" : "no")")
        println("  Near-zero detected:                     $(near_zero_detected ? "yes" : "no")")
        println("  Non-finite detected:                    $(nonfinite_detected ? "yes" : "no")")
        println("  Result: " * (sampled_pass ?
                "Sampled tuples support the T-system property." :
                "Sampled tuples found a potential T-system failure witness."))
        println()
    end

    if reference_sign == 0
        @warn("Could not establish a reliable reference sign for the " *
              "normalized collocation determinant on [$(_fmt_short(a)), " *
              "$(_fmt_short(b))].")
    elseif nonfinite_detected
        @warn("A sampled tuple produced a non-finite normalized " *
              "collocation determinant.  Worst tuple: $(_fmt_tuple(worst_tuple)).")
    elseif sign_change_detected
        @warn("The normalized collocation determinant changed sign on the " *
              "sampled tuple $(_fmt_tuple(worst_tuple)).  This is strong " *
              "evidence that the first $tuple_size basis functions do not " *
              "form a T-system.")
    elseif near_zero_detected
        @warn("The normalized collocation determinant became very small on " *
              "the sampled tuple $(_fmt_tuple(worst_tuple)).  This is " *
              "evidence against a robust T-system property, but not a proof.")
    end

    result
end

# ────────────────────────────────────────────────────────────────────────────
# Section 9: Formatting helpers
# ────────────────────────────────────────────────────────────────────────────

_fmt_short(x::BigFloat) = string(Float64(x))
_fmt_short(x) = string(x)

# `m.dde±ee` with two fractional mantissa digits; `Float64`-based (BigFloat is cast).
function _fmt_sci(x::Real)
    !isfinite(x) && return string(x)
    iszero(x) && return "0.00e+00"
    neg = signbit(x) ? "-" : ""
    ax = Float64(abs(x))
    ax == 0 && return "0.00e+00"
    e = floor(Int, log10(ax))
    m = ax / 10.0^e
    while m >= 10
        m /= 10
        e += 1
    end
    while m < 1
        m *= 10
        e -= 1
    end
    mr = round(m; digits=2)
    if mr >= 10
        mr /= 10
        e += 1
    end
    h = round(Int, round(mr; digits=2) * 100)
    intpart, frac = divrem(h, 100)
    mantissa_str = "$(intpart).$(lpad(frac, 2, '0'))"
    esign = e >= 0 ? "+" : "-"
    emag = lpad(string(abs(e)), 2, '0')
    "$(neg)$(mantissa_str)e$(esign)$(emag)"
end

function _fmt_loc(x)
    s = "x = $(_fmt_short(x))"
    length(s) > 16 ? s[1:16] : s
end

function _fmt_tuple(xs::AbstractVector)
    isempty(xs) && return "()"
    parts = map(_fmt_short, xs)
    "(" * join(parts, ", ") * ")"
end
