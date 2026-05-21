
"""
    legendre_polynomial(n, x)

Evaluate the Legendre polynomial ``P_n(x)`` on ``[-1, 1]`` by the three-term recurrence.
"""
function legendre_polynomial(n::Integer, x)
    n >= 0 || throw(ArgumentError("Legendre degree must be non-negative"))
    first(_legendre_value_derivative(n, x))
end

"""
    legendre_polynomial_derivative(n, x)

Evaluate ``P'_n(x)`` on ``[-1, 1]`` by differentiating the three-term
Legendre recurrence. This is well-defined at the endpoints `x = +/-1`.
"""
function legendre_polynomial_derivative(n::Integer, x)
    n >= 0 || throw(ArgumentError("Legendre degree must be non-negative"))
    last(_legendre_value_derivative(n, x))
end

function _legendre_value_derivative(n::Integer, x)
    if n == 0
        return one(x), zero(x)
    end

    P_prev, P_curr = one(x), x
    dP_prev, dP_curr = zero(x), one(x)

    for k in 2:n
        P_next = ((2k - 1) * x * P_curr - (k - 1) * P_prev) / k
        dP_next = ((2k - 1) * (P_curr + x * dP_curr) -
                   (k - 1) * dP_prev) / k
        P_prev, P_curr = P_curr, P_next
        dP_prev, dP_curr = dP_curr, dP_next
    end

    P_curr, dP_curr
end

function _legendre_values_only!(P::AbstractVector, n::Integer, t)
    n == 0 && return P

    @inbounds begin
        P[1] = one(t)
        n == 1 && return P

        P[2] = t
        n == 2 && return P

        P_prev, P_curr = P[1], P[2]
        for k in 2:(n - 1)
            P_next = ((2k - 1) * t * P_curr - (k - 1) * P_prev) / k
            P[k + 1] = P_next
            P_prev, P_curr = P_curr, P_next
        end
    end
    P
end

function _legendre_derivatives_from_values!(dP::AbstractVector, P::AbstractVector,
                                            n::Integer, t)
    n == 0 && return dP

    @inbounds begin
        dP[1] = zero(t)
        n == 1 && return dP

        dP[2] = one(t)
        n == 2 && return dP

        dP_prev, dP_curr = dP[1], dP[2]
        for k in 2:(n - 1)
            dP_next = ((2k - 1) * (P[k] + t * dP_curr) - (k - 1) * dP_prev) / k
            dP[k + 1] = dP_next
            dP_prev, dP_curr = dP_curr, dP_next
        end
    end
    dP
end

"""
    legendre_values!(P, dP, n, t)

Fill `P[k] = P_{k-1}(t)` and `dP[k] = P'_{k-1}(t)` for ``k = 1,…,n`` on the
reference interval ``[-1, 1]`` in ``O(n)`` work.

Both buffers must have length at least `n`. Returns `(P, dP)`.
"""
function legendre_values!(P::AbstractVector, dP::AbstractVector, n::Integer, t)
    n >= 0 || throw(ArgumentError("number of Legendre values must be non-negative"))
    length(P) >= n || throw(ArgumentError("P must have length at least n = $n"))
    length(dP) >= n || throw(ArgumentError("dP must have length at least n = $n"))
    n == 0 && return P, dP

    _legendre_values_only!(P, n, t)
    _legendre_derivatives_from_values!(dP, P, n, t)
    P, dP
end

function _reference_coordinate(x, a, b)
    (2 * x - (a + b)) / (b - a)
end

function _check_legendre_interval(a, b)
    a < b || throw(ArgumentError("Legendre interval endpoints must satisfy a < b"))
    nothing
end

# Integer and rational endpoints such as the defaults `a = -1`, `b = 1` still
# evaluate at floating `x`; use Float64 buffers unless the endpoints specify an
# explicit floating type.
function _legendre_eltype(a, b)
    T = promote_type(typeof(a), typeof(b))
    T <: AbstractFloat ? T : Float64
end

const LegendreIntervalRef = Union{
    Tuple{<:Number, <:Number},
    Pair{<:Number, <:Number},
}

_legendre_interval_endpoints(ref::LegendreIntervalRef) = (first(ref), last(ref))

function _same_legendre_point(x, cached)
    cached === nothing && return false
    x === cached && return true
    x == cached
end

"""
    LegendreFunctionBlock{T}

Shared evaluation state for the first `n` Legendre polynomials mapped to
`[a, b]`.  All callables returned by [`legendre_functions`](@ref) that share
this block batch their work: repeated evaluation at the same `x` reuses
cached values.
"""
mutable struct LegendreFunctionBlock{T}
    a::T
    b::T
    n::Int
    x_cache::Union{T, Nothing}
    dx_cache::Union{T, Nothing}
    P::Vector{T}
    dP::Vector{T}
    deriv_scale::T
end

"""
    legendre_block(a, b, n)

Allocate a [`LegendreFunctionBlock`](@ref) for `n` mapped Legendre functions on
`[a, b]`.  Floating endpoint types are preserved (e.g. `BigFloat` endpoints give
`BigFloat` buffers); integer and rational endpoints use `Float64` buffers.
"""
function legendre_block(a, b, n::Integer)
    T = _legendre_eltype(a, b)
    a = T(a)
    b = T(b)
    n >= 0 || throw(ArgumentError("number of Legendre functions must be non-negative"))
    _check_legendre_interval(a, b)
    LegendreFunctionBlock{T}(a, b, Int(n), nothing, nothing,
                             zeros(T, n), zeros(T, n), T(2) / (b - a))
end

function _eval_legendre_values!(block::LegendreFunctionBlock{T}, x) where {T}
    xT = T(x)
    _same_legendre_point(xT, block.x_cache) && return block

    t = _reference_coordinate(xT, block.a, block.b)
    _legendre_values_only!(block.P, block.n, t)
    block.x_cache = xT
    block.dx_cache = nothing
    block
end

function _eval_legendre_derivatives!(block::LegendreFunctionBlock{T}, x) where {T}
    xT = T(x)
    _same_legendre_point(xT, block.dx_cache) && return block

    _eval_legendre_values!(block, xT)
    t = _reference_coordinate(xT, block.a, block.b)
    _legendre_derivatives_from_values!(block.dP, block.P, block.n, t)
    ds = block.deriv_scale
    @inbounds for k in 1:block.n
        block.dP[k] *= ds
    end
    block.dx_cache = xT
    block
end

"""
    eval_legendre_block!(block, x)

Evaluate all `n` mapped Legendre functions and their ``x``-derivatives at `x`,
reusing cached values when `x` matches the previous point.  Returns `block`;
read `block.P[k]` and `block.dP[k]` for degrees ``k-1,…,n-1``.
"""
function eval_legendre_block!(block::LegendreFunctionBlock{T}, x) where {T}
    _eval_legendre_derivatives!(block, x)
end

function _legendre_callables(block::LegendreFunctionBlock{T}) where {T}
    n = block.n
    funs = Function[
        (let k = k
            x -> _eval_legendre_values!(block, x).P[k + 1]
        end) for k in 0:(n - 1)
    ]
    fun_derivs = Function[
        (let k = k
            x -> _eval_legendre_derivatives!(block, x).dP[k + 1]
        end) for k in 0:(n - 1)
    ]
    funs, fun_derivs
end

function _legendre_functions_impl(n::Integer, a, b)
    n >= 0 || throw(ArgumentError("number of Legendre functions must be non-negative"))
    block = legendre_block(a, b, n)
    _legendre_callables(block)
end

function _legendre_basis_impl(n::Integer, a, b)
    block = legendre_block(a, b, n)
    funs, fun_derivs = _legendre_callables(block)
    quadbasis(funs, fun_derivs, block.a, block.b)
end

"""
    legendre_functions(n, a=-1, b=1)
    legendre_functions(n, ref)

Return `(funs, fun_derivs)` for the first `n` Legendre polynomials mapped from
`[-1, 1]` to `[a, b]`.  The returned vectors are ready to pass to
`quadbasis(funs, fun_derivs, a, b)` or to `vcat` with other callables in a
mixed basis.

# Arguments
- `n`: number of basis functions (degrees `0,…,n-1`).
- `a`, `b`: interval endpoints.
- `ref`: interval as a tuple or pair, e.g. `(BigFloat(-1), BigFloat(1))` or
  `BigFloat(0) => BigFloat(1)`; arithmetic type follows the endpoints.

Evaluations at a fixed `x` share one ``O(n)`` recurrence internally.
"""
legendre_functions(n::Integer, a=-1, b=1) = _legendre_functions_impl(n, a, b)
legendre_functions(n::Integer, ref::LegendreIntervalRef) =
    _legendre_functions_impl(n, _legendre_interval_endpoints(ref)...)

"""
    legendre_basis(n, a=-1, b=1)
    legendre_basis(n, ref)

Build a [`GenericFunctionSet`](@ref) from the first `n` mapped Legendre
polynomials and their first derivatives.  See [`legendre_functions`](@ref) for
the meaning of `ref`.
"""
legendre_basis(n::Integer, a=-1, b=1) = _legendre_basis_impl(n, a, b)
legendre_basis(n::Integer, ref::LegendreIntervalRef) =
    _legendre_basis_impl(n, _legendre_interval_endpoints(ref)...)

"""
    gauss_legendre(n, ::Type{T}=BigFloat) where T

Compute `n`-point Gauss-Legendre quadrature nodes and weights on `[-1, 1]`
in arithmetic type `T`. Uses Newton iteration on the three-term Legendre
recurrence, which works natively with `BigFloat` or any `AbstractFloat`.
"""
function gauss_legendre(n::Int, ::Type{T}=BigFloat) where T
    nodes   = zeros(T, n)
    weights = zeros(T, n)
    m = (n + 1) ÷ 2          # number of non-negative roots by symmetry

    for i in 1:m
        # Tricomi initial guess.
        x = cos(T(pi) * (4i - 1) / (4n + 2))

        # Newton iterations to find the root of P_n.
        for _ in 1:200
            P_curr, dP = _legendre_value_derivative(n, x)
            delta = P_curr / dP
            x -= delta
            abs(delta) <= 10 * eps(T) && break
        end

        _, dP = _legendre_value_derivative(n, x)
        w = T(2) / ((1 - x^2) * dP^2)

        nodes[n + 1 - i] =  x;   weights[n + 1 - i] = w
        nodes[i]          = -x;   weights[i]          = w
    end

    nodes, weights
end
