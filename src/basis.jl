
"""
A generic set of functions.

This type represents a `Dictionary` as in the BasisFunctions package. The basis
functions are given by a vector of functions, and so are the derivatives.
The basis is defined on an interval `[a,b]`.
"""
struct GenericFunctionSet{S,T,F,D} <: Dictionary{S,T}
    funs        ::  F
    fun_derivs  ::  D
    a           ::  S
    b           ::  S
    orthogonalization_digits_lost :: Float64

    function GenericFunctionSet{S,T,F,D}(funs::F, fun_derivs::D, a, b,
            orthogonalization_digits_lost::Real=0.0) where {S,T,F,D}
        if fun_derivs != nothing
            @assert length(funs) == length(fun_derivs)
        end
        new(funs, fun_derivs, a, b, Float64(orthogonalization_digits_lost))
    end
end

GenericFunctionSet(funs, fun_derivs, a::S, b::T) where {S,T} =
    GenericFunctionSet(funs, fun_derivs, promote(a,b)...)
GenericFunctionSet(funs, fun_derivs, a::S, b::T,
        orthogonalization_digits_lost::Real) where {S,T} =
    GenericFunctionSet(funs, fun_derivs, promote(a,b)...,
        orthogonalization_digits_lost)
GenericFunctionSet(funs, fun_derivs, a::T, b::T) where {T} =
    GenericFunctionSet{T,T}(funs, fun_derivs, a, b)
GenericFunctionSet(funs, fun_derivs, a::T, b::T,
        orthogonalization_digits_lost::Real) where {T} =
    GenericFunctionSet{T,T}(funs, fun_derivs, a, b,
        orthogonalization_digits_lost)

GenericFunctionSet{S,T}(funs::F, fun_derivs::D, a, b) where {S,T,F,D} =
    GenericFunctionSet{S,T,F,D}(funs, fun_derivs, a, b)
GenericFunctionSet{S,T}(funs::F, fun_derivs::D, a, b,
        orthogonalization_digits_lost::Real) where {S,T,F,D} =
    GenericFunctionSet{S,T,F,D}(funs, fun_derivs, a, b,
        orthogonalization_digits_lost)

Base.size(dict::GenericFunctionSet) = (length(dict.funs),)
BasisFunctions.support(dict::GenericFunctionSet) = dict.a..dict.b

BasisFunctions.unsafe_eval_element(basis::GenericFunctionSet, i, x) = basis.funs[i](x)

_has_derivative_order(::Nothing, order::Int) = false
_has_derivative_order(::Function, order::Int) = order == 1
function _has_derivative_order(spec::Union{AbstractVector,Tuple}, order::Int)
    1 <= order <= length(spec) || return false
    spec[order] !== nothing
end

_eval_derivative_spec(::Nothing, x, order::Int) = nothing
_eval_derivative_spec(spec::Function, x, order::Int) =
    order == 1 ? spec(x) : nothing
function _eval_derivative_spec(spec::Union{AbstractVector,Tuple}, x, order::Int)
    _has_derivative_order(spec, order) || return nothing
    spec[order](x)
end

function maybe_funeval_deriv(basis::GenericFunctionSet, i, x, order::Int)
    basis.fun_derivs === nothing && return nothing
    _eval_derivative_spec(basis.fun_derivs[i], x, order)
end

function BasisFunctions.unsafe_eval_element_derivative(basis::GenericFunctionSet, i, x, order)
    deriv = maybe_funeval_deriv(basis, i, x, order)
    deriv === nothing &&
        throw(ArgumentError("No analytic derivative of order $order for basis function $i"))
    deriv
end


quadbasis(funs, fun_derivs, a, b) = GenericFunctionSet(funs, fun_derivs, a, b)

# Custom getindex: return a new GenericFunctionSet (not a BasisFunctions SubDict).
# This ensures that unsafe_eval_element and unsafe_eval_element_derivative dispatch
# correctly on the sliced dictionary, which is critical for user-defined basis
# functions (e.g. Chebyshev closures) whose derivatives rely on the
# GenericFunctionSet method specializations.
Base.getindex(dict::GenericFunctionSet, I::AbstractUnitRange) =
    GenericFunctionSet(dict.funs[I],
                       dict.fun_derivs === nothing ? nothing : dict.fun_derivs[I],
                       dict.a, dict.b,
                       dict.orthogonalization_digits_lost)


funeval(basis, i, x) = basis[i](x)
funeval(basis::Dictionary, i, x) =
    BasisFunctions.unsafe_eval_element(basis, i, x)
funeval_deriv(basis::Dictionary, i, x) =
    BasisFunctions.unsafe_eval_element_derivative(basis, i, x, 1)

function maybe_funeval_deriv(basis::Dictionary, i, x, order::Int)
    try
        BasisFunctions.unsafe_eval_element_derivative(basis, i, x, order)
    catch e
        if e isa MethodError || e isa ArgumentError ||
           e isa AssertionError || e isa ErrorException
            return nothing
        end
        rethrow(e)
    end
end

"""
Approximate a missing first derivative with a support-aware finite difference.

Use a centered stencil in the interior. Continuation can release a zero-weight
support endpoint, so use a second-order one-sided stencil when the boundary
makes a centered stencil impossible.
"""
function finite_diff_funeval_deriv(basis::Dictionary, i, x)
    h = cbrt(eps(one(x))) * max(one(x), abs(x))
    a = leftendpoint(support(basis))
    b = rightendpoint(support(basis))

    if (isfinite(a) && x < a) || (isfinite(b) && x > b)
        throw(ArgumentError("Cannot approximate a derivative outside the basis support at x=$(x)."))
    end

    if isfinite(a) && x - a < h
        h = isfinite(b) ? min(h, (b - x) / 2) : h
        h > zero(h) ||
            throw(ArgumentError("Cannot form a forward derivative stencil at x=$(x)."))
        return (-3funeval(basis, i, x) + 4funeval(basis, i, x + h) -
                funeval(basis, i, x + 2h)) / (2h)
    end
    if isfinite(b) && b - x < h
        h = isfinite(a) ? min(h, (x - a) / 2) : h
        h > zero(h) ||
            throw(ArgumentError("Cannot form a backward derivative stencil at x=$(x)."))
        return (3funeval(basis, i, x) - 4funeval(basis, i, x - h) +
                funeval(basis, i, x - 2h)) / (2h)
    end

    (funeval(basis, i, x + h) - funeval(basis, i, x - h)) / (2h)
end

"""
Evaluate an analytic first derivative when available and otherwise use a
support-aware finite difference. This keeps exact basis derivatives in mixed
bases.
"""
function funeval_deriv_or_finite_diff(basis::Dictionary, i, x)
    deriv = maybe_funeval_deriv(basis, i, x, 1)
    deriv === nothing ? finite_diff_funeval_deriv(basis, i, x) : deriv
end
