using LinearAlgebra
using Random

# Shared helpers for the fast regression tests.  Keep reference data here so
# individual test files describe behavior instead of repeating formulas.

function polynomial_basis_data(n::Int, ::Type{T}=Float64; a=T(-1), b=T(1)) where {T}
    aa = T(a)
    bb = T(b)
    funs = Function[]
    derivs = Function[]

    for k in 0:n-1
        push!(funs, let k = k
            x -> x^k
        end)
        push!(derivs, let k = k
            x -> k == 0 ? zero(x) : T(k) * x^(k - 1)
        end)
    end

    funs, derivs, aa, bb
end

function polynomial_basis(n::Int, ::Type{T}=Float64;
        a=T(-1), b=T(1), derivatives::Bool=true) where {T}
    funs, derivs, aa, bb = polynomial_basis_data(n, T; a, b)
    quadbasis(funs, derivatives ? derivs : nothing, aa, bb)
end

function exact_monomial_moments(n::Int, a, b)
    T = promote_type(typeof(a), typeof(b))
    aa = T(a)
    bb = T(b)
    [((bb^(k + 1) - aa^(k + 1)) / T(k + 1)) for k in 0:n-1]
end

function polynomial_basis_and_moments(n::Int, ::Type{T}=Float64;
        a=T(-1), b=T(1), derivatives::Bool=true) where {T}
    aa = T(a)
    bb = T(b)
    polynomial_basis(n, T; a=aa, b=bb, derivatives),
        exact_monomial_moments(n, aa, bb)
end

function reference_gl3(::Type{T}=Float64) where {T}
    s = sqrt(T(3) / T(5))
    T[T(5) / T(9), T(8) / T(9), T(5) / T(9)],
        T[-s, zero(T), s]
end

function reference_lgl4(::Type{T}=Float64) where {T}
    s = inv(sqrt(T(5)))
    T[T(1) / T(6), T(5) / T(6), T(5) / T(6), T(1) / T(6)],
        T[-one(T), -s, s, one(T)]
end

function reference_right_radau3(::Type{T}=Float64) where {T}
    s = sqrt(T(6))
    T[(T(16) - s) / T(18), (T(16) + s) / T(18), T(2) / T(9)],
        T[(-one(T) - s) / T(5), (-one(T) + s) / T(5), one(T)]
end

function reference_left_radau3(::Type{T}=Float64) where {T}
    s = sqrt(T(6))
    T[T(2) / T(9), (T(16) + s) / T(18), (T(16) - s) / T(18)],
        T[-one(T), (one(T) - s) / T(5), (one(T) + s) / T(5)]
end

function sorted_rule(w, x)
    p = sortperm(x)
    w[p], x[p]
end

function assert_rule_matches(w, x, w_ref, x_ref; atol, rtol=zero(atol))
    w_s, x_s = sorted_rule(w, x)
    w_ref_s, x_ref_s = sorted_rule(w_ref, x_ref)

    @test length(w_s) == length(w_ref_s)
    @test length(x_s) == length(x_ref_s)
    @test isapprox(x_s, x_ref_s; atol, rtol)
    @test isapprox(w_s, w_ref_s; atol, rtol)
end

function basis_residual_norm(basis, moments, w, x)
    T = promote_type(eltype(moments), eltype(w), eltype(x))
    residual_max = zero(T)

    for j in 1:length(basis)
        residual = -moments[j]
        for i in eachindex(w)
            residual += w[i] * BasisFunctions.unsafe_eval_element(basis, j, x[i])
        end
        residual_max = max(residual_max, abs(residual))
    end

    residual_max
end

function digits_of_agreement(v, v_ref)
    err = norm(v - v_ref, Inf)
    err == 0 ? Inf : -Float64(log10(big(err)))
end

function monomial_derivative_bundle(max_degree::Int, max_order::Int)
    bundles = Vector{Vector{Function}}()

    for k in 0:max_degree
        derivs = Function[]
        for m in 1:max_order
            push!(derivs, let k = k, m = m
                x -> begin
                    m > k && return zero(x)
                    coeff = prod(typeof(x)(k - q) for q in 0:m-1; init=one(x))
                    coeff * x^(k - m)
                end
            end)
        end
        push!(bundles, derivs)
    end

    bundles
end

function exp_poly_basis_and_moments(::Type{T}=Float64) where {T}
    a = zero(T)
    b = one(T)

    # F_2 = {1, x, exp(x), x exp(x), exp(2x)} is a smooth
    # exponential-polynomial Chebyshev example used for non-polynomial coverage.
    funs = Function[
        x -> one(x),
        x -> x,
        x -> exp(x),
        x -> x * exp(x),
        x -> exp(T(2) * x),
    ]
    derivs = Function[
        x -> zero(x),
        x -> one(x),
        x -> exp(x),
        x -> (one(x) + x) * exp(x),
        x -> T(2) * exp(T(2) * x),
    ]
    moments = T[
        one(T),
        T(1) / T(2),
        exp(one(T)) - one(T),
        one(T),
        (exp(T(2)) - one(T)) / T(2),
    ]

    quadbasis(funs, derivs, a, b), moments
end
