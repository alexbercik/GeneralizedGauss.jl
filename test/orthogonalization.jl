using Test
using BasisFunctions
using GeneralizedGauss
using LinearAlgebra

import GeneralizedGauss:
    funeval,
    gauss_legendre,
    maybe_funeval_deriv

isdefined(@__MODULE__, :assert_rule_matches) || include("test_helpers.jl")

# Extended orthogonalization checks.  The fast suite verifies that
# orthogonalization improves a deliberately bad basis; these tests focus on the
# transformation invariants that explain why.

function check_gram_matrix(basis, a, b; measure=nothing, quad_order=120)
    T = typeof(a)
    nodes, weights = gauss_legendre(quad_order, T)
    scale = (b - a) / 2
    shift = (a + b) / 2
    xq = scale .* nodes .+ shift
    wq = scale .* weights
    if measure !== nothing
        wq = wq .* [measure(x) for x in xq]
    end

    n = length(basis)
    G = zeros(T, n, n)
    for i in 1:n, j in 1:n
        G[i, j] = sum(wq[k] * funeval(basis, i, xq[k]) *
                      funeval(basis, j, xq[k]) for k in eachindex(xq))
    end
    G
end

function monomial_higher_derivative_spec(k::Int, max_order::Int)
    [let k = k, m = m
        x -> begin
            m > k && return zero(x)
            coeff = prod(typeof(x)(k - q) for q in 0:m-1; init=one(x))
            coeff * x^(k - m)
        end
    end for m in 1:max_order]
end

@testset "Extended orthogonalization invariants" begin
    setprecision(BigFloat, 192) do
        n = 5
        a = BigFloat(0)
        b = BigFloat(1)
        funs = Function[let k = k
            x -> x^k
        end for k in 0:n-1]
        derivs = Function[let k = k
            x -> k == 0 ? zero(x) : BigFloat(k) * x^(k - 1)
        end for k in 0:n-1]
        basis = quadbasis(funs, derivs, a, b)

        orth_basis, T_mat = orthogonalize_basis(basis)

        for i in 1:n, j in i+1:n
            @test abs(T_mat[i, j]) < BigFloat("1e-40")
        end

        G = check_gram_matrix(orth_basis, a, b; quad_order=100)
        @test norm(G - I(n)) < BigFloat("1e-30")

        old_moments = exact_monomial_moments(n, a, b)
        transformed_moments = T_mat * old_moments
        nodes, weights = gauss_legendre(100, BigFloat)
        direct_moments = [sum((b - a) / 2 * weights[k] *
                              funeval(orth_basis, i,
                                  (b - a) / 2 * nodes[k] + (a + b) / 2)
                              for k in eachindex(nodes))
                          for i in 1:n]
        @test norm(transformed_moments - direct_moments) < BigFloat("1e-30")
    end
end

@testset "Extended orthogonalization derivative handling" begin
    setprecision(BigFloat, 160) do
        n = 4
        a = BigFloat(0)
        b = BigFloat(1)
        funs = Function[let k = k
            x -> x^k
        end for k in 0:n-1]

        basis_no_derivs = quadbasis(funs, nothing, a, b)
        orth_no_derivs, _ = orthogonalize_basis(basis_no_derivs)
        @test orth_no_derivs.fun_derivs === nothing

        deriv_specs = [monomial_higher_derivative_spec(k, n) for k in 0:n-1]
        basis = quadbasis(funs, deriv_specs, a, b)
        orth_basis, T_mat = orthogonalize_basis(basis)
        x0 = BigFloat("0.37")

        for i in 1:n, order in 1:n
            ref = sum(T_mat[i, j] * maybe_funeval_deriv(basis, j, x0, order)
                      for j in 1:i)
            got = BasisFunctions.unsafe_eval_element_derivative(
                orth_basis, i, x0, order)
            @test abs(ref - got) < BigFloat("1e-25")
        end
    end
end

@testset "Extended orthogonalization warning" begin
    n = 9
    funs = Function[let k = k
        x -> x^k
    end for k in 0:n-1]
    derivs = Function[let k = k
        x -> k == 0 ? zero(x) : k * x^(k - 1)
    end for k in 0:n-1]
    basis = quadbasis(funs, derivs, 0.0, 1.0)

    @test_logs (:warn, r"compute_gauss_rule") orthogonalize_basis(basis)
end
