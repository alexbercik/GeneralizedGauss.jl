using Test
using BasisFunctions
using GeneralizedGauss
using LinearAlgebra

import GeneralizedGauss: gauss_legendre

isdefined(@__MODULE__, :assert_rule_matches) || include("test_helpers.jl")

# Extended BigFloat checks.  The fast suite owns the 30-digit default
# requirement; this opt-in suite checks a few higher-precision pipelines without
# overriding solver_tolerance or printing long diagnostic tables.

function reference_gl_mapped(n::Int, a::T, b::T) where {T}
    x_ref, w_ref = gauss_legendre(n, T)
    scale = (b - a) / 2
    shift = (a + b) / 2
    scale .* x_ref .+ shift, scale .* w_ref
end

@testset "Extended BigFloat precision" begin
    setprecision(BigFloat, 256) do
        tol = BigFloat("1e-45")

        @testset "5-point GL from orthogonalized monomials on [0, 1]" begin
            n_rule = 5
            n_basis = 2 * n_rule
            a = BigFloat(0)
            b = BigFloat(1)
            basis, moments = polynomial_basis_and_moments(n_basis, BigFloat;
                a, b)

            orth_basis, T_mat = orthogonalize_basis(basis)
            w, x = compute_gauss_rule(orth_basis, T_mat * moments)
            x_ref, w_ref = reference_gl_mapped(n_rule, a, b)

            assert_rule_matches(w, x, w_ref, x_ref; atol=tol, rtol=tol)
            @test digits_of_agreement(x, x_ref) >= 45
            @test digits_of_agreement(w, w_ref) >= 45
        end

        @testset "3-point BigFloat finite-difference path preserves precision" begin
            basis, moments = polynomial_basis_and_moments(6, BigFloat;
                derivatives=false)
            w, x = redirect_stderr(devnull) do
                compute_gauss_rule(basis, moments; principal=:lower)
            end
            ref_w, ref_x = reference_gl3(BigFloat)

            @test eltype(w) === BigFloat
            @test eltype(x) === BigFloat
            assert_rule_matches(w, x, ref_w, ref_x;
                atol=BigFloat("1e-35"), rtol=BigFloat("1e-35"))
        end

        @testset "BigFloat non-polynomial residual exactness" begin
            basis, moments = exp_poly_basis_and_moments(BigFloat)
            w, x = compute_gauss_rule(basis, moments;
                principal=:lower, add_endpoint=:left)

            @test eltype(w) === BigFloat
            @test eltype(x) === BigFloat
            @test basis_residual_norm(basis, moments, w, x) < BigFloat("1e-40")
        end
    end
end
