using Test
using BasisFunctions
using GeneralizedGauss

import GeneralizedGauss:
    legendre_polynomial,
    legendre_polynomial_derivative,
    legendre_values!,
    legendre_block,
    eval_legendre_block!

isdefined(@__MODULE__, :assert_rule_matches) || include("test_helpers.jl")

# Extended polynomial regressions.  The fast suite owns the main LG/LGL/Radau
# reference coverage; this file keeps targeted helper and continuation checks
# that are useful but not needed on every default test run.

@testset "Extended polynomial helper regressions" begin
    x0 = 0.2

    @test legendre_polynomial(0, x0) == 1.0
    @test legendre_polynomial(1, x0) == x0
    @test legendre_polynomial(2, x0) ≈ (3x0^2 - 1) / 2
    @test legendre_polynomial_derivative(0, x0) == 0.0
    @test legendre_polynomial_derivative(3, x0) ≈ (15x0^2 - 3) / 2

    P = Vector{Float64}(undef, 4)
    dP = similar(P)
    legendre_values!(P, dP, 4, x0)
    @test P ≈ [legendre_polynomial(k, x0) for k in 0:3]
    @test dP ≈ [legendre_polynomial_derivative(k, x0) for k in 0:3]

    block = legendre_block(-1.0, 1.0, 4)
    funs, derivs = legendre_functions(4)
    for k in 0:3
        @test funs[k + 1](x0) == eval_legendre_block!(block, x0).P[k + 1]
        @test derivs[k + 1](x0) == eval_legendre_block!(block, x0).dP[k + 1]
    end
end

@testset "Extended scalar and tolerance regressions" begin
    scalar_basis = quadbasis(
        Function[x -> one(x), x -> x],
        Function[x -> zero(x), x -> one(x)],
        -1.0, 1.0)

    w, x = compute_gauss_rule(scalar_basis, [2.0, 0.0])
    assert_rule_matches(w, x, [2.0], [0.0]; atol=1e-14, rtol=1e-14)

    for bad_tolerance in (0.0, -1e-3, Inf, NaN)
        @test_throws ArgumentError compute_gauss_rule(
            scalar_basis, [2.0, 0.0]; intermediate_tolerance=bad_tolerance)
    end

    # Relaxed intermediate tolerances may loosen checkpoints, but the returned
    # terminal 3-point LG rule should still satisfy the strict residual check.
    basis6, moments6 = polynomial_basis_and_moments(6, Float64)
    w_lg, x_lg = compute_gauss_rule(basis6, moments6;
        principal=:lower, intermediate_tolerance=1e-6)
    @test basis_residual_norm(basis6, moments6, w_lg, x_lg) <=
          GeneralizedGauss.solver_tolerance(Float64)

    w_lgl, x_lgl = compute_gauss_rule(basis6, moments6;
        principal=:upper, intermediate_tolerance=1e-6)
    @test basis_residual_norm(basis6, moments6, w_lgl, x_lgl) <=
          GeneralizedGauss.solver_tolerance(Float64)
end

@testset "Extended continuation API regressions" begin
    basis6, moments6 = polynomial_basis_and_moments(6, Float64)
    lgl4_w, lgl4_x = reference_lgl4(Float64)
    grr3_w, grr3_x = reference_right_radau3(Float64)
    grl3_w, grl3_x = reference_left_radau3(Float64)

    for add_endpoint in (:right, :left)
        w, x, xi_checkpoints, w_checkpoints, x_checkpoints =
            compute_gauss_rules(basis6, moments6;
                principal=:upper, add_endpoint)

        assert_rule_matches(w, x, lgl4_w, lgl4_x; atol=1e-11, rtol=1e-11)
        @test length(xi_checkpoints) == length(w_checkpoints)
        @test length(xi_checkpoints) == length(x_checkpoints)

        # The checkpoint before Lobatto is the natural Radau seed.  This catches
        # regressions where the continuation accidentally inserts a Gauss solve.
        radau_w = add_endpoint == :right ? grr3_w : grl3_w
        radau_x = add_endpoint == :right ? grr3_x : grl3_x
        assert_rule_matches(w_checkpoints[end - 1], x_checkpoints[end - 1],
            radau_w, radau_x; atol=1e-11, rtol=1e-11)
    end

    basis5, moments5 = polynomial_basis_and_moments(5, Float64)
    @test_throws ErrorException compute_gauss_rule(
        basis5, moments5; principal=:upper, add_endpoint=:left)
    @test_throws ErrorException compute_gauss_rule(
        basis5, moments5; principal=:lower, add_endpoint=:right)
end
