using Test
using BasisFunctions
using GeneralizedGauss
import GeneralizedGauss:
    legendre_polynomial,
    legendre_polynomial_derivative,
    legendre_values!,
    legendre_block,
    eval_legendre_block!

function monomial_basis(n)
    funs = [x -> x^k for k in 0:n-1]
    fun_derivs = vcat(x -> zero(x), [x -> k * x^(k - 1) for k in 1:n-1])
    quadbasis(funs, fun_derivs, -1.0, 1.0)
end

function monomial_moments(n)
    [iseven(k) ? 2 / (k + 1) : 0.0 for k in 0:n-1]
end

function assert_rule_matches(w, x, w_ref, x_ref; atol=1e-10, rtol=1e-10)
    @test length(w) == length(w_ref)
    @test length(x) == length(x_ref)
    @test x ≈ x_ref atol=atol rtol=rtol
    @test w ≈ w_ref atol=atol rtol=rtol
end

@testset "Legendre polynomial helpers" begin
    x0 = 0.2

    @test legendre_polynomial(0, x0) == 1.0
    @test legendre_polynomial(1, x0) == x0
    @test legendre_polynomial(2, x0) ≈ (3x0^2 - 1) / 2
    @test legendre_polynomial(3, x0) ≈ (5x0^3 - 3x0) / 2

    @test legendre_polynomial_derivative(0, x0) == 0.0
    @test legendre_polynomial_derivative(1, x0) == 1.0
    @test legendre_polynomial_derivative(2, x0) ≈ 3x0
    @test legendre_polynomial_derivative(3, x0) ≈ (15x0^2 - 3) / 2

    funs, fun_derivs = legendre_functions(4)
    @test [f(x0) for f in funs] ≈ [
        1.0,
        x0,
        (3x0^2 - 1) / 2,
        (5x0^3 - 3x0) / 2,
    ]
    @test [df(x0) for df in fun_derivs] ≈ [
        0.0,
        1.0,
        3x0,
        (15x0^2 - 3) / 2,
    ]

    a = BigFloat(0)
    b = BigFloat(1)
    x_bf = BigFloat("0.25")
    t = 2x_bf - 1
    mapped_funs, mapped_derivs = legendre_functions(3, a, b)
    @test mapped_funs[3](x_bf) == (3t^2 - 1) / 2
    @test mapped_derivs[3](x_bf) == 6t

    ref = (a, b)
    ref_funs, ref_derivs = legendre_functions(3, ref)
    @test ref_funs[3](x_bf) == mapped_funs[3](x_bf)
    @test ref_derivs[3](x_bf) == mapped_derivs[3](x_bf)

    basis = legendre_basis(3, ref)
    @test length(basis) == 3
    @test BasisFunctions.unsafe_eval_element(basis, 3, x_bf) == (3t^2 - 1) / 2
    @test BasisFunctions.unsafe_eval_element_derivative(basis, 3, x_bf, 1) == 6t

    default_basis = legendre_basis(3)
    @test default_basis.a === -1.0
    @test default_basis.b === 1.0
    @test BasisFunctions.unsafe_eval_element(default_basis, 3, x0) ≈ (3x0^2 - 1) / 2

    rational_funs, rational_derivs = legendre_functions(3, -1//1, 1//1)
    @test rational_funs[3](x0) isa Float64
    @test rational_funs[3](x0) ≈ (3x0^2 - 1) / 2
    @test rational_derivs[3](x0) ≈ 3x0

    P = Vector{Float64}(undef, 4)
    dP = similar(P)
    legendre_values!(P, dP, 4, x0)
    @test P ≈ [legendre_polynomial(k, x0) for k in 0:3]
    @test dP ≈ [legendre_polynomial_derivative(k, x0) for k in 0:3]

    blk = legendre_block(-1.0, 1.0, 4)
    for k in 0:3
        @test funs[k + 1](x0) == eval_legendre_block!(blk, x0).P[k + 1]
        @test fun_derivs[k + 1](x0) == eval_legendre_block!(blk, x0).dP[k + 1]
    end

    # Mixed-basis style: one Legendre block, other callables appended separately.
    L_funs, L_derivs = legendre_functions(3, -1.0, 1.0)
    mixed_funs = vcat(L_funs, [x -> exp(x)])
    mixed_derivs = vcat(L_derivs, [x -> exp(x)])
    @test mixed_funs[3](x0) ≈ (3x0^2 - 1) / 2
    @test mixed_derivs[3](x0) ≈ 3x0
    @test mixed_funs[4](x0) ≈ exp(x0)
end

const GL3_X = [-sqrt(3 / 5), 0.0, sqrt(3 / 5)]
const GL3_W = [5 / 9, 8 / 9, 5 / 9]

const GLL4_X = [-1.0, -1 / sqrt(5), 1 / sqrt(5), 1.0]
const GLL4_W = [1 / 6, 5 / 6, 5 / 6, 1 / 6]

const GRR3_X = [(-1 - sqrt(6)) / 5, (-1 + sqrt(6)) / 5, 1.0]
const GRR3_W = [(16 - sqrt(6)) / 18, (16 + sqrt(6)) / 18, 2 / 9]

@testset "One-point scalar solve" begin
    funs = Function[x -> one(x), x -> x]
    fun_derivs = Function[x -> zero(x), x -> one(x)]
    moments = [2.0, 0.0]

    basis_newton = quadbasis(funs, fun_derivs, -1.0, 1.0)
    w_newton, x_newton = compute_gauss_rule(basis_newton, moments)
    assert_rule_matches(w_newton, x_newton, [2.0], [0.0])

    basis_no_deriv = quadbasis(funs, nothing, -1.0, 1.0)
    w_brent, x_brent = compute_gauss_rule(basis_no_deriv, moments; solver=:brent)
    assert_rule_matches(w_brent, x_brent, [2.0], [0.0])

    shifted_moments = [2.0, 1.0]
    redirect_stdout(devnull) do
        @test_throws ErrorException compute_gauss_rule(
            basis_newton, shifted_moments; maxiter=0, principal_lost_digits=0)
    end
    w_lost, x_lost = compute_gauss_rule(
        basis_newton, shifted_moments; maxiter=0, principal_lost_digits=400)
    @test isfinite(first(x_lost))
    @test isfinite(first(w_lost))
    @test abs(2 * first(x_lost) - 1) > 1e-4

    w_fallback, x_fallback =
        @test_logs (:warn, r"No analytic first derivatives") compute_gauss_rule(
            basis_no_deriv, moments)
    assert_rule_matches(w_fallback, x_fallback, [2.0], [0.0])

    higher_funs = Function[x -> one(x), x -> x, x -> x^2]
    higher_basis_no_deriv = quadbasis(higher_funs, nothing, -1.0, 1.0)
    @test_logs (:warn, r"No analytic first derivatives") begin
        @test_throws ErrorException compute_gauss_rule(
            higher_basis_no_deriv, [2.0, 0.0, 2 / 3])
    end
end

@testset "Known polynomial quadratures" begin
    @testset "3-point Gauss-Legendre from polynomial bases" begin
        bases = (
            ("ChebyshevT(6)", ChebyshevT(6), nothing),
            ("Legendre(6)", Legendre(6), nothing),
            ("manual monomials", monomial_basis(6), monomial_moments(6)),
        )

        for (label, basis, moments) in bases
            w, x = isnothing(moments) ?
                compute_gauss_rule(basis) :
                compute_gauss_rule(basis, moments)

            @testset "$label" begin
                assert_rule_matches(w, x, GL3_W, GL3_X)
            end
        end
    end

    @testset "4-point Gauss-Lobatto-Legendre from polynomial bases" begin
        bases = (
            ("ChebyshevT(6)", ChebyshevT(6), nothing),
            ("Legendre(6)", Legendre(6), nothing),
            ("manual monomials", monomial_basis(6), monomial_moments(6)),
        )

        for (label, basis, moments) in bases
            w, x = isnothing(moments) ?
                compute_gauss_rule(basis; principal=:upper) :
                compute_gauss_rule(basis, moments; principal=:upper)

            @testset "$label" begin
                assert_rule_matches(w, x, GLL4_W, GLL4_X)
            end
        end
    end

    @testset "3-point right Gauss-Radau-Legendre from polynomial bases" begin
        bases = (
            ("ChebyshevT(5)", ChebyshevT(5), nothing),
            ("Legendre(5)", Legendre(5), nothing),
            ("manual monomials", monomial_basis(5), monomial_moments(5)),
        )

        for (label, basis, moments) in bases
            w, x = isnothing(moments) ?
                compute_gauss_rule(basis; principal=:upper) :
                compute_gauss_rule(basis, moments; principal=:upper)

            @testset "$label" begin
                assert_rule_matches(w, x, GRR3_W, GRR3_X)
            end
        end
    end

    @testset "Continuation API returns the known terminal Gauss rule" begin
        w, x, xi_checkpoints, w_checkpoints, x_checkpoints =
            compute_gauss_rules(ChebyshevT(6))

        assert_rule_matches(w, x, GL3_W, GL3_X)
        @test !isempty(xi_checkpoints)
        @test length(xi_checkpoints) == length(w_checkpoints)
        @test length(xi_checkpoints) == length(x_checkpoints)
        @test w_checkpoints[end] ≈ GL3_W
        @test x_checkpoints[end] ≈ GL3_X
    end
end
