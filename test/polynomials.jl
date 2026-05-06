using Test
using BasisFunctions
using GeneralizedGauss

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

const GL3_X = [-sqrt(3 / 5), 0.0, sqrt(3 / 5)]
const GL3_W = [5 / 9, 8 / 9, 5 / 9]

const GLL4_X = [-1.0, -1 / sqrt(5), 1 / sqrt(5), 1.0]
const GLL4_W = [1 / 6, 5 / 6, 5 / 6, 1 / 6]

const GRR3_X = [(-1 - sqrt(6)) / 5, (-1 + sqrt(6)) / 5, 1.0]
const GRR3_W = [(16 - sqrt(6)) / 18, (16 + sqrt(6)) / 18, 2 / 9]

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
