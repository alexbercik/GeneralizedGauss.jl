using Test
using BasisFunctions
using GeneralizedGauss

isdefined(@__MODULE__, :assert_rule_matches) || include("test_helpers.jl")

@testset "Public quadrature API contracts" begin
    @testset "legendre_basis reproduces Gauss-Legendre" begin
        basis = legendre_basis(6, -1.0, 1.0)
        moments = [2.0; zeros(5)]
        w, x = compute_gauss_rule(basis, moments; principal=:lower)
        ref_w, ref_x = reference_gl3(Float64)

        assert_rule_matches(w, x, ref_w, ref_x; atol=1e-12, rtol=1e-12)
    end

    @testset "automatic and explicit moments agree" begin
        basis = ChebyshevT(6)
        mu = measure(basis)
        moments = compute_moments(basis; measure=mu)

        explicit_w, explicit_x = compute_gauss_rule(basis, moments)
        automatic_w, automatic_x = compute_gauss_rule(basis; measure=mu)

        assert_rule_matches(automatic_w, automatic_x, explicit_w, explicit_x;
            atol=1e-12, rtol=1e-12)
    end

    @testset "one-point rule" begin
        basis = quadbasis(
            Function[x -> one(x), x -> x],
            Function[x -> zero(x), x -> one(x)],
            -1.0, 1.0)
        w, x = compute_gauss_rule(basis, [2.0, 0.0])

        assert_rule_matches(w, x, [2.0], [0.0]; atol=1e-14, rtol=1e-14)
    end

    @testset "Lobatto checkpoint contains the Radau seed" begin
        basis, moments = polynomial_basis_and_moments(6, Float64)

        for add_endpoint in (:right, :left)
            w, x, xi_checkpoints, w_checkpoints, x_checkpoints =
                compute_gauss_rules(basis, moments;
                    principal=:upper, add_endpoint)

            lgl_w, lgl_x = reference_lgl4(Float64)
            assert_rule_matches(w, x, lgl_w, lgl_x;
                atol=1e-11, rtol=1e-11)
            @test length(xi_checkpoints) == length(w_checkpoints)
            @test length(xi_checkpoints) == length(x_checkpoints)

            radau_w, radau_x = add_endpoint == :right ?
                reference_right_radau3(Float64) :
                reference_left_radau3(Float64)
            assert_rule_matches(w_checkpoints[end - 1], x_checkpoints[end - 1],
                radau_w, radau_x; atol=1e-11, rtol=1e-11)
        end
    end

    @testset "odd basis rejects crossed endpoint pairings" begin
        basis, moments = polynomial_basis_and_moments(5, Float64)
        @test_throws ErrorException compute_gauss_rule(
            basis, moments; principal=:upper, add_endpoint=:left)
        @test_throws ErrorException compute_gauss_rule(
            basis, moments; principal=:lower, add_endpoint=:right)
    end
end
