using Test
using BasisFunctions
using GeneralizedGauss
using LinearAlgebra
using Random

import GeneralizedGauss:
    wronskian,
    maybe_funeval_deriv,
    _prepare_basis_derivs,
    _derivative_table

isdefined(@__MODULE__, :assert_rule_matches) || include("test_helpers.jl")

# Extended diagnostics for the T-system / ECT-system machinery.  The default
# fast diagnostics cover public pass/fail behavior; this file keeps focused
# internals that are useful for regression hunting without the old long sweeps.

@testset "Extended Wronskian references" begin
    setprecision(BigFloat, 96) do
        n = 5
        a = BigFloat(0)
        b = BigFloat(1)
        funs = Function[let k = k
            x -> x^k
        end for k in 0:n-1]
        derivs = monomial_derivative_bundle(n - 1, n)
        basis = quadbasis(funs, derivs, a, b)

        for k in 1:n, x0 in (BigFloat("0.2"), BigFloat("0.7"))
            exact = prod(BigFloat(factorial(big(j))) for j in 0:k-1)
            @test abs(wronskian(basis, k, x0) - exact) / exact <
                  BigFloat("1e-40")
        end

        mixed_basis = quadbasis(
            Function[x -> one(x), x -> -x, x -> x^2],
            [
                [x -> zero(x), x -> zero(x), x -> zero(x)],
                [x -> -one(x), x -> zero(x), x -> zero(x)],
                [x -> 2x, x -> 2one(x), x -> zero(x)],
            ],
            a, b)
        result = check_ECT_system(mixed_basis; n_points=24, verbose=false)
        @test result.is_ect
        @test [info.sign for info in result.wronskians] == [1, -1, -1]
    end
end

@testset "Extended derivative recovery diagnostics" begin
    setprecision(BigFloat, 96) do
        a = BigFloat(0)
        b = BigFloat(1)
        x0 = BigFloat("0.3")

        basis = quadbasis(
            Function[x -> x^4],
            [[x -> 4x^3, nothing, x -> 24x, x -> 24one(x)]],
            a, b)
        info = _prepare_basis_derivs(basis, 1, a, b, 4;
            emit_warnings=false)[1]

        @test collect(info.exact_available) == [true, true, false, true, true]
        @test info.missing_base_order[3] == 1
        @test info.surrogates[2] !== nothing

        scale = BigFloat(2) / (b - a)
        D = _derivative_table(basis, x0, 1, 4, [info], a, b, scale)
        exact = [x0^4, 4x0^3, 12x0^2, 24x0, 24one(x0)]
        @test maximum(abs.(D[1, :] .- exact)) < BigFloat("1e-25")

        sqrt_basis = quadbasis(
            Function[x -> one(x), x -> x, x -> sqrt(x)],
            Function[x -> zero(x), x -> one(x), x -> inv(2sqrt(x))],
            a, b)
        W = wronskian(sqrt_basis, 3, BigFloat("0.25"))
        @test abs(W + inv(4 * BigFloat("0.25")^(BigFloat(3) / 2))) <
              BigFloat("1e-25")
        @test maybe_funeval_deriv(sqrt_basis, 3, x0, 2) === nothing
    end
end

@testset "Extended ECT certification regression" begin
    setprecision(BigFloat, 96) do
        a = BigFloat(0)
        b = BigFloat(1)
        m = 300
        omega = 2 * BigFloat(π) * BigFloat(m)
        amplitude = BigFloat("2.005")
        phase = BigFloat("3.5430183815484892")
        eps_osc = amplitude / omega^2

        # Depending on the sample grid, this can fail either by an observed sign
        # change or by an uncertified small Wronskian.  The regression is that
        # it must not be certified as an ECT-system.
        basis = quadbasis(
            Function[
                x -> one(x),
                x -> x,
                x -> x^2 + eps_osc * sin(omega * x + phase),
            ],
            Function[
                x -> zero(x),
                x -> one(x),
                x -> 2x + eps_osc * omega * cos(omega * x + phase),
            ],
            a, b)

        result = redirect_stderr(devnull) do
            check_ECT_system(basis; n_points=120, verbose=false)
        end

        @test !result.is_ect
        @test !result.wronskians[3].certified
    end
end

@testset "Extended clustered T-system collocation" begin
    setprecision(BigFloat, 96) do
        a = BigFloat(0)
        b = BigFloat(1)
        n = 4
        cheb = ChebyshevT(n + 1) → (a..b)

        poly_funs = Function[
            let j = j
                x -> BasisFunctions.unsafe_eval_element(cheb, j, x)
            end for j in 1:n+1
        ]
        poly_derivs = Function[
            let j = j
                x -> BasisFunctions.unsafe_eval_element_derivative(cheb, j, x, 1)
            end for j in 1:n+1
        ]
        exp_funs = Function[
            let j = j
                x -> BasisFunctions.unsafe_eval_element(cheb, j, x) * exp(x)
            end for j in 1:n
        ]
        exp_derivs = Function[
            let j = j
                x -> begin
                    val = BasisFunctions.unsafe_eval_element(cheb, j, x)
                    der = BasisFunctions.unsafe_eval_element_derivative(cheb, j, x, 1)
                    (der + val) * exp(x)
                end
            end for j in 1:n
        ]

        basis = quadbasis(vcat(poly_funs, exp_funs, x -> exp(2x)),
            vcat(poly_derivs, exp_derivs, x -> 2exp(2x)), a, b)
        result = redirect_stderr(devnull) do
            check_T_system(basis; num_tuples=120,
                rng=MersenneTwister(1234), verbose=false)
        end

        @test result.sampled_pass
        @test !result.sign_change_detected
        @test !result.near_zero_detected
    end
end
