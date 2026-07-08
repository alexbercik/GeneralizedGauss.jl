@testset "T-system and ECT diagnostics" begin
    @testset "ECT accepts monomials" begin
        setprecision(BigFloat, 96) do
            n = 4
            a = BigFloat(0)
            b = BigFloat(1)
            funs = Function[let k = k
                x -> x^k
            end for k in 0:n-1]
            derivs = monomial_derivative_bundle(n - 1, n)
            basis = quadbasis(funs, derivs, a, b)

            result = check_ECT_system(basis; n_points=24, verbose=false)
            @test result.is_ect
            @test result.sampled_constant_sign
            @test all(info -> info.sign == 1, result.wronskians)
        end
    end

    @testset "ECT accepts a smooth exponential-polynomial system" begin
        setprecision(BigFloat, 96) do
            a = BigFloat(0)
            b = BigFloat(1)
            funs = Function[
                x -> one(x),
                x -> x,
                x -> exp(x),
            ]
            derivs = [
                [x -> zero(x), x -> zero(x), x -> zero(x)],
                [x -> one(x), x -> zero(x), x -> zero(x)],
                [x -> exp(x), x -> exp(x), x -> exp(x)],
            ]
            basis = quadbasis(funs, derivs, a, b)

            result = check_ECT_system(basis; n_points=24, verbose=false)
            @test result.is_ect
            @test all(info -> info.sign == 1, result.wronskians)
        end
    end

    @testset "ECT rejects a sinusoidal counterexample" begin
        setprecision(BigFloat, 96) do
            a = BigFloat(0)
            b = BigFloat(1)
            omega = BigFloat(π)
            funs = Function[
                x -> one(x),
                x -> sin(omega * x),
                x -> sin(2 * omega * x),
            ]
            derivs = Function[
                x -> zero(x),
                x -> omega * cos(omega * x),
                x -> 2 * omega * cos(2 * omega * x),
            ]
            basis = quadbasis(funs, derivs, a, b)

            result = redirect_stderr(devnull) do
                check_ECT_system(basis; n_points=80, verbose=false)
            end
            @test !result.is_ect
            @test !result.sampled_constant_sign
        end
    end

    @testset "T-system collocation accepts and rejects known cases" begin
        setprecision(BigFloat, 96) do
            a = BigFloat(0)
            b = BigFloat(1)

            pass_funs = Function[let k = k
                x -> x^k
            end for k in 0:3]
            pass_basis = quadbasis(pass_funs, monomial_derivative_bundle(3, 4),
                a, b)
            pass_result = redirect_stderr(devnull) do
                check_T_system(pass_basis; num_tuples=120,
                    rng=MersenneTwister(1234), verbose=false)
            end
            @test pass_result.sampled_pass
            @test !pass_result.sign_change_detected
            @test !pass_result.near_zero_detected

            fail_funs = Function[
                x -> one(x),
                x -> x,
                x -> 2 * x - one(x),
            ]
            fail_derivs = Function[
                x -> zero(x),
                x -> one(x),
                x -> 2 * one(x),
            ]
            fail_basis = quadbasis(fail_funs, fail_derivs, a, b)
            fail_result = redirect_stderr(devnull) do
                check_T_system(fail_basis; num_tuples=120,
                    rng=MersenneTwister(1234), verbose=false)
            end
            @test !fail_result.sampled_pass
            @test fail_result.sign_change_detected ||
                  fail_result.near_zero_detected
        end
    end

    @testset "supplied derivative verification" begin
        funs = Function[x -> one(x), x -> x, x -> x^2]
        correct_derivs =
            Function[x -> zero(x), x -> one(x), x -> 2x]
        incorrect_derivs =
            Function[x -> zero(x), x -> one(x), x -> 3x]

        correct_basis = quadbasis(funs, correct_derivs, 0.0, 1.0)
        incorrect_basis = quadbasis(funs, incorrect_derivs, 0.0, 1.0)

        @test check_basis_derivs(correct_basis;
            num_samples=8, verbose=false, rng=MersenneTwister(1234))
        @test !check_basis_derivs(incorrect_basis;
            num_samples=8, verbose=false, rng=MersenneTwister(1234))
    end
end
