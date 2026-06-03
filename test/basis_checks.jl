using Test
using BasisFunctions
using GeneralizedGauss
using LinearAlgebra
using Random

import GeneralizedGauss: funeval, wronskian,
    maybe_funeval_deriv,
    _prepare_basis_derivs, _derivative_table,
    _TaylorCoeffs, _tvar, _extract_derivs,
    _cheb_approximate, _cheb_differentiate, _cheb_evaluate, _cheb_coefficients,
    _cheb_nodes, _lipschitz_upper_bound

function monomial_derivative_bundle(max_degree::Int, max_order::Int)
    [[let kk = k, mm = m
        x -> begin
            mm > kk && return zero(x)
            coeff = prod(typeof(x)(kk - j) for j in 0:mm-1; init=one(x))
            coeff * x^(kk - mm)
        end
     end for m in 1:max_order] for k in 0:max_degree]
end

bf_tight_tol() = eps(BigFloat)^(BigFloat(2) / 3)
bf_medium_tol() = eps(BigFloat)^(BigFloat(1) / 2)

# ============================================================================
# Tests for the user-supplied first derivative checker
# ============================================================================

@testset "check_basis_derivs: finite-difference derivative diagnostics" begin
    a, b = -1.0, 1.0
    funs = [x -> one(x),
            x -> x,
            x -> x^2,
            x -> sin(x)]
    good_derivs = [x -> zero(x),
                   x -> one(x),
                   x -> 2x,
                   x -> cos(x)]
    good_basis = quadbasis(funs, good_derivs, a, b)

    @test check_basis_derivs === GeneralizedGauss.check_basis_derivs
    @test check_basis_derivs(good_basis; num_samples=8,
                             rng=MersenneTwister(1234), verbose=false)
    @test check_basis_derivs(good_basis; step_size=1e-5, tol=1e-7,
                             num_samples=8, rng=MersenneTwister(1234),
                             verbose=false)

    bad_derivs = [x -> zero(x),
                  x -> one(x),
                  x -> 2x + 0.1,
                  x -> cos(x)]
    bad_basis = quadbasis(funs, bad_derivs, a, b)

    @test !check_basis_derivs(bad_basis; step_size=1e-5, tol=1e-7,
                              num_samples=8, rng=MersenneTwister(1234),
                              verbose=false)

    missing_basis = quadbasis(funs, nothing, a, b)
    @test !check_basis_derivs(missing_basis; step_size=1e-5, tol=1e-7,
                              num_samples=2, rng=MersenneTwister(1234),
                              verbose=false)

    verbose_result = redirect_stdout(devnull) do
        check_basis_derivs(good_basis; step_size=1e-5, tol=1e-7,
                           num_samples=2, rng=MersenneTwister(1234),
                           verbose=true)
    end
    @test verbose_result
end

# ============================================================================
# Tests for Wronskian computation and ECT-system checks
# ============================================================================

@testset verbose=true "Basis checks (Wronskian criterion)" begin

    # -------------------------------------------------------------------
    # 1. Wronskian of monomials: W(1, x, ..., x^m)(x) = prod(j!, j=0:m)
    #    (constant, independent of x — Theorem 3 in existence.pdf)
    # -------------------------------------------------------------------
    @testset "Wronskian of monomials (analytical reference)" begin
        N = 6
        a, b = BigFloat(-1), BigFloat(1)
        setprecision(BigFloat, 64)

        funs = [x -> x^k for k in 0:N-1]
        fun_derivs = monomial_derivative_bundle(N - 1, N)
        basis = quadbasis(funs, fun_derivs, a, b)

        test_pts = [BigFloat("-0.7"), BigFloat("0.0"), BigFloat("0.5")]

        for k in 1:N
            # Exact: W(1, x, ..., x^{k-1}) = prod(j!, j=0:k-1)
            exact_W = prod(BigFloat(factorial(big(j))) for j in 0:k-1)

            for x0 in test_pts
                W = wronskian(basis, k, x0)
                rel_err = abs(W - exact_W) / exact_W
                @test rel_err < bf_medium_tol()
            end
        end
    end

    # -------------------------------------------------------------------
    # 2. Wronskian of monomials should be constant (same at all points)
    # -------------------------------------------------------------------
    @testset "Wronskian of monomials is constant across points" begin
        N = 5
        a, b = BigFloat(0), BigFloat(1)
        setprecision(BigFloat, 64)

        funs = [x -> x^k for k in 0:N-1]
        fun_derivs = monomial_derivative_bundle(N - 1, N)
        basis = quadbasis(funs, fun_derivs, a, b)

        pts = [BigFloat("0.1"), BigFloat("0.3"), BigFloat("0.5"),
               BigFloat("0.7"), BigFloat("0.9")]

        for k in 1:N
            vals = [wronskian(basis, k, x) for x in pts]
            for i in 2:length(vals)
                rel_diff = abs(vals[i] - vals[1]) / abs(vals[1])
                @test rel_diff < bf_medium_tol()
            end
        end
    end

    # -------------------------------------------------------------------
    # 3. Monomials should pass check_ECT_system (known ECT-system)
    # -------------------------------------------------------------------
    @testset "check_ECT_system: monomials on [0, 1]" begin
        N = 5
        a, b = BigFloat(0), BigFloat(1)
        setprecision(BigFloat, 64)

        funs = [x -> x^k for k in 0:N-1]
        fun_derivs = monomial_derivative_bundle(N - 1, N)
        basis = quadbasis(funs, fun_derivs, a, b)

        result = check_ECT_system(basis; n_points=30, verbose=false)

        @test result.is_ect
        @test result.n == N
        @test all(w -> w.certified, result.wronskians)
        for info in result.wronskians
            @test info.sign == 1       # all positive for monomials
            @test info.min_abs > 0
        end
    end

    # -------------------------------------------------------------------
    # 4. Constant but mixed Wronskian signs still satisfy the ECT diagnostic.
    #    The basis {1, -x, x^2} has W_1 > 0, W_2 < 0, W_3 < 0.
    # -------------------------------------------------------------------
    @testset "check_ECT_system: mixed-sign polynomial Wronskians" begin
        N = 3
        a, b = BigFloat(0), BigFloat(1)
        setprecision(BigFloat, 64)

        funs = [x -> one(x),
                x -> -x,
                x -> x^2]
        fun_derivs = [
            [x -> zero(x), x -> zero(x), x -> zero(x)],
            [x -> -one(x), x -> zero(x), x -> zero(x)],
            [x -> 2x, x -> 2one(x), x -> zero(x)]
        ]
        basis = quadbasis(funs, fun_derivs, a, b)

        result = check_ECT_system(basis; n_points=30, verbose=false)

        @test result.is_ect
        @test all(w -> w.certified, result.wronskians)
        @test [w.sign for w in result.wronskians] == [1, -1, -1]
    end

    # -------------------------------------------------------------------
    # 5. Built-in Legendre basis remains sign-definite on the sampled grid.
    #    Certification is conservative here because BasisFunctions only
    #    exposes first derivatives.
    # -------------------------------------------------------------------
    @testset "check_ECT_system: Legendre on [-1, 1]" begin
        setprecision(BigFloat, 64)
        a, b = BigFloat(-1), BigFloat(1)
        basis = Legendre(6) → (a..b)

        result = check_ECT_system(basis; n_points=30, verbose=false)
        @test result.sampled_constant_sign
        @test all(w -> w.sign != 0, result.wronskians)
    end

    # -------------------------------------------------------------------
    # 6. Built-in ChebyshevT basis remains sign-definite on the sampled grid.
    # -------------------------------------------------------------------
    @testset "check_ECT_system: ChebyshevT on [0, 1]" begin
        setprecision(BigFloat, 64)
        a, b = BigFloat(0), BigFloat(1)
        basis = ChebyshevT(6) → (a..b)

        result = check_ECT_system(basis; n_points=100, verbose=false)
        @test result.sampled_constant_sign
        @test all(w -> w.sign != 0, result.wronskians)
    end

    # -------------------------------------------------------------------
    # 7. ChebyshevT on [-1, 1] (native domain)
    # -------------------------------------------------------------------
    @testset "check_ECT_system: ChebyshevT on [-1, 1]" begin
        setprecision(BigFloat, 64)
        a, b = BigFloat(-1), BigFloat(1)
        basis = ChebyshevT(8) → (a..b)

        result = check_ECT_system(basis; n_points=40, verbose=false)
        @test result.sampled_constant_sign
        @test all(w -> w.sign != 0, result.wronskians)
    end

    # -------------------------------------------------------------------
    # 8. Orthogonalized monomials should still be an ECT-system
    #    (orthogonalization with triangular T preserves ordering)
    # -------------------------------------------------------------------
    @testset "check_ECT_system: orthogonalized monomials" begin
        setprecision(BigFloat, 64)
        N = 6
        a, b = BigFloat(0), BigFloat(1)

        funs = [x -> x^k for k in 0:N-1]
        fun_derivs = monomial_derivative_bundle(N - 1, N)
        basis = quadbasis(funs, fun_derivs, a, b)

        orth_basis, _ = orthogonalize_basis(basis)
        result = check_ECT_system(orth_basis; n_points=100, verbose=false)

        @test result.is_ect
    end

    # -------------------------------------------------------------------
    # 9. A deliberately non-Chebyshev basis: {1, sin(πx), sin(2πx)}
    #    on [0, 1].  This is NOT a Chebyshev system because sin(πx)
    #    and sin(2πx) = 2 sin(πx) cos(πx) share a zero structure that
    #    allows nontrivial combinations with too many roots.
    #
    #    W(1, sin(πx), sin(2πx)) should change sign or vanish.
    # -------------------------------------------------------------------
    @testset "check_ECT_system: non-Chebyshev basis (sin functions)" begin
        setprecision(BigFloat, 64)
        a, b = BigFloat(0), BigFloat(1)

        funs = [x -> one(x),
                x -> sin(BigFloat(π) * x),
                x -> sin(2 * BigFloat(π) * x)]
        fun_derivs = [x -> zero(x),
                      x -> BigFloat(π) * cos(BigFloat(π) * x),
                      x -> 2 * BigFloat(π) * cos(2 * BigFloat(π) * x)]
        basis = quadbasis(funs, fun_derivs, a, b)

        result = check_ECT_system(basis; n_points=200, verbose=false)

        @test !result.is_ect
        @test !result.sampled_constant_sign
    end

    # -------------------------------------------------------------------
    # 10. Verbose output smoke test (just ensure it runs without error)
    # -------------------------------------------------------------------
    @testset "check_ECT_system: verbose output runs" begin
        setprecision(BigFloat, 64)
        a, b = BigFloat(-1), BigFloat(1)
        basis = Legendre(4) → (a..b)

        # Redirect stdout to suppress output in test summary
        result = redirect_stdout(devnull) do
            check_ECT_system(basis; n_points=20, verbose=true)
        end

        @test result isa GeneralizedGauss.BasisCheckResult
        @test result.sampled_constant_sign
    end

    # -------------------------------------------------------------------
    # 11. Wronskian of {1, e^x, xe^x, e^{2x}} = 4e^{4x}
    #     (Theorem 3 / Section 3.4: F_2 basis, W = C_d * e^{(d+2)x}
    #      with d=2, C_2 = 4)
    # -------------------------------------------------------------------
    @testset "Wronskian of exponential SBP basis F_2 (analytical)" begin
        setprecision(BigFloat, 64)
        a, b = BigFloat(0), BigFloat(1)

        funs = [x -> one(x),
                x -> x,
                x -> exp(x),
                x -> x * exp(x),
                x -> exp(2x)]
        fun_derivs = [
            [x -> zero(x), x -> zero(x), x -> zero(x), x -> zero(x), x -> zero(x)],
            [x -> one(x), x -> zero(x), x -> zero(x), x -> zero(x), x -> zero(x)],
            [x -> exp(x), x -> exp(x), x -> exp(x), x -> exp(x), x -> exp(x)],
            [let m = m
                x -> (x + BigFloat(m)) * exp(x)
             end for m in 1:5],
            [let m = m
                x -> BigFloat(2)^m * exp(2x)
             end for m in 1:5]
        ]
        basis = quadbasis(funs, fun_derivs, a, b)

        test_pts = [BigFloat("0.1"), BigFloat("0.5"), BigFloat("0.9")]

        for x0 in test_pts
            W = wronskian(basis, 5, x0)
            exact_W = 4 * exp(4 * x0)
            rel_err = abs(W - exact_W) / exact_W
            @test rel_err < bf_medium_tol()
        end

        # At 64-bit BigFloat precision the basis remains sign-definite on the
        # sampled grid, but the conservative certificate may not close.
        result = check_ECT_system(basis; n_points=100, verbose=false)
        @test result.sampled_constant_sign
        @test all(w -> w.sign != 0, result.wronskians)
    end

    # ===================================================================
    # New tests for Chebyshev approximation + Taylor-mode AD
    # ===================================================================

    # -------------------------------------------------------------------
    # 12. Taylor arithmetic: derivatives of exp(x) at a point
    # -------------------------------------------------------------------
    @testset "Taylor arithmetic: exp(x) derivatives" begin
        setprecision(BigFloat, 64)
        x0 = BigFloat("0.7")
        order = 8

        t = _tvar(x0, order)
        result = exp(t)
        derivs = _extract_derivs(result, order)

        # All derivatives of exp(x) are exp(x)
        exact = exp(x0)
        for k in 0:order
            rel_err = abs(derivs[k+1] - exact) / exact
            @test rel_err < bf_tight_tol()
        end
    end

    # -------------------------------------------------------------------
    # 13. Taylor arithmetic: derivatives of sqrt(x) at interior point
    # -------------------------------------------------------------------
    @testset "Taylor arithmetic: sqrt(x) derivatives" begin
        setprecision(BigFloat, 64)
        x0 = BigFloat(1)
        order = 6

        t = _tvar(x0, order)
        result = sqrt(t)
        derivs = _extract_derivs(result, order)

        # f(x) = x^{1/2}: f^{(k)}(1) = prod_{j=0}^{k-1}(1/2 - j)
        for k in 0:order
            exact = prod(BigFloat(1) / 2 - BigFloat(j) for j in 0:k-1;
                         init=one(BigFloat))
            rel_err = exact == 0 ? abs(derivs[k+1]) :
                                   abs(derivs[k+1] - exact) / abs(exact)
            @test rel_err < bf_tight_tol()
        end
    end

    # -------------------------------------------------------------------
    # 13. Taylor arithmetic: derivatives of sin(x) at a point
    # -------------------------------------------------------------------
    @testset "Taylor arithmetic: sin(x) derivatives" begin
        setprecision(BigFloat, 64)
        x0 = BigFloat("0.3")
        order = 8

        t = _tvar(x0, order)
        result = sin(t)
        derivs = _extract_derivs(result, order)

        # Derivatives of sin cycle: sin, cos, -sin, -cos, ...
        exact_cycle = [sin(x0), cos(x0), -sin(x0), -cos(x0)]
        for k in 0:order
            exact = exact_cycle[mod(k, 4) + 1]
            err = abs(derivs[k+1] - exact)
            @test err < bf_tight_tol()
        end
    end

    # -------------------------------------------------------------------
    # 14. Chebyshev approximation converges for smooth functions
    # -------------------------------------------------------------------
    @testset "Chebyshev approximation: exp(x) convergence" begin
        setprecision(BigFloat, 64)
        a, b = BigFloat(-1), BigFloat(1)

        f = x -> exp(x)
        coeffs, conv = _cheb_approximate(f, a, b)

        @test conv
        @test length(coeffs) > 1

        # Evaluate at a few points and check accuracy
        for x in [BigFloat("-0.5"), BigFloat("0.0"), BigFloat("0.8")]
            x_ref = (2x - (a + b)) / (b - a)
            approx_val = _cheb_evaluate(coeffs, x_ref)
            exact_val  = exp(x)
            rel_err = abs(approx_val - exact_val) / abs(exact_val)
            @test rel_err < bf_tight_tol()
        end
    end

    # -------------------------------------------------------------------
    # 15. Chebyshev approximation does NOT converge for sqrt(x) on [0,1]
    # -------------------------------------------------------------------
    @testset "Chebyshev approximation: sqrt(x) non-convergence" begin
        setprecision(BigFloat, 64)
        a, b = BigFloat(0), BigFloat(1)

        f = x -> sqrt(x)
        coeffs, conv = _cheb_approximate(f, a, b; N_max=512)

        @test !conv
    end

    # -------------------------------------------------------------------
    # 16. Lipschitz helper bounds a known smooth function on the interval.
    # -------------------------------------------------------------------
    @testset "Lipschitz upper bound: sin(3πx)" begin
        setprecision(BigFloat, 64)
        a, b = BigFloat(0), BigFloat(1)

        g = x -> sin(3 * BigFloat(π) * x)
        bound, available = _lipschitz_upper_bound(g, a, b)

        pts = [a + (b - a) * BigFloat(i) / BigFloat(1000) for i in 0:1000]
        sampled_max = maximum(abs(g(x)) for x in pts)

        @test available
        @test isfinite(bound)
        @test sampled_max <= bound
    end

    # -------------------------------------------------------------------
    # 17. Chebyshev differentiation reproduces polynomial derivatives
    # -------------------------------------------------------------------
    @testset "Chebyshev differentiation: x^3 on [-1,1]" begin
        setprecision(BigFloat, 64)

        # f(x) = x^3 = (1/4)(3T_1 + T_3) on [-1,1]
        # But compute coefficients numerically:
        N = 8
        nodes = _cheb_nodes(N, BigFloat)
        fvals = [x^3 for x in nodes]
        coeffs = _cheb_coefficients(fvals)

        # First derivative: 3x^2
        dc1 = _cheb_differentiate(coeffs)
        for x in [BigFloat("-0.5"), BigFloat("0.3"), BigFloat("0.9")]
            approx = _cheb_evaluate(dc1, x)
            exact  = 3 * x^2
            @test abs(approx - exact) < bf_tight_tol()
        end

        # Second derivative: 6x
        dc2 = _cheb_differentiate(dc1)
        for x in [BigFloat("-0.5"), BigFloat("0.3"), BigFloat("0.9")]
            approx = _cheb_evaluate(dc2, x)
            exact  = 6 * x
            @test abs(approx - exact) < bf_tight_tol()
        end

        # Third derivative: 6
        dc3 = _cheb_differentiate(dc2)
        for x in [BigFloat("-0.5"), BigFloat("0.3"), BigFloat("0.9")]
            approx = _cheb_evaluate(dc3, x)
            @test abs(approx - 6) < bf_tight_tol()
        end
    end

    # -------------------------------------------------------------------
    # 17. GenericFunctionSet supports higher-order analytic derivatives.
    #     This lets check_ECT_system avoid surrogate derivatives when the user
    #     provides exact formulas.
    # -------------------------------------------------------------------
    @testset "GenericFunctionSet higher-order analytic derivatives" begin
        setprecision(BigFloat, 64)
        a, b = BigFloat(0), BigFloat(1)
        x0 = BigFloat("0.5")

        funs = [x -> one(x),
                x -> x,
                x -> sqrt(x)]
        fun_derivs = [
            [x -> zero(x), x -> zero(x), x -> zero(x)],
            [x -> one(x), x -> zero(x), x -> zero(x)],
            [x -> inv(2 * sqrt(x)),
             x -> -inv(4 * x^(BigFloat(3) / 2)),
             x -> 3 / (8 * x^(BigFloat(5) / 2))]
        ]
        basis = quadbasis(funs, fun_derivs, a, b)

        @test maybe_funeval_deriv(basis, 3, x0, 1) == inv(2 * sqrt(x0))
        @test maybe_funeval_deriv(basis, 3, x0, 2) ==
              -inv(4 * x0^(BigFloat(3) / 2))
        @test maybe_funeval_deriv(basis, 3, x0, 3) ==
              3 / (8 * x0^(BigFloat(5) / 2))
        @test maybe_funeval_deriv(basis, 3, x0, 4) === nothing
    end

    # -------------------------------------------------------------------
    # 18. Missing derivative orders are recovered from the highest
    #     available lower-order analytic derivative, not from the base
    #     function if a better anchor exists.
    # -------------------------------------------------------------------
    @testset "Derivative recovery uses highest exact lower order" begin
        setprecision(BigFloat, 64)
        a, b = BigFloat(0), BigFloat(1)
        x0 = BigFloat("0.3")

        funs = [x -> x^4]
        fun_derivs = [[x -> 4 * x^3,
                       nothing,
                       x -> 24 * x,
                       x -> 24 * one(x)]]
        basis = quadbasis(funs, fun_derivs, a, b)

        info = _prepare_basis_derivs(basis, 1, a, b, 4; emit_warnings=false)[1]

        @test collect(info.exact_available) == [true, true, false, true, true]
        @test info.missing_base_order[3] == 1
        @test info.surrogates[1] === nothing
        @test info.surrogates[2] !== nothing
        @test info.surrogates[2].base_order == 1

        scale = BigFloat(2) / (b - a)
        D = _derivative_table(basis, x0, 1, 4, [info], a, b, scale)
        exact = [x0^4, 4 * x0^3, 12 * x0^2, 24 * x0, 24 * one(x0)]
        for m in 1:5
            @test abs(D[1, m] - exact[m]) < bf_tight_tol()
        end
    end

    # -------------------------------------------------------------------
    # 19. check_ECT_system should keep using available analytic derivatives even
    #     when higher orders are missing, and approximate only the missing
    #     orders from the highest exact lower derivative.
    # -------------------------------------------------------------------
    @testset "check_ECT_system: partial analytic derivatives" begin
        setprecision(BigFloat, 64)
        a, b = BigFloat(0), BigFloat(1)

        funs = [x -> one(x),
                x -> x,
                x -> exp(x)]
        fun_derivs = [
            [x -> zero(x), x -> zero(x), x -> zero(x)],
            [x -> one(x), x -> zero(x), x -> zero(x)],
            x -> exp(x)
        ]
        basis = quadbasis(funs, fun_derivs, a, b)

        info = _prepare_basis_derivs(basis, 3, a, b, 3; emit_warnings=false)
        @test collect(info[3].exact_available) == [true, true, false, false]
        @test info[3].missing_base_order[3] == 1
        @test info[3].missing_base_order[4] == 1
        @test info[3].surrogates[2] !== nothing

        result = redirect_stderr(devnull) do
            check_ECT_system(basis; n_points=40, verbose=false)
        end

        @test result.is_ect
        @test result.wronskians[3].sign == 1
    end

    # -------------------------------------------------------------------
    # 20. Basis with endpoint singularity: {1, x, sqrt(x)} on [0, 1]
    #     This IS an ECT-system on (0, 1).  The Wronskian is:
    #       W_3(x) = det [1   x   x^{1/2}  ]
    #                    [0   1   1/(2√x)   ]
    #                    [0   0   -1/(4x√x) ]
    #     = -1/(4 x^{3/2}), which is negative and nonzero on (0,1).
    #
    #     The higher-order analytic derivatives let the certificate stay exact.
    # -------------------------------------------------------------------
    @testset "check_ECT_system: sqrt(x) basis with analytic higher derivatives" begin
        setprecision(BigFloat, 64)
        a, b = BigFloat(0), BigFloat(1)

        funs = [x -> one(x),
                x -> x,
                x -> sqrt(x)]
        fun_derivs = [
            [x -> zero(x), x -> zero(x), x -> zero(x)],
            [x -> one(x), x -> zero(x), x -> zero(x)],
            [x -> inv(2 * sqrt(x)),
             x -> -inv(4 * x^(BigFloat(3) / 2)),
             x -> 3 / (8 * x^(BigFloat(5) / 2))]
        ]
        basis = quadbasis(funs, fun_derivs, a, b)

        result = check_ECT_system(basis; n_points=100, verbose=false)

        @test result.sampled_constant_sign
        # W_3 should be negative (constant sign -1)
        @test result.wronskians[3].sign == -1
    end

    # -------------------------------------------------------------------
    # 21. Wronskian of {1, x, sqrt(x)} via standalone function
    #     W_3(x) = -1/(4 x^{3/2}) at interior points
    # -------------------------------------------------------------------
    @testset "Wronskian of sqrt(x) basis (analytical)" begin
        setprecision(BigFloat, 64)
        a, b = BigFloat(0), BigFloat(1)

        funs = [x -> one(x),
                x -> x,
                x -> sqrt(x)]
        fun_derivs = [x -> zero(x),
                      x -> one(x),
                      x -> inv(2 * sqrt(x))]
        basis = quadbasis(funs, fun_derivs, a, b)

        test_pts = [BigFloat("0.1"), BigFloat("0.25"), BigFloat("0.5"),
                    BigFloat("0.9")]

        for x0 in test_pts
            W = wronskian(basis, 3, x0)
            exact_W = -one(BigFloat) / (4 * x0^(BigFloat(3)/2))
            rel_err = abs(W - exact_W) / abs(exact_W)
            @test rel_err < bf_medium_tol()
        end
    end

    # -------------------------------------------------------------------
    # 22. Oscillatory counterexample: the sample signs alone are misleading,
    #     but the refined minimum and W' bound reject the basis.
    # -------------------------------------------------------------------
    @testset "check_ECT_system: oscillatory false positive regression" begin
        setprecision(BigFloat, 64)
        a, b = BigFloat(0), BigFloat(1)
        m = 500
        omega = 2 * BigFloat(π) * BigFloat(m)
        A = BigFloat("2.005")
        phi = BigFloat("3.5430183815484892")
        eps_osc = A / omega^2

        funs = [x -> one(x),
                x -> x,
                x -> x^2 + eps_osc * sin(omega * x + phi)]
        fun_derivs = [x -> zero(x),
                      x -> one(x),
                      x -> 2 * x + eps_osc * omega * cos(omega * x + phi)]
        basis = quadbasis(funs, fun_derivs, a, b)

        result = redirect_stderr(devnull) do
            check_ECT_system(basis; n_points=200, verbose=false)
        end

        @test result.sampled_constant_sign
        @test !result.is_ect
        @test result.wronskians[3].sign == 1
        @test !result.wronskians[3].certified
        @test result.wronskians[3].lipschitz_lower_bound <= 0
    end

    # -------------------------------------------------------------------
    # 23. wronskian should fail loudly when neither analytic derivatives,
    #     Taylor arithmetic, nor Chebyshev approximation can supply them.
    # -------------------------------------------------------------------
    @testset "wronskian: derivative failure throws" begin
        setprecision(BigFloat, 64)
        a, b = BigFloat(0), BigFloat(1)

        funs = [x -> one(x),
                x -> floor(10 * x)]
        basis = quadbasis(funs, nothing, a, b)

        @test_throws ErrorException wronskian(basis, 2, BigFloat("0.35"))
    end

    # -------------------------------------------------------------------
    # 24. Taylor arithmetic: compositions (x^2 * exp(x))
    # -------------------------------------------------------------------
    @testset "Taylor arithmetic: x^2 * exp(x) derivatives" begin
        setprecision(BigFloat, 64)
        x0 = BigFloat("0.5")
        order = 6

        t = _tvar(x0, order)
        result = t^2 * exp(t)
        derivs = _extract_derivs(result, order)

        # f(x) = x^2 e^x: f^(k) = e^x (x^2 + 2kx + k(k-1))
        for k in 0:order
            exact = exp(x0) * (x0^2 + 2 * BigFloat(k) * x0 +
                               BigFloat(k) * BigFloat(k - 1))
            rel_err = abs(derivs[k+1] - exact) / abs(exact)
            @test rel_err < bf_tight_tol()
        end
    end

end

# -------------------------------------------------------------------
# 25–26. Collocation-based T-system diagnostics (full determinant sampling)
# -------------------------------------------------------------------
@testset "check_T_system: collocation diagnostics" begin

    # 25. Collocation-based T-system diagnostic: monomials should pass.
    @testset "monomials on [0, 1]" begin
        setprecision(BigFloat, 64)
        a, b = BigFloat(0), BigFloat(1)
        N = 4

        funs = [x -> x^k for k in 0:N-1]
        fun_derivs = monomial_derivative_bundle(N - 1, N)
        basis = quadbasis(funs, fun_derivs, a, b)

        result = redirect_stderr(devnull) do
            check_T_system(basis; num_tuples=400, rng=MersenneTwister(1234),
                           verbose=false)
        end

        @test result.sampled_pass
        @test result.reference_sign == 1
        @test !result.sign_change_detected
        @test !result.near_zero_detected
    end

    # 26. tuple_size should test only the corresponding initial subset.
    @testset "tuple_size initial subset" begin
        setprecision(BigFloat, 64)
        a, b = BigFloat(0), BigFloat(1)

        funs = [x -> one(x),
                x -> x,
                x -> sin(2 * BigFloat(π) * x)]
        fun_derivs = [x -> zero(x),
                      x -> one(x),
                      x -> 2 * BigFloat(π) * cos(2 * BigFloat(π) * x)]
        basis = quadbasis(funs, fun_derivs, a, b)

        subset_result = redirect_stderr(devnull) do
            check_T_system(basis; tuple_size=2, num_tuples=300,
                           rng=MersenneTwister(1234), verbose=false)
        end
        full_result = redirect_stderr(devnull) do
            check_T_system(basis; tuple_size=3, num_tuples=300,
                           rng=MersenneTwister(1234), verbose=false)
        end

        @test subset_result.sampled_pass
        @test !full_result.sampled_pass
        @test full_result.sign_change_detected || full_result.near_zero_detected
    end

    # 27. Non-Chebyshev basis (linear dependence) should fail.
    @testset "linearly dependent basis fails" begin
        setprecision(BigFloat, 64)
        a, b = BigFloat(0), BigFloat(1)

        # {1, x, x} is linearly dependent, so the collocation determinant
        # must vanish and the sampled T-system check should fail.
        funs = [x -> one(x),
                x -> x,
                x -> 2*x - one(x)]
        fun_derivs = [x -> zero(x),
                      x -> one(x),
                      x -> 2*one(x)]
        basis = quadbasis(funs, fun_derivs, a, b)

        result = redirect_stderr(devnull) do
            check_T_system(basis; num_tuples=300, rng=MersenneTwister(1234),
                           verbose=false)
        end

        @test !result.sampled_pass
        @test result.sign_change_detected || result.near_zero_detected
    end

    # 28. Clustered tuples should not blow up the normalized determinant for a
    #     well-conditioned exponential-polynomial T-system.
    @testset "clustered exponential-polynomial basis is stable" begin
        setprecision(BigFloat, 80)
        a, b = big"0", big"1"
        n = 4

        cheb = ChebyshevT(n + 1) → (a..b)
        poly_funs = [x -> BasisFunctions.unsafe_eval_element(cheb, j, x)
                     for j in 1:n+1]
        poly_derivs = [x -> BasisFunctions.unsafe_eval_element_derivative(cheb, j, x, 1)
                       for j in 1:n+1]

        exp_funs = [x -> BasisFunctions.unsafe_eval_element(cheb, j, x) * exp(x)
                    for j in 1:n]
        exp_derivs = [x -> begin
                          T_val = BasisFunctions.unsafe_eval_element(cheb, j, x)
                          T_der = BasisFunctions.unsafe_eval_element_derivative(cheb, j, x, 1)
                          (T_der + T_val) * exp(x)
                      end
                      for j in 1:n]

        funs = vcat(poly_funs, exp_funs, x -> exp(2x))
        fun_derivs = vcat(poly_derivs, exp_derivs, x -> 2exp(2x))
        basis = quadbasis(funs, fun_derivs, a, b)

        result = redirect_stderr(devnull) do
            check_T_system(basis; num_tuples=200, rng=MersenneTwister(1234),
                           verbose=false)
        end

        @test result.sampled_pass
        @test !result.sign_change_detected
        @test !result.near_zero_detected
        @test result.max_abs_normalized_det < big"1e-3"
    end
end
