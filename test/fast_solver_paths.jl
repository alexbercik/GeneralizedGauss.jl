@testset "Solver path comparison on 3-point Gauss-Legendre" begin
    funs, derivs, a, b = polynomial_basis_data(6, Float64)
    moments = exact_monomial_moments(6, a, b)
    ref_w, ref_x = reference_gl3(Float64)

    analytic_basis = quadbasis(funs, derivs, a, b)
    no_deriv_basis = quadbasis(funs, nothing, a, b)

    @test GeneralizedGauss._resolve_gauss_solver_mode(
        analytic_basis, true) == :analytic_nlsolve
    @test_logs (:warn, r"Analytic first derivatives are missing") begin
        @test GeneralizedGauss._resolve_gauss_solver_mode(
            no_deriv_basis, true) == :finite_diff_nlsolve
    end
    @test GeneralizedGauss._resolve_gauss_solver_mode(
        no_deriv_basis, false) == :mads

    # Warm each path before timing.  Otherwise the first derivative-free
    # differentiable call reports Julia specialization and logging setup cost
    # rather than the finite-difference nonlinear solve time.
    compute_gauss_rule(analytic_basis, moments; principal=:lower)
    @test_logs (:warn, r"Analytic first derivatives are missing") compute_gauss_rule(
        no_deriv_basis, moments; principal=:lower)
    redirect_stderr(devnull) do
        compute_gauss_rule(no_deriv_basis, moments; principal=:lower)
    end
    compute_gauss_rule(no_deriv_basis, moments;
        principal=:lower, differentiable=false)

    analytic_time, analytic_result = best_elapsed_result(samples=5) do
        compute_gauss_rule(analytic_basis, moments; principal=:lower)
    end
    analytic_w, analytic_x = analytic_result

    finite_diff_time, finite_diff_result = best_elapsed_result(samples=5) do
        redirect_stderr(devnull) do
            compute_gauss_rule(no_deriv_basis, moments; principal=:lower)
        end
    end
    finite_diff_w, finite_diff_x = finite_diff_result

    mads_time, mads_result = best_elapsed_result(samples=2) do
        compute_gauss_rule(
            no_deriv_basis, moments; principal=:lower, differentiable=false)
    end
    mads_w, mads_x = mads_result

    # The timing comparison is intentionally diagnostic.  Use best-of warmed
    # samples to avoid reporting one-off specialization/cache effects.
    print_runtime_comparison("3-point LG", (;
        analytic=analytic_time,
        finite_diff=finite_diff_time,
        mads=mads_time))

    for (w, x) in ((analytic_w, analytic_x),
                   (finite_diff_w, finite_diff_x),
                   (mads_w, mads_x))
        assert_rule_matches(w, x, ref_w, ref_x; atol=1e-6, rtol=1e-6)
    end

    assert_rule_matches(finite_diff_w, finite_diff_x, analytic_w, analytic_x;
        atol=1e-7, rtol=1e-7)
    assert_rule_matches(mads_w, mads_x, analytic_w, analytic_x;
        atol=1e-6, rtol=1e-6)
end

@testset "MADS adapter rejects invalid node trials before evaluation" begin
    moments = exact_monomial_moments(4, -1.0, 1.0)
    evaluation_count = Ref(0)
    funs = Function[
        x -> (evaluation_count[] += 1; one(x)),
        x -> (evaluation_count[] += 1; x),
        x -> (evaluation_count[] += 1; x^2),
        x -> (evaluation_count[] += 1; x^3),
    ]
    basis = quadbasis(funs, nothing, -1.0, 1.0)
    rule = GeneralizedGauss.LowerPrincipalOdd(basis, moments)

    _, _, order_outputs = GeneralizedGauss._mads_trial_outputs(
        rule, [0.5, -0.5], Float64)
    @test order_outputs == [Inf, 1.0]
    @test evaluation_count[] == 0

    _, _, support_outputs = GeneralizedGauss._mads_trial_outputs(
        rule, [-2.0, 0.5], Float64)
    @test support_outputs == [Inf, 1.0]
    @test evaluation_count[] == 0
end
