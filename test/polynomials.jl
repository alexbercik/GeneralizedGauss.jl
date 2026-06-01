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

function monomial_basis_without_derivatives(n, ::Type{T}=Float64) where {T}
    funs = [x -> x^k for k in 0:n-1]
    quadbasis(funs, nothing, T(-1), T(1))
end

function monomial_moments(n, ::Type{T}=Float64) where {T}
    [iseven(k) ? T(2) / T(k + 1) : zero(T) for k in 0:n-1]
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

const GRL3_X = [-1.0, (1 - sqrt(6)) / 5, (1 + sqrt(6)) / 5]
const GRL3_W = [2 / 9, (16 + sqrt(6)) / 18, (16 - sqrt(6)) / 18]

@testset "Principal Newton canonical recovery" begin
    linear_basis = quadbasis(
        Function[x -> one(x), x -> x],
        Function[x -> zero(x), x -> one(x)],
        -2.0, 2.0)
    linear_moments = [1.0, 0.25]
    linear_sample(xi) =
        GeneralizedGauss.CanonicalSample(xi, xi - linear_moments[end],
            [1.0], [xi])
    linear_state = GeneralizedGauss._canonical_bracket_state(
        linear_sample(-1.0), linear_sample(1.0))

    warm_seeds = Float64[]
    compute_linear = function (dict, moments, xi, w_seed, x_seed; kwargs...)
        push!(warm_seeds, only(x_seed))
        true, [1.0], [xi], nothing
    end

    refined = GeneralizedGauss._refine_canonical_bracket(
        compute_linear, "linear canonical", linear_basis, linear_moments,
        linear_state)
    @test only(warm_seeds) == 1.0
    @test refined.width < linear_state.width
    @test abs(refined.best.F) <= abs(linear_state.best.F)

    empty!(warm_seeds)
    principal_seeds = Float64[]
    solve_principal = function (state)
        seed = state.best
        push!(principal_seeds, only(seed.x))
        if length(principal_seeds) == 1
            false, [99.0], [99.0], (; ratio=Inf)
        else
            true, seed.w, seed.x, (; ratio=0.0)
        end
    end
    converged, w, x, _ =
        GeneralizedGauss._compute_principal_with_canonical_recovery(
            solve_principal, _ -> false, compute_linear,
            "linear principal", "linear canonical",
            linear_basis, linear_moments, linear_state)
    @test converged
    @test principal_seeds == [1.0, 0.25]
    @test warm_seeds == [1.0]
    @test w == [1.0]
    @test x == [0.25]

    invalid_canonical = function (dict, moments, xi, w_seed, x_seed; kwargs...)
        true, [1.0], [10.0], nothing
    end
    invalid_error = try
        GeneralizedGauss._refine_canonical_bracket(
            invalid_canonical, "invalid canonical",
            linear_basis, linear_moments, linear_state)
        nothing
    catch e
        e
    end
    @test invalid_error isa ErrorException
    @test occursin("conditioning problems", sprint(showerror, invalid_error))
    @test occursin("not a CT-system", sprint(showerror, invalid_error))

    quadratic_basis = quadbasis(
        Function[x -> one(x), x -> x^2],
        Function[x -> zero(x), x -> 2x],
        0.0, 2.0)
    quadratic_moments = [1.0, 0.25]
    quadratic_sample(xi) =
        GeneralizedGauss.CanonicalSample(xi, xi^2 - quadratic_moments[end],
            [1.0], [xi])
    quadratic_state = GeneralizedGauss._canonical_bracket_state(
        quadratic_sample(0.0), quadratic_sample(2.0))
    canonical_attempts = Ref(0)
    principal_attempts = Ref(0)
    compute_quadratic =
        function (dict, moments, xi, w_seed, x_seed; kwargs...)
            canonical_attempts[] += 1
            true, [1.0], [xi], nothing
        end
    always_fail = function (state)
        seed = state.best
        principal_attempts[] += 1
        false, seed.w, seed.x, (; ratio=Inf)
    end
    converged, _, _, _ =
        GeneralizedGauss._compute_principal_with_canonical_recovery(
            always_fail, _ -> false, compute_quadratic,
            "quadratic principal", "quadratic canonical",
            quadratic_basis, quadratic_moments, quadratic_state)
    @test !converged
    @test canonical_attempts[] ==
          GeneralizedGauss._MAX_PRINCIPAL_RECOVERY_REFINEMENTS
    @test principal_attempts[] ==
          GeneralizedGauss._MAX_PRINCIPAL_RECOVERY_REFINEMENTS + 1
end

@testset "One-point scalar solve" begin
    funs = Function[x -> one(x), x -> x]
    fun_derivs = Function[x -> zero(x), x -> one(x)]
    moments = [2.0, 0.0]

    basis_newton = quadbasis(funs, fun_derivs, -1.0, 1.0)
    w_newton, x_newton = compute_gauss_rule(basis_newton, moments)
    assert_rule_matches(w_newton, x_newton, [2.0], [0.0])

    basis_no_deriv = quadbasis(funs, nothing, -1.0, 1.0)
    w_brent, x_brent = compute_gauss_rule(
        basis_no_deriv, moments; differentiable=false)
    assert_rule_matches(w_brent, x_brent, [2.0], [0.0])

    w_two, x_two = compute_gauss_rule(basis_no_deriv, moments; principal=:upper)
    assert_rule_matches(w_two, x_two, [1.0, 1.0], [-1.0, 1.0])

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
        @test_logs (:warn, r"Analytic first derivatives are missing") compute_gauss_rule(
            basis_no_deriv, moments)
    assert_rule_matches(w_fallback, x_fallback, [2.0], [0.0])

    higher_funs = Function[x -> one(x), x -> x, x -> x^2]
    higher_basis_no_deriv = quadbasis(higher_funs, nothing, -1.0, 1.0)
    w_fd, x_fd =
        @test_logs (:warn, r"Analytic first derivatives are missing") compute_gauss_rule(
            higher_basis_no_deriv, [2.0, 0.0, 2 / 3])
    assert_rule_matches(w_fd, x_fd, [0.5, 1.5], [-1.0, 1 / 3])
end

@testset "Centered basis derivatives" begin
    for T in (Float64, BigFloat)
        basis = monomial_basis(4)
        basis_no_deriv = monomial_basis_without_derivatives(4, T)
        moments = monomial_moments(4, T)
        rule = GeneralizedGauss.LowerPrincipalOdd(basis_no_deriv, moments)
        w = T[1, 1]
        x = T[-inv(sqrt(T(3))), inv(sqrt(T(3)))]
        newton_x = GeneralizedGauss.quad_to_newton(rule, w, x)
        J_fd = zeros(T, size(rule))
        GeneralizedGauss.jacobian!(J_fd, rule, newton_x,
            GeneralizedGauss.funeval_deriv_or_finite_diff)

        analytic_rule = GeneralizedGauss.LowerPrincipalOdd(
            T === Float64 ? basis : monomial_basis_without_derivatives(4, T),
            moments)
        J_exact = similar(J_fd)
        GeneralizedGauss.jacobian!(J_exact, analytic_rule, newton_x,
            (_, j, y) -> j == 1 ? zero(y) : T(j - 1) * y^(j - 2))

        # Weight derivatives remain exact. Only node columns use the centered
        # finite-difference helper.
        @test J_fd[:, 1:length(w)] == J_exact[:, 1:length(w)]
        @test maximum(abs, J_fd - J_exact) <= 100 * cbrt(eps(T))^2
        @test GeneralizedGauss.funeval_deriv_or_finite_diff(
            basis_no_deriv, 3, T(-1)) ≈ T(-2)
    end

    analytic_calls = Ref(0)
    mixed_basis = quadbasis(
        Function[x -> x^2, x -> x^3],
        Any[x -> (analytic_calls[] += 1; 7.0), nothing],
        -1.0, 1.0)
    @test GeneralizedGauss.funeval_deriv_or_finite_diff(
        mixed_basis, 1, 0.5) == 7.0
    @test analytic_calls[] == 1
    @test GeneralizedGauss.funeval_deriv_or_finite_diff(
        mixed_basis, 2, 0.5) ≈ 0.75
end

@testset "MADS adapter" begin
    moments = monomial_moments(4)
    evaluation_count = Ref(0)
    counting_funs = Function[
        x -> (evaluation_count[] += 1; one(x)),
        x -> (evaluation_count[] += 1; x),
        x -> (evaluation_count[] += 1; x^2),
        x -> (evaluation_count[] += 1; x^3),
    ]
    basis = quadbasis(counting_funs, nothing, -1.0, 1.0)
    rule = GeneralizedGauss.LowerPrincipalOdd(basis, moments)

    # Invalid MADS trials must hit the extreme barrier before basis evaluation.
    _, _, order_outputs = GeneralizedGauss._mads_trial_outputs(
        rule, [0.5, -0.5], Float64)
    @test order_outputs == [Inf, 1.0]
    @test evaluation_count[] == 0

    _, _, support_outputs = GeneralizedGauss._mads_trial_outputs(
        rule, [-2.0, 0.5], Float64)
    @test support_outputs == [Inf, 1.0]
    @test evaluation_count[] == 0

    _, _, valid_outputs = GeneralizedGauss._mads_trial_outputs(
        rule, [-0.5, 0.5], Float64)
    @test valid_outputs[1] < 1 / 36
    @test valid_outputs[2] == -1.0
    @test evaluation_count[] > 0
    @test GeneralizedGauss._mads_rule_is_valid(
        [-1.0, 3.0], [-0.5, 0.5], -1.0, 1.0)

    # MADS searches only free nodes. Weights are projected with a least-squares
    # solve in the rule's working type for every valid node trial.
    projected_w = zeros(2)
    basis_matrix = Matrix{Float64}(undef, length(basis), 2)
    objective = GeneralizedGauss._mads_project_weights!(
        projected_w, basis_matrix, rule, [-inv(sqrt(3.0)), inv(sqrt(3.0))])
    @test projected_w ≈ [1.0, 1.0]
    @test objective < eps(Float64)
    canonical_mesh = GeneralizedGauss._mads_canonical_initial_mesh(
        rule, 0.1)
    @test canonical_mesh == [0.1, 0.1]

    fixed_rule = GeneralizedGauss.UpperPrincipalOdd(
        monomial_basis_without_derivatives(4), moments)
    left = GeneralizedGauss.CanonicalSample(
        -0.4, -1.0, [0.2, 1.6, 0.2], [-1.0, -0.4, 1.0])
    right = GeneralizedGauss.CanonicalSample(
        0.4, 1.0, [0.2, 1.6, 0.2], [-1.0, 0.4, 1.0])
    bracket = GeneralizedGauss._canonical_bracket_state(left, right)
    @test bracket.xi_index == 2
    lower, upper = GeneralizedGauss._mads_variable_bounds(fixed_rule, bracket)

    # Only the released xi node receives NOMAD bounds. Support and ordering
    # remain extreme-barrier checks for every node.
    @test lower == [-0.4]
    @test upper == [0.4]
    @test GeneralizedGauss._mads_node_variables(fixed_rule, left.x) == [-0.4]
    principal_mesh = GeneralizedGauss._mads_initial_mesh(
        fixed_rule, 0.1, bracket)
    @test principal_mesh ≈ [0.8]

    bf_basis = monomial_basis_without_derivatives(4, BigFloat)
    bf_rule = GeneralizedGauss.LowerPrincipalOdd(
        bf_basis, monomial_moments(4, BigFloat))
    _, bf_w, bf_x, _ = GeneralizedGauss._solve_system_mads(
        bf_rule, BigFloat[1.05, 0.95], BigFloat[-0.6, 0.6];
        dx=BigFloat("0.1"), max_bb_eval=5)
    @test eltype(bf_w) === BigFloat
    @test eltype(bf_x) === BigFloat
    @test GeneralizedGauss._mads_solver_tolerance(BigFloat) >=
          BigFloat(10) * BigFloat(eps(Float64))

    infinite_basis = quadbasis(
        Function[x -> one(x), x -> x, x -> x^2, x -> x^3],
        nothing, -Inf, Inf)
    infinite_rule = GeneralizedGauss.LowerPrincipalOdd(infinite_basis, moments)
    @test_throws ErrorException GeneralizedGauss._mads_support_bounds(infinite_rule)
end

@testset "Automatic finite-difference and MADS paths" begin
    analytic_basis3 = monomial_basis(3)
    analytic_basis4 = monomial_basis(4)
    basis3 = monomial_basis_without_derivatives(3)
    moments3 = monomial_moments(3)
    basis4 = monomial_basis_without_derivatives(4)
    moments4 = monomial_moments(4)

    w_radau_analytic, x_radau_analytic =
        compute_gauss_rule(analytic_basis3, moments3)
    w_gauss_analytic, x_gauss_analytic =
        compute_gauss_rule(analytic_basis4, moments4)
    w_lobatto_analytic, x_lobatto_analytic =
        compute_gauss_rule(analytic_basis4, moments4; principal=:upper)

    w_radau_fd, x_radau_fd =
        @test_logs (:warn, r"Analytic first derivatives are missing") compute_gauss_rule(
            basis3, moments3)
    assert_rule_matches(w_radau_fd, x_radau_fd, [0.5, 1.5], [-1.0, 1 / 3])
    assert_rule_matches(w_radau_fd, x_radau_fd, w_radau_analytic, x_radau_analytic)

    w_gauss_fd, x_gauss_fd =
        @test_logs (:warn, r"Analytic first derivatives are missing") compute_gauss_rule(
            basis4, moments4)
    assert_rule_matches(
        w_gauss_fd, x_gauss_fd, [1.0, 1.0], [-inv(sqrt(3.0)), inv(sqrt(3.0))])
    assert_rule_matches(w_gauss_fd, x_gauss_fd, w_gauss_analytic, x_gauss_analytic)

    w_lobatto_fd, x_lobatto_fd =
        @test_logs (:warn, r"Analytic first derivatives are missing") compute_gauss_rule(
            basis4, moments4; principal=:upper)
    assert_rule_matches(
        w_lobatto_fd, x_lobatto_fd, [1 / 3, 4 / 3, 1 / 3], [-1.0, 0.0, 1.0])
    assert_rule_matches(
        w_lobatto_fd, x_lobatto_fd, w_lobatto_analytic, x_lobatto_analytic)

    w_radau_mads, x_radau_mads = compute_gauss_rule(
        basis3, moments3; differentiable=false)
    assert_rule_matches(w_radau_mads, x_radau_mads, [0.5, 1.5], [-1.0, 1 / 3])
    assert_rule_matches(
        w_radau_mads, x_radau_mads, w_radau_analytic, x_radau_analytic)

    w_gauss_mads, x_gauss_mads = compute_gauss_rule(
        basis4, moments4; differentiable=false)
    assert_rule_matches(
        w_gauss_mads, x_gauss_mads, [1.0, 1.0], [-inv(sqrt(3.0)), inv(sqrt(3.0))])
    assert_rule_matches(w_gauss_mads, x_gauss_mads, w_gauss_analytic, x_gauss_analytic)

    w_lobatto_mads, x_lobatto_mads = compute_gauss_rule(
        basis4, moments4; principal=:upper, differentiable=false)
    assert_rule_matches(
        w_lobatto_mads, x_lobatto_mads, [1 / 3, 4 / 3, 1 / 3], [-1.0, 0.0, 1.0])
    assert_rule_matches(
        w_lobatto_mads, x_lobatto_mads, w_lobatto_analytic, x_lobatto_analytic)

    bf_basis = monomial_basis_without_derivatives(3, BigFloat)
    bf_moments = monomial_moments(3, BigFloat)
    bf_w, bf_x, bf_xi_checkpoints, bf_w_checkpoints, bf_x_checkpoints =
        compute_gauss_rules(bf_basis, bf_moments; differentiable=false)
    @test eltype(bf_w) === BigFloat
    @test eltype(bf_x) === BigFloat
    @test eltype(bf_xi_checkpoints) === BigFloat
    @test all(w -> eltype(w) === BigFloat, bf_w_checkpoints)
    @test all(x -> eltype(x) === BigFloat, bf_x_checkpoints)
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

    @testset "3-point left Gauss-Radau-Legendre" begin
        w, x = compute_gauss_rule(ChebyshevT(5); principal=:lower)
        assert_rule_matches(w, x, GRL3_W, GRL3_X)
    end

    @testset "Left and right continuation paths" begin
        for add_endpoint in (:left, :right)
            w_gauss, x_gauss = compute_gauss_rule(
                ChebyshevT(6); principal=:lower, add_endpoint)
            assert_rule_matches(w_gauss, x_gauss, GL3_W, GL3_X)

            w_lobatto, x_lobatto = compute_gauss_rule(
                ChebyshevT(6); principal=:upper, add_endpoint)
            assert_rule_matches(w_lobatto, x_lobatto, GLL4_W, GLL4_X)
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
