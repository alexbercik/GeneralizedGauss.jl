@testset "Solver paths reproduce 3-point Gauss-Legendre" begin
    funs, derivs, a, b = polynomial_basis_data(6, Float64)
    moments = exact_monomial_moments(6, a, b)
    ref_w, ref_x = reference_gl3(Float64)

    analytic_basis = quadbasis(funs, derivs, a, b)
    finite_diff_basis = quadbasis(funs, nothing, a, b)
    mads_basis = finite_diff_basis

    @testset "analytic derivatives" begin
        w, x = compute_gauss_rule(analytic_basis, moments; principal=:lower)
        assert_rule_matches(w, x, ref_w, ref_x; atol=1e-11, rtol=1e-11)
        @test basis_residual_norm(analytic_basis, moments, w, x) <= 1e-12
    end

    @testset "finite-difference derivatives" begin
        w, x = redirect_stderr(devnull) do
            compute_gauss_rule(finite_diff_basis, moments; principal=:lower)
        end
        assert_rule_matches(w, x, ref_w, ref_x; atol=1e-8, rtol=1e-8)
        @test basis_residual_norm(finite_diff_basis, moments, w, x) <= 1e-8
    end

    @testset "MADS derivative-free solve" begin
        w, x = compute_gauss_rule(mads_basis, moments;
            principal=:lower, differentiable=false)
        assert_rule_matches(w, x, ref_w, ref_x; atol=1e-6, rtol=1e-6)
        @test basis_residual_norm(mads_basis, moments, w, x) <= 1e-6
    end
end
