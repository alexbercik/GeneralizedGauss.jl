@testset "Smooth non-polynomial quadrature" begin
    basis, moments = exp_poly_basis_and_moments(Float64)

    # Five functions produce a 3-point principal rule, so this still exercises
    # the continuation and nonlinear solve machinery.
    w, x = compute_gauss_rule(basis, moments;
        principal=:lower, add_endpoint=:left)

    @test length(w) == 3
    @test length(x) == 3
    @test all(isfinite, w)
    @test all(isfinite, x)
    @test all(diff(x) .> 0)
    @test basis_residual_norm(basis, moments, w, x) <= 1e-10
end
