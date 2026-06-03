@testset "Fast polynomial quadrature references" begin
    basis6, moments6 = polynomial_basis_and_moments(6, Float64)
    basis5, moments5 = polynomial_basis_and_moments(5, Float64)

    gl3_w, gl3_x = reference_gl3(Float64)
    lgl4_w, lgl4_x = reference_lgl4(Float64)
    grl3_w, grl3_x = reference_left_radau3(Float64)
    grr3_w, grr3_x = reference_right_radau3(Float64)

    @testset "3-point Gauss-Legendre" begin
        w, x = compute_gauss_rule(basis6, moments6; principal=:lower)
        assert_rule_matches(w, x, gl3_w, gl3_x; atol=1e-11, rtol=1e-11)
        @test basis_residual_norm(basis6, moments6, w, x) <= 1e-12
    end

    @testset "4-point Gauss-Lobatto-Legendre" begin
        w, x = compute_gauss_rule(basis6, moments6; principal=:upper)
        assert_rule_matches(w, x, lgl4_w, lgl4_x; atol=1e-11, rtol=1e-11)
        @test basis_residual_norm(basis6, moments6, w, x) <= 1e-12
    end

    @testset "3-point left Radau" begin
        w, x = compute_gauss_rule(basis5, moments5;
            principal=:lower, add_endpoint=:left)
        assert_rule_matches(w, x, grl3_w, grl3_x; atol=1e-11, rtol=1e-11)
        @test basis_residual_norm(basis5, moments5, w, x) <= 1e-12
    end

    @testset "3-point right Radau" begin
        w, x = compute_gauss_rule(basis5, moments5;
            principal=:upper, add_endpoint=:right)
        assert_rule_matches(w, x, grr3_w, grr3_x; atol=1e-11, rtol=1e-11)
        @test basis_residual_norm(basis5, moments5, w, x) <= 1e-12
    end

    @testset "Even-basis continuation endpoint choices" begin
        for add_endpoint in (:left, :right)
            w_lg, x_lg = compute_gauss_rule(basis6, moments6;
                principal=:lower, add_endpoint)
            assert_rule_matches(w_lg, x_lg, gl3_w, gl3_x;
                atol=1e-11, rtol=1e-11)

            w_lgl, x_lgl = compute_gauss_rule(basis6, moments6;
                principal=:upper, add_endpoint)
            assert_rule_matches(w_lgl, x_lgl, lgl4_w, lgl4_x;
                atol=1e-11, rtol=1e-11)
        end
    end
end
