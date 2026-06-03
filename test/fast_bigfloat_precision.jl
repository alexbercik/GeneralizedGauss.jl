@testset "Fast BigFloat polynomial precision" begin
    setprecision(BigFloat, 192) do
        tol = BigFloat("1e-30")

        @testset "3-point Gauss-Legendre reaches 30 digits" begin
            basis, moments = polynomial_basis_and_moments(6, BigFloat)
            w, x = compute_gauss_rule(basis, moments; principal=:lower)
            ref_w, ref_x = reference_gl3(BigFloat)

            assert_rule_matches(w, x, ref_w, ref_x; atol=tol, rtol=tol)
            @test digits_of_agreement(x, ref_x) >= 30
            @test digits_of_agreement(w, ref_w) >= 30
        end

        @testset "3-point Radau reaches 30 digits" begin
            basis, moments = polynomial_basis_and_moments(5, BigFloat)

            left_w, left_x = compute_gauss_rule(basis, moments;
                principal=:lower, add_endpoint=:left)
            left_ref_w, left_ref_x = reference_left_radau3(BigFloat)
            assert_rule_matches(left_w, left_x, left_ref_w, left_ref_x;
                atol=tol, rtol=tol)

            right_w, right_x = compute_gauss_rule(basis, moments;
                principal=:upper, add_endpoint=:right)
            right_ref_w, right_ref_x = reference_right_radau3(BigFloat)
            assert_rule_matches(right_w, right_x, right_ref_w, right_ref_x;
                atol=tol, rtol=tol)
        end
    end
end
