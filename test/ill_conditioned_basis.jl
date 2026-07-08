using LinearAlgebra

import GeneralizedGauss: gauss_legendre

@testset "Orthogonalization improves a badly scaled basis" begin
    # These functions span the degree-seven polynomials, but their scales
    # decrease by fourteen orders of magnitude. The basis is intentionally
    # much worse conditioned than ordinary monomials.
    N = 8
    scale_ratio = 1e-2
    scales = [scale_ratio^k for k in 0:N-1]
    funs = [let k = k, scale = scales[k + 1]
        x -> scale * x^k
    end for k in 0:N-1]
    fun_derivs = vcat(
        x -> zero(x),
        [let k = k, scale = scales[k + 1]
            x -> scale * k * x^(k - 1)
        end for k in 1:N-1],
    )
    basis = quadbasis(funs, fun_derivs, 0.0, 1.0)

    # Supply exact moments so this test isolates the effect of the basis used
    # by the quadrature algorithm, rather than numerical integration error.
    moments = [scales[k + 1] / (k + 1) for k in 0:N-1]

    # The expected rule is the four-point Gauss-Legendre rule mapped from
    # [-1, 1] to [0, 1].
    reference_x, reference_w = gauss_legendre(N ÷ 2, Float64)
    reference_x = (reference_x .+ 1) ./ 2
    reference_w ./= 2

    raw_w, raw_x = compute_gauss_rule(basis, moments;
        intermediate_tolerance=:strict)

    # Orthogonalization removes the harmful scale disparity before the
    # continuation solve, even though the change-of-basis matrix is itself
    # extremely ill-conditioned.
    orth_basis, transform = redirect_stderr(devnull) do
        orthogonalize_basis(basis)
    end
    orth_w, orth_x = compute_gauss_rule(orth_basis, transform * moments)

    # Sort defensively before comparing rules. The continuation currently
    # returns ordered nodes, but node ordering is not the behavior under test.
    raw_order = sortperm(raw_x)
    orth_order = sortperm(orth_x)
    raw_node_error = norm(raw_x[raw_order] - reference_x, Inf)
    raw_weight_error = norm(raw_w[raw_order] - reference_w, Inf)
    orth_node_error = norm(orth_x[orth_order] - reference_x, Inf)
    orth_weight_error = norm(orth_w[orth_order] - reference_w, Inf)

    @test cond(Matrix(transform)) > 1e16
    @test raw_node_error > 1e-4
    @test raw_weight_error > 1e-4
    @test orth_node_error < 1e-10
    @test orth_weight_error < 1e-10
    @test orth_node_error < raw_node_error / 1e8
    @test orth_weight_error < raw_weight_error / 1e8
end
