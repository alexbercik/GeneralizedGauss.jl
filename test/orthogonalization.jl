using Test
using BasisFunctions
using GeneralizedGauss
using LinearAlgebra

import GeneralizedGauss: funeval, funeval_deriv, gauss_legendre

# ============================================================================
# Helpers
# ============================================================================

"""
Evaluate every basis function at `x` and return a vector.
Works for both GenericFunctionSet and BasisFunctions Dictionary types.
"""
eval_all(basis::BasisFunctions.Dictionary, x) =
    [BasisFunctions.unsafe_eval_element(basis, i, x) for i in eachindex(basis)]

eval_all_derivs(basis::BasisFunctions.Dictionary, x) =
    [BasisFunctions.unsafe_eval_element_derivative(basis, i, x, 1) for i in eachindex(basis)]

"""
Build the Gram matrix by evaluating ⟨ψ_i, ψ_j⟩ on a high-order GL rule
(independent of the one used to build the orthogonalization).
"""
function verify_orthonormality(basis, a, b; measure=nothing, quad_order=120)
    T = typeof(a)
    nodes, weights = gauss_legendre(quad_order, T)
    scale = (b - a) / 2
    shift = (a + b) / 2
    qx = scale .* nodes  .+ shift
    qw = scale .* weights
    if measure !== nothing
        qw = qw .* [measure(qx[k]) for k in 1:quad_order]
    end

    n = length(basis)
    G = zeros(T, n, n)
    for i in 1:n, j in 1:n
        G[i, j] = sum(qw[k] * funeval(basis, i, qx[k]) * funeval(basis, j, qx[k])
                       for k in 1:quad_order)
    end
    G
end


# ============================================================================
# Tests
# ============================================================================

@testset "Gauss-Legendre quadrature nodes & weights" begin
    # Compare against known 3-point GL rule on [-1,1]
    nodes, weights = gauss_legendre(3, Float64)
    @test nodes ≈ [-sqrt(3/5), 0.0, sqrt(3/5)] atol=1e-14
    @test weights ≈ [5/9, 8/9, 5/9] atol=1e-14

    # 1-point rule
    n1, w1 = gauss_legendre(1, Float64)
    @test n1 ≈ [0.0]
    @test w1 ≈ [2.0]

    # BigFloat precision: weights should sum to 2
    nodes_bf, weights_bf = gauss_legendre(50, BigFloat)
    @test abs(sum(weights_bf) - 2) < BigFloat(10)^(-50)
end

@testset "Orthogonalization: Lebesgue measure" begin
    setprecision(BigFloat, 256)

    N = 5  # number of basis functions (monomials 1, x, ..., x^{N-1})
    a = BigFloat(0)
    b = BigFloat(1)

    # --- Our implementation: monomial GenericFunctionSet ---
    funs = [x -> x^k for k in 0:N-1]
    fun_derivs = vcat(x -> zero(x), [x -> k * x^(k-1) for k in 1:N-1])
    basis_gfs = quadbasis(funs, fun_derivs, a, b)

    orth_basis, T_mat = orthogonalize_basis(basis_gfs)

    @testset "T_mat is lower-triangular" begin
        for i in 1:N, j in (i+1):N
            @test abs(T_mat[i, j]) < BigFloat(10)^(-40)
        end
    end

    @testset "T_mat matches Cholesky of exact Hilbert matrix (analytical reference)" begin
        # The Gram matrix of monomials on [0,1] is the Hilbert matrix:
        #   H_{ij} = ∫_0^1 x^{i-1} x^{j-1} dx = 1/(i+j-1)
        # QR of A gives RᵀR = AᵀA ≈ H, so T = R⁻ᵀ ≈ L⁻¹ where H = LLᵀ.
        H = [BigFloat(1) / (i + j - 1) for i in 1:N, j in 1:N]
        L_ref = cholesky(Symmetric(H)).L
        T_ref = Matrix(L_ref) \ I(N)

        @test norm(T_mat - T_ref) < BigFloat(10)^(-30)
    end

    @testset "Result is orthonormal (independent verification)" begin
        G_check = verify_orthonormality(orth_basis, a, b; quad_order=100)
        @test norm(G_check - I(N)) < BigFloat(10)^(-20)
    end

    @testset "Derivatives are consistently transformed" begin
        # Check at a few test points that ψ'(x) matches numerical derivative of ψ(x)
        test_pts = [BigFloat("0.1"), BigFloat("0.5"), BigFloat("0.9")]
        h = BigFloat(10)^(-30)
        for x0 in test_pts
            for i in 1:N
                deriv_analytical = funeval_deriv(orth_basis, i, x0)
                deriv_numerical  = (funeval(orth_basis, i, x0 + h) -
                                    funeval(orth_basis, i, x0 - h)) / (2h)
                @test abs(deriv_analytical - deriv_numerical) < BigFloat(10)^(-20)
            end
        end
    end

    @testset "Moment transformation is consistent" begin
        # Original moments: ∫_0^1 x^k dx = 1/(k+1)
        old_moments = [BigFloat(1) / (k + 1) for k in 0:N-1]
        new_moments_via_transform = T_mat * old_moments

        # Compute moments of the new basis directly via quadrature
        nodes, weights = gauss_legendre(100, BigFloat)
        scale = (b - a) / 2;  shift = (a + b) / 2
        qx = scale .* nodes .+ shift
        qw = scale .* weights
        new_moments_direct = [sum(qw[k] * funeval(orth_basis, i, qx[k])
                                  for k in 1:100) for i in 1:N]

        @test norm(new_moments_via_transform - new_moments_direct) < BigFloat(10)^(-20)
    end

    # --- Comparison with BasisFunctions.orthogonalize ---
    #
    # BasisFunctions uses Löwdin (symmetric G^{-1/2}), while our method uses
    # QR (triangular R⁻ᵀ).  These produce different orthonormal bases
    # spanning the same space.  We verify:
    #   1. Both are independently orthonormal.
    #   2. They span the same space (the cross-Gram matrix is orthogonal).
    @testset "Compare with BasisFunctions.orthogonalize (ChebyshevT, Lebesgue)" begin
        cheb = ChebyshevT(N) → (a..b)
        mu_lebesgue = BasisFunctions.GenericWeight(support(cheb), x -> one(x))

        bf_orth  = orthogonalize(cheb, mu_lebesgue)
        our_orth, _ = orthogonalize_basis(cheb; measure=x -> one(x))

        # Both should be orthonormal w.r.t. the same inner product
        G_bf  = verify_orthonormality(bf_orth,  a, b; quad_order=100)
        G_our = verify_orthonormality(our_orth, a, b; quad_order=100)

        @test norm(G_bf  - I(N)) < BigFloat(10)^(-15)
        @test norm(G_our - I(N)) < BigFloat(10)^(-15)

        # Cross-Gram matrix ⟨ψ^bf_i, ψ^our_j⟩ should be orthogonal (Q Qᵀ = I)
        nodes_v, weights_v = gauss_legendre(100, BigFloat)
        scale = (b - a) / 2;  shift = (a + b) / 2
        qx = scale .* nodes_v .+ shift
        qw = scale .* weights_v
        cross = zeros(BigFloat, N, N)
        for i in 1:N, j in 1:N
            cross[i, j] = sum(qw[k] *
                BasisFunctions.unsafe_eval_element(bf_orth, i, qx[k]) *
                funeval(our_orth, j, qx[k]) for k in 1:100)
        end
        @test norm(cross * cross' - I(N)) < BigFloat(10)^(-15)
    end
end

@testset "Orthogonalization: custom weight w(x) = exp(x)" begin
    setprecision(BigFloat, 256)

    N = 5
    a = BigFloat(0)
    b = BigFloat(1)
    w_fun = x -> exp(x)

    # --- Our implementation ---
    cheb = ChebyshevT(N) → (a..b)
    # Non-polynomial weight needs more quadrature points than the default
    our_orth, our_T_mat = orthogonalize_basis(cheb; measure=w_fun, quad_order=50)

    @testset "Result is orthonormal w.r.t. exp(x) weight" begin
        G_check = verify_orthonormality(our_orth, a, b; measure=w_fun, quad_order=100)
        @test norm(G_check - I(N)) < BigFloat(10)^(-20)
    end

    # --- Comparison with BasisFunctions.orthogonalize ---
    # BasisFunctions uses Löwdin (symmetric), ours uses QR (triangular).
    # Different orthonormal bases spanning the same space.
    @testset "Compare with BasisFunctions.orthogonalize (ChebyshevT, exp weight)" begin
        mu = BasisFunctions.GenericWeight(support(cheb), w_fun)
        bf_orth = orthogonalize(cheb, mu)

        # Both orthonormal w.r.t. exp(x) weight
        G_bf  = verify_orthonormality(bf_orth,  a, b; measure=w_fun, quad_order=100)
        G_our = verify_orthonormality(our_orth, a, b; measure=w_fun, quad_order=100)
        @test norm(G_bf  - I(N)) < BigFloat(10)^(-15)
        @test norm(G_our - I(N)) < BigFloat(10)^(-15)

        # Cross-Gram matrix should be orthogonal
        nodes_v, weights_v = gauss_legendre(100, BigFloat)
        scale = (b - a) / 2;  shift = (a + b) / 2
        qx = scale .* nodes_v .+ shift
        qw = scale .* weights_v .* [w_fun(qx[k]) for k in 1:100]
        cross = zeros(BigFloat, N, N)
        for i in 1:N, j in 1:N
            cross[i, j] = sum(qw[k] *
                BasisFunctions.unsafe_eval_element(bf_orth, i, qx[k]) *
                funeval(our_orth, j, qx[k]) for k in 1:100)
        end
        @test norm(cross * cross' - I(N)) < BigFloat(10)^(-15)
    end

    @testset "Derivatives are correctly transformed under exp weight" begin
        test_pts = [BigFloat("0.1"), BigFloat("0.5"), BigFloat("0.9")]
        h = BigFloat(10)^(-30)
        for x0 in test_pts
            for i in 1:N
                deriv_analytical = funeval_deriv(our_orth, i, x0)
                deriv_numerical  = (funeval(our_orth, i, x0 + h) -
                                    funeval(our_orth, i, x0 - h)) / (2h)
                @test abs(deriv_analytical - deriv_numerical) < BigFloat(10)^(-20)
            end
        end
    end
end

@testset "Orthogonalization: GenericFunctionSet without derivatives" begin
    setprecision(BigFloat, 256)

    N = 4
    a = BigFloat(-1)
    b = BigFloat(1)
    funs = [x -> x^k for k in 0:N-1]
    basis_no_deriv = quadbasis(funs, nothing, a, b)

    orth_basis, T_mat = orthogonalize_basis(basis_no_deriv)

    @testset "Orthonormality holds" begin
        G_check = verify_orthonormality(orth_basis, a, b; quad_order=80)
        @test norm(G_check - I(N)) < BigFloat(10)^(-20)
    end

    @testset "No derivatives in output" begin
        @test orth_basis.fun_derivs === nothing
    end
end

@testset "Orthogonalization feeds into compute_gauss_rule" begin
    setprecision(BigFloat, 128)

    # Use monomials on [0,1] — known to be ill-conditioned past degree ~7.
    # Orthogonalize first, then compute the Gauss rule.
    N = 6   # 6 monomials → 3-point GL rule on [0,1]
    a = BigFloat(0)
    b = BigFloat(1)

    funs = [x -> x^k for k in 0:N-1]
    fun_derivs = vcat(x -> zero(x), [x -> k * x^(k-1) for k in 1:N-1])
    basis = quadbasis(funs, fun_derivs, a, b)

    # Analytical moments ∫_0^1 x^k dx = 1/(k+1)
    moments = [BigFloat(1) / (k + 1) for k in 0:N-1]

    # Orthogonalize and transform moments
    orth_basis, T_mat = orthogonalize_basis(basis)
    orth_moments = T_mat * moments

    # Reference: 3-point GL on [0,1] (mapped from [-1,1])
    gl3_x_ref = ([-sqrt(BigFloat(3)/5), BigFloat(0), sqrt(BigFloat(3)/5)] .+ 1) ./ 2
    gl3_w_ref = [BigFloat(5)/9, BigFloat(8)/9, BigFloat(5)/9] ./ 2

    w, x = compute_gauss_rule(orth_basis, orth_moments)

    @test length(w) == 3
    @test x ≈ gl3_x_ref atol=BigFloat(10)^(-20)
    @test w ≈ gl3_w_ref atol=BigFloat(10)^(-20)
end
