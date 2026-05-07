using Test
using BasisFunctions
using GeneralizedGauss
using LinearAlgebra
using Printf

import GeneralizedGauss: funeval, funeval_deriv, solver_tolerance, gauss_legendre

# ============================================================================
# Configuration
# ============================================================================

# We work at 120 decimal digits of precision and set the Newton solver
# tolerance to 10^{-110}, leaving ~10 digits of headroom for the
# working arithmetic.

const TARGET_DIGITS  = 120
const NEWTON_DIGITS  = 110
const PRECISION_BITS = ceil(Int, TARGET_DIGITS * log2(big(10)))

# Override the Newton solver tolerance for BigFloat to use our explicit setting
GeneralizedGauss.solver_tolerance(::Type{BigFloat}) = BigFloat(10)^(-NEWTON_DIGITS)

# ============================================================================
# Helpers
# ============================================================================

"""
Compute the number of correct decimal digits between `val` and `ref`.
Returns `TARGET_DIGITS` when the error is exactly zero.
"""
function digits_of_agreement(val, ref)
    err = abs(val - ref)
    err == 0 ? Float64(TARGET_DIGITS) : -Float64(log10(err))
end

"""
Compute the number of correct decimal digits between vectors (∞-norm).
"""
function digits_of_agreement_vec(v, v_ref)
    err = norm(v - v_ref, Inf)
    err == 0 ? Float64(TARGET_DIGITS) : -Float64(log10(err))
end

# Fixed-width table formatters for readable aligned output.
fmt_table_value(x) = @sprintf("%20.12e", Float64(x))
fmt_table_digits(d) = @sprintf("%7.1f", d)

"""
Reference n-point Gauss-Legendre rule on [-1, 1] via our own high-precision
`gauss_legendre` routine.  This is an independent computation (Newton on
Legendre recurrence) that does not go through the continuation algorithm.
"""
function reference_gl(n, ::Type{T}) where T
    gauss_legendre(n, T)
end

"""
Reference n-point Gauss-Legendre rule on [a, b] via affine map of [-1, 1].
"""
function reference_gl_mapped(n, a::T, b::T) where T
    x_ref, w_ref = gauss_legendre(n, T)
    scale = (b - a) / 2
    shift = (a + b) / 2
    x_mapped = scale .* x_ref .+ shift
    w_mapped = scale .* w_ref
    x_mapped, w_mapped
end

"""
Exact Lebesgue moments of monomials on [a, b]:
  ∫_a^b x^k dx = (b^{k+1} - a^{k+1}) / (k+1)
"""
exact_monomial_moments(N, a::T, b::T) where T =
    [(b^(k+1) - a^(k+1)) / T(k+1) for k in 0:N-1]

"""
Build a BigFloat monomial basis on [a, b] with derivatives.
"""
function bigfloat_monomial_basis(N, a::BigFloat, b::BigFloat)
    funs = [x -> x^k for k in 0:N-1]
    fun_derivs = vcat([x -> one(x)], [x -> BigFloat(k) * x^(k-1) for k in 1:N-1])
    quadbasis(funs, fun_derivs, a, b)
end


# ============================================================================
# Tests
# ============================================================================

const RUNNING_BIGFLOAT_PRECISION_AS_SCRIPT = abspath(PROGRAM_FILE) == @__FILE__
const SILENT_FAILURE_COUNT = Ref(0)
const SILENT_FAILURE_LABELS = String[]
function silent_check(cond::Bool, label::AbstractString)
    if !cond
        SILENT_FAILURE_COUNT[] += 1
        push!(SILENT_FAILURE_LABELS, String(label))
    end
end
struct SilentTableFailure <: Exception end

try
@testset verbose=true "BigFloat precision tests ($(TARGET_DIGITS) digits)" begin
    setprecision(BigFloat, PRECISION_BITS)

    # -----------------------------------------------------------------------
    # 1. Moment accuracy
    # -----------------------------------------------------------------------
    @testset verbose=true "Moment accuracy" begin
        N = 10
        a, b = BigFloat(-1), BigFloat(1)

        @testset "Exact monomial moments on [-1, 1]" begin
            exact = exact_monomial_moments(N, a, b)
            @test exact[1] == BigFloat(2)
            @test exact[2] == BigFloat(0)
            @test exact[3] == BigFloat(2) / 3
        end

        @testset "BasisFunctions Legendre moments (Lebesgue) on [-1, 1]" begin
            basis = Legendre(N) → (a..b)
            auto_moments = compute_moments(basis)

            # Exact: ∫ P_k(x) dx = 2 δ_{k0}
            exact_moments = zeros(BigFloat, N)
            exact_moments[1] = BigFloat(2)

            println("\n  Legendre moments on [-1,1] (Lebesgue):")
            println("  ┌───────┬──────────────────────┬──────────────────────┬─────────┐")
            println("  │   k   │    compute_moments    │       exact          │  digits │")
            println("  ├───────┼──────────────────────┼──────────────────────┼─────────┤")
            for k in 1:N
                d = digits_of_agreement(auto_moments[k], exact_moments[k])
                println("  │  $(lpad(k-1, 3)) │ $(fmt_table_value(auto_moments[k])) │ " *
                        "$(fmt_table_value(exact_moments[k])) │ $(fmt_table_digits(d)) │")
                @test d > 14  # at least Float64-level
            end
            println("  └───────┴──────────────────────┴──────────────────────┴─────────┘")
        end

        @testset "BasisFunctions ChebyshevT moments (Lebesgue) on [-1, 1]" begin
            basis = ChebyshevT(N) → (a..b)
            auto_moments = compute_moments(basis)

            function exact_cheb_lebesgue_moment(k)
                if k == 0;       return BigFloat(2)
                elseif isodd(k); return BigFloat(0)
                else;            return -BigFloat(2) / (BigFloat(k)^2 - 1)
                end
            end

            exact_moments = [exact_cheb_lebesgue_moment(k - 1) for k in 1:N]

            println("\n  ChebyshevT moments on [-1,1] (Lebesgue):")
            println("  ┌───────┬──────────────────────┬──────────────────────┬─────────┐")
            println("  │   k   │    compute_moments    │       exact          │  digits │")
            println("  ├───────┼──────────────────────┼──────────────────────┼─────────┤")
            for k in 1:N
                d = digits_of_agreement(auto_moments[k], exact_moments[k])
                println("  │  $(lpad(k-1, 3)) │ $(fmt_table_value(auto_moments[k])) │ " *
                        "$(fmt_table_value(exact_moments[k])) │ $(fmt_table_digits(d)) │")
                @test d > 14
            end
            println("  └───────┴──────────────────────┴──────────────────────┴─────────┘")
        end

        @testset "BasisFunctions ChebyshevT moments on [0, 1]" begin
            a01, b01 = BigFloat(0), BigFloat(1)
            basis = ChebyshevT(N) → (a01..b01)
            auto_moments = compute_moments(basis)

            # Independent verification via high-order GL quadrature
            Q = 200
            ref_nodes, ref_weights = gauss_legendre(Q, BigFloat)
            scale = (b01 - a01) / 2
            shift = (a01 + b01) / 2
            qx = scale .* ref_nodes .+ shift
            qw = scale .* ref_weights

            println("\n  ChebyshevT moments on [0,1] (Lebesgue):")
            println("  ┌───────┬──────────────────────┬──────────────────────┬─────────┐")
            println("  │   k   │    compute_moments    │    GL(200) ref       │  digits │")
            println("  ├───────┼──────────────────────┼──────────────────────┼─────────┤")
            for k in 1:N
                gl_moment = sum(qw[q] * funeval(basis, k, qx[q]) for q in 1:Q)
                d = digits_of_agreement(auto_moments[k], gl_moment)
                println("  │  $(lpad(k-1, 3)) │ $(fmt_table_value(auto_moments[k])) │ " *
                        "$(fmt_table_value(gl_moment)) │ $(fmt_table_digits(d)) │")
                @test d > 14
            end
            println("  └───────┴──────────────────────┴──────────────────────┴─────────┘")
        end
    end

    # -----------------------------------------------------------------------
    # 2. GL rule from Legendre basis (automatic moments)
    # -----------------------------------------------------------------------
    @testset verbose=true "GL rule from Legendre basis" begin
        println("\n  GL rules from Legendre basis (automatic moments):")
        println("  ┌────────────┬───────────────┬────────────────┐")
        println("  │    rule    │  node digits  │ weight digits   │")
        println("  ├────────────┼───────────────┼────────────────┤")

        for n_rule in [3, 5, 7]
            N_basis = 2 * n_rule
            a, b = BigFloat(-1), BigFloat(1)
            basis = Legendre(N_basis) → (a..b)

            w, x = compute_gauss_rule(basis)
            x_ref, w_ref = reference_gl(n_rule, BigFloat)

            dx = digits_of_agreement_vec(x, x_ref)
            dw = digits_of_agreement_vec(w, w_ref)
            println("  │ $(lpad(n_rule, 3))-pt GL  │  $(lpad(round(dx, digits=1), 11)) │  $(lpad(round(dw, digits=1), 12))  │")

            @testset "$(n_rule)-point GL from Legendre($(N_basis))" begin
                @test length(w) == n_rule
                @test length(x) == n_rule
                #@info "$(n_rule)-pt GL from Legendre: nodes ≈ $(round(dx, digits=1)) digits, weights ≈ $(round(dw, digits=1)) digits"
                silent_check(dx > 14, "$(n_rule)-pt Legendre: node digits < 14")
                silent_check(dw > 14, "$(n_rule)-pt Legendre: weight digits < 14")
            end
        end
        println("  └────────────┴───────────────┴────────────────┘")
    end

    # -----------------------------------------------------------------------
    # 3. GL rule from ChebyshevT basis (automatic moments)
    # -----------------------------------------------------------------------
    @testset verbose=true "GL rule from ChebyshevT basis" begin
        println("\n  GL rules from ChebyshevT basis (automatic moments):")
        println("  ┌────────────┬───────────────┬────────────────┐")
        println("  │    rule    │  node digits  │ weight digits   │")
        println("  ├────────────┼───────────────┼────────────────┤")

        for n_rule in [3, 5, 7]
            N_basis = 2 * n_rule
            a, b = BigFloat(-1), BigFloat(1)
            basis = ChebyshevT(N_basis) → (a..b)

            w, x = compute_gauss_rule(basis)
            x_ref, w_ref = reference_gl(n_rule, BigFloat)

            dx = digits_of_agreement_vec(x, x_ref)
            dw = digits_of_agreement_vec(w, w_ref)
            println("  │ $(lpad(n_rule, 3))-pt GL  │  $(lpad(round(dx, digits=1), 11)) │  $(lpad(round(dw, digits=1), 12))  │")

            @testset "$(n_rule)-point GL from ChebyshevT($(N_basis))" begin
                @test length(w) == n_rule
                @test length(x) == n_rule
                #@info "$(n_rule)-pt GL from ChebyshevT: nodes ≈ $(round(dx, digits=1)) digits, weights ≈ $(round(dw, digits=1)) digits"
                silent_check(dx > 14, "$(n_rule)-pt ChebyshevT: node digits < 14")
                silent_check(dw > 14, "$(n_rule)-pt ChebyshevT: weight digits < 14")
            end
        end
        println("  └────────────┴───────────────┴────────────────┘")
    end

    # -----------------------------------------------------------------------
    # 4. GL rule from orthogonalized monomials (exact moments)
    # -----------------------------------------------------------------------
    @testset verbose=true "GL rule from orthogonalized monomials" begin
        println("\n  GL rules from orthogonalized monomials (exact moments):")
        println("  ┌────────────┬───────────────┬────────────────┐")
        println("  │    rule    │  node digits  │ weight digits   │")
        println("  ├────────────┼───────────────┼────────────────┤")

        for n_rule in [3, 5]
            N_basis = 2 * n_rule
            a, b = BigFloat(-1), BigFloat(1)

            mono_basis = bigfloat_monomial_basis(N_basis, a, b)
            exact_mom = exact_monomial_moments(N_basis, a, b)

            orth_basis, T_mat = orthogonalize_basis(mono_basis)
            orth_moments = T_mat * exact_mom

            w, x = compute_gauss_rule(orth_basis, orth_moments)
            x_ref, w_ref = reference_gl(n_rule, BigFloat)

            dx = digits_of_agreement_vec(x, x_ref)
            dw = digits_of_agreement_vec(w, w_ref)
            println("  │ $(lpad(n_rule, 3))-pt GL  │  $(lpad(round(dx, digits=1), 11)) │  $(lpad(round(dw, digits=1), 12))  │")

            @testset "$(n_rule)-point GL from orth. monomials ($(N_basis))" begin
                @test length(w) == n_rule
                @test length(x) == n_rule
                #@info "$(n_rule)-pt GL from orth. monomials: nodes ≈ $(round(dx, digits=1)) digits, weights ≈ $(round(dw, digits=1)) digits"
                @test dx > NEWTON_DIGITS - 15
                @test dw > NEWTON_DIGITS - 15
            end
        end
        println("  └────────────┴───────────────┴────────────────┘")
    end

    # -----------------------------------------------------------------------
    # 4.1 GL rule from orthogonalized monomials (automatic moments)
    # -----------------------------------------------------------------------
    @testset verbose=true "GL rule from orthogonalized monomials" begin
        println("\n  GL rules from orthogonalized monomials (automatic moments):")
        println("  ┌────────────┬───────────────┬────────────────┐")
        println("  │    rule    │  node digits  │ weight digits   │")
        println("  ├────────────┼───────────────┼────────────────┤")

        for n_rule in [3, 5]
            N_basis = 2 * n_rule
            a, b = BigFloat(-1), BigFloat(1)

            mono_basis = bigfloat_monomial_basis(N_basis, a, b)

            orth_basis, T_mat = orthogonalize_basis(mono_basis)

            w, x = compute_gauss_rule(orth_basis)
            x_ref, w_ref = reference_gl(n_rule, BigFloat)

            dx = digits_of_agreement_vec(x, x_ref)
            dw = digits_of_agreement_vec(w, w_ref)
            println("  │ $(lpad(n_rule, 3))-pt GL  │  $(lpad(round(dx, digits=1), 11)) │  $(lpad(round(dw, digits=1), 12))  │")

            @testset "$(n_rule)-point GL from orth. monomials ($(N_basis))" begin
                @test length(w) == n_rule
                @test length(x) == n_rule
                #@info "$(n_rule)-pt GL from orth. monomials: nodes ≈ $(round(dx, digits=1)) digits, weights ≈ $(round(dw, digits=1)) digits"
                @test dx > NEWTON_DIGITS - 15
                @test dw > NEWTON_DIGITS - 15
            end
        end
        println("  └────────────┴───────────────┴────────────────┘")
    end

    # -----------------------------------------------------------------------
    # 5. GL rule on [0, 1] — mapped interval
    # -----------------------------------------------------------------------
    @testset verbose=true "GL rule on [0, 1]" begin
        n_rule = 5
        N_basis = 2 * n_rule
        a, b = BigFloat(0), BigFloat(1)
        x_ref, w_ref = reference_gl_mapped(n_rule, a, b)

        println("\n  5-pt GL on [0,1] from different bases:")
        println("  ┌──────────────────────────┬───────────────┬────────────────┐")
        println("  │         basis            │  node digits  │ weight digits   │")
        println("  ├──────────────────────────┼───────────────┼────────────────┤")

        @testset "Legendre basis mapped to [0, 1]" begin
            basis = Legendre(N_basis) → (a..b)
            w, x = compute_gauss_rule(basis)
            dx = digits_of_agreement_vec(x, x_ref)
            dw = digits_of_agreement_vec(w, w_ref)
            println("  │ $(rpad("Legendre(auto mom.)", 24)) │  $(lpad(round(dx, digits=1), 11)) │  $(lpad(round(dw, digits=1), 12))  │")
            #@info "Legendre on [0,1]: nodes ≈ $(round(dx, digits=1)) digits, weights ≈ $(round(dw, digits=1)) digits"
            silent_check(dx > 14, "Legendre [0,1]: node digits < 14")
            silent_check(dw > 14, "Legendre [0,1]: weight digits < 14")
        end

        @testset "ChebyshevT basis mapped to [0, 1]" begin
            basis = ChebyshevT(N_basis) → (a..b)
            w, x = compute_gauss_rule(basis)
            dx = digits_of_agreement_vec(x, x_ref)
            dw = digits_of_agreement_vec(w, w_ref)
            println("  │ $(rpad("ChebyshevT(auto mom.)", 24)) │  $(lpad(round(dx, digits=1), 11)) │  $(lpad(round(dw, digits=1), 12))  │")
            #@info "ChebyshevT on [0,1]: nodes ≈ $(round(dx, digits=1)) digits, weights ≈ $(round(dw, digits=1)) digits"
            silent_check(dx > 14, "ChebyshevT [0,1]: node digits < 14")
            silent_check(dw > 14, "ChebyshevT [0,1]: weight digits < 14")
        end

        @testset "Orthogonalized monomials on [0, 1]" begin
            mono_basis = bigfloat_monomial_basis(N_basis, a, b)
            exact_mom = exact_monomial_moments(N_basis, a, b)

            orth_basis, T_mat = orthogonalize_basis(mono_basis)
            orth_moments = T_mat * exact_mom

            w, x = compute_gauss_rule(orth_basis, orth_moments)
            dx = digits_of_agreement_vec(x, x_ref)
            dw = digits_of_agreement_vec(w, w_ref)
            println("  │ $(rpad("Orth. mono.(exact mom.)", 24)) │  $(lpad(round(dx, digits=1), 11)) │  $(lpad(round(dw, digits=1), 12))  │")
            #@info "Orth. monomials on [0,1]: nodes ≈ $(round(dx, digits=1)) digits, weights ≈ $(round(dw, digits=1)) digits"
            @test dx > NEWTON_DIGITS - 15
            @test dw > NEWTON_DIGITS - 15
        end

        println("  └──────────────────────────┴───────────────┴────────────────┘")
    end

    # -----------------------------------------------------------------------
    # 6. Exactness verification: the rule integrates polynomials exactly
    # -----------------------------------------------------------------------
    @testset verbose=true "Quadrature exactness on polynomials" begin
        n_rule = 5
        N_basis = 2 * n_rule
        a, b = BigFloat(-1), BigFloat(1)

        # Test with both automatic-moment and exact-moment rules
        cheb_basis = ChebyshevT(N_basis) → (a..b)
        w_auto, x_auto = compute_gauss_rule(cheb_basis)

        mono_basis = bigfloat_monomial_basis(N_basis, a, b)
        exact_mom = exact_monomial_moments(N_basis, a, b)
        orth_basis, T_mat = orthogonalize_basis(mono_basis)
        w_exact, x_exact = compute_gauss_rule(orth_basis, T_mat * exact_mom)

        max_exact_degree = 2 * n_rule - 1

        println("\n  Quadrature exactness (∫ x^k dx on [-1,1], 5-pt rule):")
        println("  ┌───────┬───────────────────────┬───────────────────────┐")
        println("  │  deg  │  ChebyshevT(auto) dig │  Orth.mono(exact) dig │")
        println("  ├───────┼───────────────────────┼───────────────────────┤")

        for deg in 0:max_exact_degree
            exact_integral = exact_monomial_moments(deg + 1, a, b)[end]
            quad_auto  = sum(w_auto[k]  * x_auto[k]^deg  for k in 1:n_rule)
            quad_exact = sum(w_exact[k] * x_exact[k]^deg for k in 1:n_rule)
            d_auto  = digits_of_agreement(quad_auto,  exact_integral)
            d_exact = digits_of_agreement(quad_exact, exact_integral)
            println("  │  $(lpad(deg, 3)) │  $(lpad(round(d_auto, digits=1), 19))  │  $(lpad(round(d_exact, digits=1), 19))  │")

            @testset "degree $deg" begin
                silent_check(d_auto > 14, "Polynomial exactness degree $deg: ChebyshevT(auto) digits < 14")
                @test d_exact > NEWTON_DIGITS - 15
            end
        end
        println("  └───────┴───────────────────────┴───────────────────────┘")
    end

    # -----------------------------------------------------------------------
    # 7. Consistency: all three bases give the same rule
    # -----------------------------------------------------------------------
    @testset verbose=true "Cross-basis consistency on [0, 1]" begin
        n_rule = 5
        N_basis = 2 * n_rule
        a, b = BigFloat(0), BigFloat(1)

        basis_leg = Legendre(N_basis) → (a..b)
        w_leg, x_leg = compute_gauss_rule(basis_leg)

        basis_cheb = ChebyshevT(N_basis) → (a..b)
        w_cheb, x_cheb = compute_gauss_rule(basis_cheb)

        mono_basis = bigfloat_monomial_basis(N_basis, a, b)
        exact_mom = exact_monomial_moments(N_basis, a, b)
        orth_basis, T_mat = orthogonalize_basis(mono_basis)
        orth_moments = T_mat * exact_mom
        w_orth, x_orth = compute_gauss_rule(orth_basis, orth_moments)

        d_leg_cheb_x = digits_of_agreement_vec(x_leg, x_cheb)
        d_leg_cheb_w = digits_of_agreement_vec(w_leg, w_cheb)
        d_leg_orth_x = digits_of_agreement_vec(x_leg, x_orth)
        d_leg_orth_w = digits_of_agreement_vec(w_leg, w_orth)
        d_cheb_orth_x = digits_of_agreement_vec(x_cheb, x_orth)
        d_cheb_orth_w = digits_of_agreement_vec(w_cheb, w_orth)

        println("\n  Cross-basis consistency (5-pt GL on [0,1]):")
        println("  ┌────────────────────────────┬───────────────┬────────────────┐")
        println("  │         comparison          │  node digits  │ weight digits   │")
        println("  ├────────────────────────────┼───────────────┼────────────────┤")
        println("  │ $(rpad("Legendre vs ChebyshevT", 26)) │  $(lpad(round(d_leg_cheb_x, digits=1), 11)) │  $(lpad(round(d_leg_cheb_w, digits=1), 12))  │")
        println("  │ $(rpad("Legendre vs Orth.mono", 26)) │  $(lpad(round(d_leg_orth_x, digits=1), 11)) │  $(lpad(round(d_leg_orth_w, digits=1), 12))  │")
        println("  │ $(rpad("ChebyshevT vs Orth.mono", 26)) │  $(lpad(round(d_cheb_orth_x, digits=1), 11)) │  $(lpad(round(d_cheb_orth_w, digits=1), 12))  │")
        println("  └────────────────────────────┴───────────────┴────────────────┘")

        # Legendre and ChebyshevT use the same moment backend, so they
        # should agree to whatever precision BasisFunctions achieves
        silent_check(d_leg_cheb_x > 14, "Cross-basis Legendre vs ChebyshevT: node digits < 14")
        silent_check(d_leg_cheb_w > 14, "Cross-basis Legendre vs ChebyshevT: weight digits < 14")
        # Orth. monomials with exact moments might differ from the auto-moment
        # bases by however many digits the auto moments are off
        silent_check(d_leg_orth_x > 14, "Cross-basis Legendre vs Orth.mono: node digits < 14")
        silent_check(d_leg_orth_w > 14, "Cross-basis Legendre vs Orth.mono: weight digits < 14")
    end

    # -----------------------------------------------------------------------
    # 8. Summary: digits achieved by each pipeline
    # -----------------------------------------------------------------------
    @testset verbose=true "Summary: digits achieved per pipeline" begin
        println("\n" * "="^72)
        println("  SUMMARY: digits of agreement with reference GL rule")
        println("="^72)

        for (interval_label, a, b) in [("[-1,1]", BigFloat(-1), BigFloat(1)),
                                        ("[0,1]",  BigFloat(0),  BigFloat(1))]
            println("\n  Interval: $interval_label")
            println("  ┌──────────────────────────────────┬───────────────┬────────────────┐")
            println("  │           pipeline               │  node digits  │ weight digits   │")
            println("  ├──────────────────────────────────┼───────────────┼────────────────┤")

            for n_rule in [3, 5, 7]
                N_basis = 2 * n_rule
                x_ref, w_ref = (a == -1 && b == 1) ?
                    reference_gl(n_rule, BigFloat) :
                    reference_gl_mapped(n_rule, a, b)

                configs = [
                    ("Leg($(N_basis)) auto mom.",
                     () -> compute_gauss_rule(Legendre(N_basis) → (a..b))),
                    ("Cheb($(N_basis)) auto mom.",
                     () -> compute_gauss_rule(ChebyshevT(N_basis) → (a..b))),
                    ("Mono($(N_basis)) orth+exact",
                     () -> begin
                         mono = bigfloat_monomial_basis(N_basis, a, b)
                         mom = exact_monomial_moments(N_basis, a, b)
                         ob, T = orthogonalize_basis(mono)
                         compute_gauss_rule(ob, T * mom)
                     end),
                ]

                for (label, compute) in configs
                    w, x = compute()
                    dx = digits_of_agreement_vec(x, x_ref)
                    dw = digits_of_agreement_vec(w, w_ref)
                    tag = "$(n_rule)-pt $label"
                    println("  │ $(rpad(tag, 32)) │  $(lpad(round(dx, digits=1), 11)) │  $(lpad(round(dw, digits=1), 12))  │")
                end
            end
            println("  └──────────────────────────────────┴───────────────┴────────────────┘")
        end

        # This testset primarily drives printout tables.
        @test true
    end
    if SILENT_FAILURE_COUNT[] > 0
        println(stderr, "\nSilent threshold check failures ($(SILENT_FAILURE_COUNT[])):")
        for label in SILENT_FAILURE_LABELS
            println(stderr, "  - $label")
        end
        throw(SilentTableFailure())
    end
end
catch err
    is_testset_failure = err isa Test.TestSetException ||
                         (err isa LoadError && err.error isa Test.TestSetException)
    is_silent_failure = err isa SilentTableFailure ||
                        (err isa LoadError && err.error isa SilentTableFailure)
    if RUNNING_BIGFLOAT_PRECISION_AS_SCRIPT && (is_testset_failure || is_silent_failure)
        exit(1)
    else
        rethrow()
    end
end
