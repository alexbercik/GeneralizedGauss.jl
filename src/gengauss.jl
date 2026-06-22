"""
    _gengauss_debug_enabled() -> Bool

When `true`, internal trace `println`s in this file are shown. Controlled by
`ENV["GENGAUSS_DEBUG"]`: any of `"1"`, `"true"`, `"yes"`, `"y"` (case-insensitive)
enables tracing; anything else (including unset) disables it. Read at call time,
so you can toggle from the REPL without reloading the package.
"""
function _gengauss_debug_enabled()
    v = strip(lowercase(get(ENV, "GENGAUSS_DEBUG", "")))
    v in ("1", "true", "yes", "y")
end

function _gengauss_debug_println(args...; kwargs...)
    _gengauss_debug_enabled() && println(args...; kwargs...)
    return nothing
end

eval_moment_error(w, x, u, c) = apply_quad(w, x, u) - c

struct GaussRuleConfig
    principal::Symbol
    add_endpoint::Symbol
end

function GaussRuleConfig(; principal::Symbol = :lower, add_endpoint::Symbol = :right)
    @assert principal in (:lower, :upper)
    @assert add_endpoint in (:left, :right)
    GaussRuleConfig(principal, add_endpoint)
end

default_add_endpoint(principal::Symbol) = principal == :upper ? :right : :left

function resolve_add_endpoint(principal::Symbol, add_endpoint::Union{Nothing,Symbol})
    isnothing(add_endpoint) ? default_add_endpoint(principal) : add_endpoint
end

# Helper function to get direction from add_endpoint
get_direction(config::GaussRuleConfig) = config.add_endpoint == :right ? :right_to_left : :left_to_right

endpoint_value(dict, config::GaussRuleConfig) =
    config.add_endpoint == :right ? supportright(dict) : supportleft(dict)

pivot_value(x, config::GaussRuleConfig) =
    config.add_endpoint == :right ? x[1] : x[end]


# Returns the principal representation of c^{2k+1} (length(dict) = 2k+1, n=2k even)
# whose root structure includes the anchored endpoint:
#  - add_endpoint=:right → UpperPrincipalEven (right endpoint fixed) = right-Radau
#  - add_endpoint=:left  → LowerPrincipalEven (left  endpoint fixed) = left-Radau
# Despite the name "upper_principal_rule", for :left configurations this returns
# what is technically the *lower* principal of c^{2k+1}; the loop calls this from
# its `:upper :principal` arm purely as a structural label, not as the actual
# K-S "upper principal" classification.
function upper_principal_rule(dict, moments, config::GaussRuleConfig)
    config.add_endpoint == :right ?
        UpperPrincipalEven(dict, moments) :
        LowerPrincipalEven(dict, moments)
end

default_threshold(dict::Dictionary) = default_threshold(codomaintype(dict))
default_threshold(::Type{T}) where {T <: AbstractFloat} = sqrt(eps(T))/100
default_threshold(::Type{BigFloat}) = BigFloat(10)^(-20)

function _resolve_solver_tolerance(::Type{T}, solver_tolerance;
        tolerance_floor=zero(T)) where {T}
    # By default use one decimal digit above machine epsilon. This avoids
    # treating harmless final-roundoff residuals as nonlinear solve failures.
    base_tolerance = solver_tolerance === nothing ?
        10 * eps(T) :
        solver_tolerance
    isfinite(base_tolerance) && base_tolerance > 0 ||
        throw(ArgumentError("solver_tolerance must be finite and positive"))
    strict_tolerance = max(T(base_tolerance), T(tolerance_floor))
    isfinite(strict_tolerance) ||
        throw(ArgumentError("solver_tolerance must be representable as a finite value in $(T)"))
    strict_tolerance
end

const DEFAULT_INTERMEDIATE_MIN_TOLERANCE = 1e-8
const DEFAULT_INTERMEDIATE_MAX_TOLERANCE = 1e-3

function _default_intermediate_tolerance(::Type{T}, strict_tolerance::T) where {T}
    sqrt_term = T(10) * sqrt(strict_tolerance)
    absolute_floor = T(DEFAULT_INTERMEDIATE_MIN_TOLERANCE)
    hybrid = max(sqrt_term, absolute_floor)
    min(hybrid, T(DEFAULT_INTERMEDIATE_MAX_TOLERANCE))
end

# Intermediate continuation solves only need to provide useful seeds and
# brackets. Keep the strict tolerance separately so diagnostics and lost-digit
# fallback policies remain tied to the accuracy requested for final rules.
function _resolve_solver_tolerances(::Type{T}, solver_tolerance=nothing,
        intermediate_tolerance=nothing;
        tolerance_floor=zero(T)) where {T}
    strict_tolerance = _resolve_solver_tolerance(T, solver_tolerance;
        tolerance_floor)
    if intermediate_tolerance === :strict
        return strict_tolerance, strict_tolerance
    elseif intermediate_tolerance === nothing
        default_intermediate =
            _default_intermediate_tolerance(T, strict_tolerance)
        active_tolerance = max(strict_tolerance, default_intermediate)
        return strict_tolerance, active_tolerance
    elseif intermediate_tolerance isa Symbol
        throw(ArgumentError("intermediate_tolerance must be a positive number, " *
            "`:strict`, or `nothing`; got $(intermediate_tolerance)"))
    end

    isfinite(intermediate_tolerance) && intermediate_tolerance > 0 ||
        throw(ArgumentError("intermediate_tolerance must be finite and positive"))
    active_tolerance = max(strict_tolerance, T(intermediate_tolerance))
    isfinite(active_tolerance) ||
        throw(ArgumentError("intermediate_tolerance must be representable as a finite value in $(T)"))
    strict_tolerance, active_tolerance
end

function _continuation_solver_policy_type(;
        solver_tolerance=nothing, intermediate_tolerance=nothing)
    if solver_tolerance !== nothing
        return typeof(solver_tolerance)
    elseif intermediate_tolerance isa Real
        return typeof(intermediate_tolerance)
    else
        return Float64
    end
end

function _continuation_solver_hint(;
        solver_tolerance=nothing,
        intermediate_tolerance=nothing,
        policy_type::Type=Float64)
    intermediate_tolerance === :strict && return ""
    strict, active = _resolve_solver_tolerances(policy_type,
        solver_tolerance, intermediate_tolerance)
    active <= strict && return ""
    " If continuation used a coarse intermediate tolerance, try lowering " *
    "intermediate_tolerance or pass intermediate_tolerance=:strict " *
    "(active=$(_fmt_sci(active)), strict=$(_fmt_sci(strict)))."
end

function _continuation_solver_hint(options)
    solver_tolerance = get(options, :solver_tolerance, nothing)
    intermediate_tolerance = get(options, :intermediate_tolerance, nothing)
    policy_type = _continuation_solver_policy_type(;
        solver_tolerance, intermediate_tolerance)
    _continuation_solver_hint(;
        solver_tolerance, intermediate_tolerance, policy_type)
end

# Default extra decimal digits accepted above the nonlinear solver tolerance.
# Use the `lost_digits=...` keyword to override this for a specific call.
const DEFAULT_LOST_DIGITS = 2

struct RepresentationStep
    branch::Symbol
    stage::Symbol
    fixed_endpoint::Union{Symbol,Nothing}
    k::Int
end

function default_representation_steps(principal::Symbol, add_endpoint::Symbol, even_basis_length::Bool)
    # Note: the step list depends only on `add_endpoint`. The choice of where to
    # exit the loop (Step 2 of the last iter for Radau / Step 4 for Gauss-Legendre /
    # post-loop for Lobatto) is made in compute_gauss_rules using `principal` and
    # the parity of length(dict). Both parities use the same step list — for odd
    # length(dict) the loop is short-circuited at Step 2 of iter k = l via
    # stop_target_len before Step 3/4 try to access moments past the basis.
    if add_endpoint == :right
        # Anchor at b. Each iteration:
        #   Step 1: append b (weight 0) to the previous Gauss rule, trace the
        #           K-canonical of c^{2k} as ξ slides leftward.
        #   Step 2: pop into UP of c^{2k+1} = right-Radau.
        #   Step 3: trace the J-canonical of c^{2k+1} as ξ continues leftward.
        #   Step 4: pop into LP of c^{2k+2} = (k+1)-point Gauss-Legendre.
        steps = RepresentationStep[
        RepresentationStep(:upper, :canonical, :right, 1),
        RepresentationStep(:upper, :principal, :right, 1),
        RepresentationStep(:lower, :canonical, nothing, 2),
        RepresentationStep(:lower, :principal, nothing, 2),
        ]
    elseif add_endpoint == :left
        # Mirror image of the :right step list, sweeping left-to-right.
        # Step 2 here is the LP of c^{2k+1} = left-Radau.
        steps = RepresentationStep[
        RepresentationStep(:lower, :canonical, :left, 1),
        RepresentationStep(:lower, :principal, :left, 1),
        RepresentationStep(:lower, :canonical, nothing, 2),
        RepresentationStep(:lower, :principal, nothing, 2),
        ]
    else
        error("default_representation_steps: unknown add_endpoint=$(add_endpoint) (expected :left or :right)")
    end
    steps
end

function _solver_diagnostic(rule, newton_x, strict_tolerance,
        active_tolerance=strict_tolerance)
    Fx = residual(rule, newton_x)
    residual_norm = maximum(abs, Fx)
    (; residual_norm, tolerance=strict_tolerance, active_tolerance,
        ratio=residual_norm / strict_tolerance)
end

function _solver_exception_diagnostic(e)
    (; residual_norm=nothing, tolerance=nothing, ratio=nothing, error=e)
end

function _format_solver_diagnostic(diag)
    diag === nothing && return "diagnostic unavailable"
    if :error in keys(diag)
        return "diagnostic unavailable: solver threw $(typeof(diag.error))"
    end
    if :active_tolerance in keys(diag) && diag.active_tolerance != diag.tolerance
        return "residual=$(_fmt_sci(diag.residual_norm)), strict_ftol=$(_fmt_sci(diag.tolerance)), active_ftol=$(_fmt_sci(diag.active_tolerance)), residual/strict_ftol=$(_fmt_sci(diag.ratio))"
    end
    "residual=$(_fmt_sci(diag.residual_norm)), ftol=$(_fmt_sci(diag.tolerance)), residual/ftol=$(_fmt_sci(diag.ratio))"
end

orthogonalization_digits_lost(::Dictionary) = 0.0
orthogonalization_digits_lost(dict::GenericFunctionSet) =
    dict.orthogonalization_digits_lost

function _orthogonalization_lost_digits_floor(dict::Dictionary)
    lost = orthogonalization_digits_lost(dict)
    isfinite(lost) || return 0
    ceil(Int, max(0.0, lost/2))
end

function _lost_digits_acceptance_limit(diag, lost_digits)
    lost_digits < 0 && throw(ArgumentError("lost_digits must be nonnegative"))
    residual = diag.residual_norm
    strict = diag.tolerance
    active = (:active_tolerance in propertynames(diag) &&
        diag.active_tolerance !== nothing) ?
        diag.active_tolerance : strict
    T = typeof(residual)
    # Lost digits relax only the strict target. When continuation uses a looser
    # active tolerance, do not accept residuals above min(active, strict)*10^LD;
    # with active >= strict this reduces to the strict slack band.
    min(T(strict), T(active)) * T(10)^T(lost_digits)
end

function _lost_digits_accepts(diag, lost_digits)
    diag === nothing && return false
    :error in keys(diag) && return false
    lost_digits < 0 && return false
    residual = diag.residual_norm
    isfinite(residual) || return false
    residual <= _lost_digits_acceptance_limit(diag, lost_digits)
end

function _warn_lost_digits_acceptance(label, location, diag, lost_digits,
        option_name)
    where = location === nothing ? "" : " for $(location)"
    println("WARNING: accepted $(label) nonlinear solve$(where) above ftol " *
            "[$(_format_solver_diagnostic(diag)), $(option_name)=$(lost_digits)]; " *
            "some accuracy may be lost.")
end

function _require_principal_convergence(label, location, converged, diag,
        lost_digits; verbose=false,
        solver_tolerance=nothing, intermediate_tolerance=nothing,
        policy_type::Type=Float64)
    converged && return true
    if _lost_digits_accepts(diag, lost_digits)
        verbose && _warn_lost_digits_acceptance(label, location, diag,
            lost_digits, "lost_digits")
        return true
    end

    where = location === nothing ? "" : " for $(location)"
    println("ERROR: $(label) nonlinear solve$(where) failed " *
            "[$(_format_solver_diagnostic(diag)), " *
            "lost_digits=$(lost_digits)]; stopping.")
    hint = _continuation_solver_hint(;
        solver_tolerance, intermediate_tolerance, policy_type)
    error("$(label) nonlinear solve failed. Consider increasing lost_digits if " *
          "residual is still acceptable." * hint)
end

function _has_first_derivatives(dict::Dictionary)
    x_probe = (supportleft(dict) + supportright(dict)) / 2
    all(i -> maybe_funeval_deriv(dict, i, x_probe, 1) !== nothing,
        1:length(dict))
end

function _resolve_gauss_solver_mode(dict::Dictionary, differentiable::Bool)
    differentiable || return :mads
    if !_has_first_derivatives(dict)
        @warn("Analytic first derivatives are missing for at least one basis " *
              "function. Missing node derivatives will be approximated with " *
              "support-aware finite differences.")
        return :finite_diff_nlsolve
    end
    :analytic_nlsolve
end

function solve_system(rule, w0, x0; verbose=false,
        solver_mode::Symbol=:analytic_nlsolve, mads_dx=nothing,
        mads_bracket=nothing, solver_tolerance=nothing,
        intermediate_tolerance=nothing,
        lost_digits::Real=DEFAULT_LOST_DIGITS, options...)
    x_init = quad_to_newton(rule, w0, x0)
    F!(Fx, x) = residual!(Fx, rule, x)

    strict_tolerance, active_tolerance =
        _resolve_solver_tolerances(eltype(x_init), solver_tolerance,
            intermediate_tolerance)

    if solver_mode == :mads
        return _solve_system_mads(rule, w0, x0; verbose, dx=mads_dx,
            bracket=mads_bracket, intermediate_tolerance,
            solver_tolerance)
    elseif solver_mode == :analytic_nlsolve
        eval_deriv = funeval_deriv
    elseif solver_mode == :finite_diff_nlsolve
        eval_deriv = funeval_deriv_or_finite_diff
    else
        error("Unknown nonlinear solver mode $(solver_mode).")
    end

    J!(Jx, x) = jacobian!(Jx, rule, x, eval_deriv)
    r = nlsolve(F!, J!, x_init; ftol=active_tolerance, options...)
    w, x = newton_to_quad(rule, r.zero)
    converged(r), w, x,
        _solver_diagnostic(rule, r.zero, strict_tolerance, active_tolerance)
end

function supportleft(dict)
    t = leftendpoint(support(dict))
    isinf(t) ? -one(t)*10 : t
end
supportright(dict) = rightendpoint(support(dict))

function compute_one_point_rule(dict, moments; verbose=false,
        config::GaussRuleConfig=GaussRuleConfig(),
        lost_digits::Real=DEFAULT_LOST_DIGITS, differentiable::Bool=true,
        solver_tolerance=nothing, intermediate_tolerance=nothing, options...)
    @assert length(dict) == 2
    @assert length(moments) == 2

    a = supportleft(dict)
    b = supportright(dict)
    c1, c2 = moments[1], moments[2]
    if c1 == zero(c1)
        error("Degenerate one-point rule: first moment is zero, so c₂/c₁ is undefined.")
    end

    x0 = (a + b) / 2
    f = x -> c1 * funeval(dict, 2, x) - c2 * funeval(dict, 1, x)
    strict_base_tol, active_base_tol =
        _resolve_solver_tolerances(typeof(x0), solver_tolerance,
            intermediate_tolerance)
    x_scale = max(one(x0), abs(a), abs(b))
    f_scale = max(one(x0), maximum(abs, moments))
    strict_f_tol = strict_base_tol * f_scale
    active_x_tol = active_base_tol * x_scale
    active_f_tol = active_base_tol * f_scale

    if differentiable
        df = x -> c1 * funeval_deriv_or_finite_diff(dict, 2, x) -
                  c2 * funeval_deriv_or_finite_diff(dict, 1, x)
        converged, x, fx = _scalar_newton_root(f, df, a, b, x0;
            x_tol=active_x_tol, f_tol=active_f_tol, verbose, options...)
        if !converged
            verbose && println("WARNING: one-point scalar safeguarded Newton did not " *
                "converge; falling back to Brent root finding.")
            converged, x, fx = _brent_root_on_interval(f, a, b, x;
                x_tol=active_x_tol, f_tol=active_f_tol, options...)
        end
    else
        converged, x, fx = _brent_root_on_interval(f, a, b, x0;
            x_tol=active_x_tol, f_tol=active_f_tol, options...)
    end

    if !converged
        diag = (; residual_norm=abs(fx), tolerance=strict_f_tol,
                active_tolerance=active_f_tol, ratio=abs(fx) / strict_f_tol)
        _require_principal_convergence("one-point scalar", x,
            false, diag, lost_digits; verbose,
            solver_tolerance, intermediate_tolerance,
            policy_type=typeof(x0))
    end

    f1x = funeval(dict, 1, x)
    if f1x == zero(f1x)
        error("Degenerate one-point rule: f₁(x) is zero at x=$(x), so the weight is undefined.")
    end
    w = moments[1] / f1x
    [w], [x]
end

function compute_two_point_rule(dict, moments,
        a = supportleft(dict), b = supportright(dict);
        solver_tolerance=nothing, options...)
    @assert length(dict) == 2
    @assert length(moments) == 2

    x = [a, b]

    u0a = funeval(dict, 1, a)
    u0b = funeval(dict, 1, b)
    u1a = funeval(dict, 2, a)
    u1b = funeval(dict, 2, b)

    denom = u0a * u1b - u0b * u1a

    tol = _resolve_solver_tolerance(typeof(denom), solver_tolerance)
    if abs(denom) < tol
        error("Degenerate system: u₀(a)u₁(b) - u₀(b)u₁(a) ≈ 0.\n" *
              "The functions are linearly dependent on {a,b}, so a unique 2-point rule does not exist.")
    end

    wa = (moments[1] * u1b - moments[2] * u0b) / denom
    wb = (moments[2] * u0a - moments[1] * u1a) / denom
    w = [wa, wb]
    w, x
end

function _quad_rule_diagnostic(dict, moments, w, x;
        solver_tolerance=nothing, options...)
    residual_norm = zero(abs(moments[1]))
    for j in 1:length(dict)
        rj = -moments[j]
        for i in eachindex(w)
            rj += w[i] * funeval(dict, j, x[i])
        end
        residual_norm = max(residual_norm, abs(rj))
    end
    tol = _resolve_solver_tolerance(typeof(residual_norm), solver_tolerance)
    (; residual_norm, tolerance=tol, ratio=residual_norm / tol)
end

function _compute_two_point_canonical_representation(dict, moments, a, b;
        options...)
    w, x = compute_two_point_rule(dict, moments, a, b; options...)
    true, w, x, _quad_rule_diagnostic(dict, moments, w, x; options...)
end


function compute_upper_canonical_representation(dict, moments, xi, w0, x0;
        verbose=false, config::GaussRuleConfig=GaussRuleConfig(),
        options...)
    @assert length(moments) == length(dict)

    _gengauss_debug_println("DEBUG: compute_upper_canonical_representation, even number of basis functions, (", length(dict), ")")

    if length(dict) == 2
        try
            return _compute_two_point_canonical_representation(
                dict, moments, xi, supportright(dict); options...)
        catch e
            if e isa InterruptException
                rethrow()
            end
            println("ERROR THROWN at $(xi) in computation of two-point upper canonical")
            @show e
            return false, w0, x0, _solver_exception_diagnostic(e)
        end
    end

    if iseven(length(dict))
        # even number of (n+1) basis functions (odd n)
        @assert length(w0) == (length(dict)>>1)+1
        fixed_idx = config.add_endpoint == :right ? [1,length(w0)] : [length(w0)-1,length(w0)]
        rule = CanonicalRepresentationOdd_K1(dict, xi, moments, fixed_idx)
    else
        error("TODO")
    end

    try
        solve_system(rule, w0, x0; verbose, options...)
    catch e
        if e isa InterruptException
            rethrow()
        end
        println("ERROR THROWN at $(xi) in computation of upper canonical")
        @show e
        false, w0, x0, _solver_exception_diagnostic(e)
    end
end


function estimate_upper_canonical_representation(dict, moments, a, b, w0, x0;
        verbose=false, config::GaussRuleConfig=GaussRuleConfig(),
        sweep_direction::Union{Symbol,Nothing}=nothing,
        max_adaptive_steps::Int=32, lost_digits::Real=DEFAULT_LOST_DIGITS,
        options...)
    @assert isodd(length(dict))
    @assert length(dict) == length(moments)
    l = (length(dict)-1) >> 1
    @assert length(w0) == l+1
    @assert length(x0) == l+1

    verbose && println("Estimating upper canonical representation, xi between $(a) and $(b)")

    digits = max(lost_digits, _orthogonalization_lost_digits_floor(dict))

    seed = _estimate_canonical_representation_by_sweep(
        compute_upper_canonical_representation, "Upper canonical",
        dict, moments, a, b, w0, x0;
        verbose, config, sweep_direction, max_adaptive_steps,
        lost_digits=digits, options...)
    if seed === nothing
        hint = _continuation_solver_hint(options)
        error("Upper canonical sweep failed to locate a next-moment " *
              "sign change within the adaptive sweep budget. Check the basis " *
              "or increase max_adaptive_steps/lost_digits if the " *
              "nonlinear residuals are acceptable." * hint)
    end
    seed
end

const _CANONICAL_SWEEP_INITIAL_SUBDIVISIONS = 9
const _MADS_CANONICAL_SWEEP_INITIAL_SUBDIVISIONS = 6
const _MAX_PRINCIPAL_RECOVERY_REFINEMENTS = 3

struct CanonicalSample{TXI,TF,TW,TX}
    xi::TXI
    F::TF
    w::TW
    x::TX
end

struct CanonicalBracketState{TXI,TF,TW,TX}
    left::CanonicalSample{TXI,TF,TW,TX}
    right::CanonicalSample{TXI,TF,TW,TX}
    best::CanonicalSample{TXI,TF,TW,TX}
    width::TXI
    xi_index::Int
end

function CanonicalBracketState(left::CanonicalSample{TXI,TF,TW,TX},
        right::CanonicalSample{TXI,TF,TW,TX};
        continuation_hint="") where {TXI,TF,TW,TX}
    # The canonical representation fixes one node exactly at xi. Infer its
    # index from the realized rules so branch-specific index formulas cannot
    # disagree with the representation sent to the nonlinear solver.
    left_xi_index = _canonical_xi_index(left; continuation_hint)
    right_xi_index = _canonical_xi_index(right; continuation_hint)
    left_xi_index == right_xi_index ||
        _canonical_refinement_error("canonical bracket changed the fixed xi " *
            "node index from $(left_xi_index) to $(right_xi_index).";
            continuation_hint)
    best = abs(left.F) <= abs(right.F) ? left : right
    CanonicalBracketState{TXI,TF,TW,TX}(
        left, right, best, right.xi - left.xi, left_xi_index)
end

function _canonical_sample(xi, w, x, dict, moments)
    CanonicalSample(xi,
        eval_moment_error(w, x, dict[end], moments[end]),
        w, x)
end

function _canonical_xi_index(sample::CanonicalSample; continuation_hint="")
    matches = findall(==(sample.xi), sample.x)
    length(matches) == 1 ||
        _canonical_refinement_error("expected exactly one node fixed at " *
            "xi=$(sample.xi), found $(length(matches)).";
            continuation_hint)
    only(matches)
end

function _canonical_bracket_state(first::CanonicalSample,
        second::CanonicalSample; continuation_hint="")
    iszero(first.F) && return CanonicalBracketState(first, first)
    iszero(second.F) && return CanonicalBracketState(second, second)

    left, right = first.xi <= second.xi ?
        (first, second) : (second, first)
    if left.xi == right.xi
        iszero(left.F) && iszero(right.F) ||
            _canonical_refinement_error("canonical bracket has zero width " *
                "without a zero residual.";
                continuation_hint)
    elseif same_sign(left.F, right.F)
        _canonical_refinement_error("canonical bracket endpoints do not " *
            "have opposite residual signs.";
            continuation_hint)
    end
    CanonicalBracketState(left, right; continuation_hint)
end

function _canonical_refinement_error(detail; continuation_hint="")
    error("Canonical bracket refinement failed: $(detail) This may indicate " *
          "conditioning problems or a basis that is not a CT-system." *
          continuation_hint)
end

function _canonical_secant_target(state::CanonicalBracketState;
        continuation_hint="")
    left, right = state.left, state.right
    state.width > zero(state.width) ||
        _canonical_refinement_error("cannot refine a zero-width bracket.";
            continuation_hint)

    midpoint = left.xi + state.width / 2
    ΔF = right.F - left.F
    target = iszero(ΔF) ? midpoint :
        left.xi - left.F * state.width / ΔF

    if !(left.xi < target < right.xi)
        target = midpoint
    end

    if !(left.xi < target < right.xi) || _canonical_x_stalled(left.xi, target)
        _canonical_refinement_error("secant interpolation and midpoint " *
            "fallback stalled inside bracket [$(left.xi), $(right.xi)].";
            continuation_hint)
    end
    target
end

function _update_canonical_bracket_after_refinement(
        state::CanonicalBracketState, refined::CanonicalSample;
        continuation_hint="")
    left, right = state.left, state.right
    old_width = state.width
    old_best = abs(state.best.F)

    slack = 100 * eps(float(left.F))
    residual_lo = min(left.F, right.F)
    residual_hi = max(left.F, right.F)
    residual_lo - slack <= refined.F <= residual_hi + slack ||
        _canonical_refinement_error("refined next-moment residual " *
            "$(refined.F) is inconsistent with the monotonic bracket " *
            "residuals [$(left.F), $(right.F)].";
            continuation_hint)

    updated = if iszero(refined.F)
        CanonicalBracketState(refined, refined)
    elseif same_sign(left.F, refined.F)
        CanonicalBracketState(refined, right)
    elseif same_sign(right.F, refined.F)
        CanonicalBracketState(left, refined)
    else
        _canonical_refinement_error("refined residual sign is inconsistent " *
            "with the saved sign-changing bracket.";
            continuation_hint)
    end

    new_width = updated.width
    new_width < old_width ||
        _canonical_refinement_error("updated bracket width did not shrink " *
            "from $(old_width) to $(new_width).";
            continuation_hint)

    new_best = abs(updated.best.F)
    new_best <= old_best + slack ||
        _canonical_refinement_error("best absolute residual worsened from " *
            "$(old_best) to $(new_best).";
            continuation_hint)
    updated
end

function _refine_canonical_bracket(compute_one, label, dict, moments,
        state::CanonicalBracketState;
        verbose=false, config::GaussRuleConfig=GaussRuleConfig(),
        max_adaptive_steps::Int=32, lost_digits::Real=DEFAULT_LOST_DIGITS,
        options...)
    continuation_hint = _continuation_solver_hint(options)
    target = _canonical_secant_target(state; continuation_hint)
    warm_seed = abs(target - state.left.xi) <= abs(state.right.xi - target) ?
        state.left : state.right
    accepted, xi, w, x, _ =
        _try_canonical_step(compute_one, label, dict[1:end-1],
            moments[1:end-1], warm_seed.xi, target, warm_seed.w, warm_seed.x;
            verbose, config, max_adaptive_steps, lost_digits,
            options...)
    accepted || return nothing

    refined = _canonical_sample(xi, w, x, dict, moments)
    _update_canonical_bracket_after_refinement(state, refined;
        continuation_hint)
end

function _accept_canonical_step(compute_one, label, dict, moments, xi, w_seed, x_seed;
        verbose=false, config::GaussRuleConfig=GaussRuleConfig(),
        lost_digits::Real=DEFAULT_LOST_DIGITS, mads_dx=nothing, options...)
    converged, w, x, diag =
        compute_one(dict, moments, xi, w_seed, x_seed;
            verbose, config, mads_dx, options...)
    if converged
        return true, w, x, diag
    end

    lost_digits_converged =
        _lost_digits_accepts(diag, lost_digits)
    if lost_digits_converged
        verbose && _warn_lost_digits_acceptance(label, xi, diag,
            lost_digits, "lost_digits")
        return true, w, x, diag
    end

    false, w, x, diag
end

function _try_canonical_step(compute_one, label, dict, moments, xi_prev, target,
        w_prev, x_prev;
        verbose=false, config::GaussRuleConfig=GaussRuleConfig(),
        max_adaptive_steps::Int=32, lost_digits::Real=DEFAULT_LOST_DIGITS,
        options...)
    trial = target
    last_diagnostic = nothing
    for adaptive_step in 0:max_adaptive_steps
        accepted, w, x, last_diagnostic =
            _accept_canonical_step(compute_one, label, dict, moments, trial,
                w_prev, x_prev; verbose, config, lost_digits,
                mads_dx=abs(trial - xi_prev),
                options...)
        accepted && return true, trial, w, x, last_diagnostic

        adaptive_step == max_adaptive_steps && break

        next_trial = xi_prev + (trial - xi_prev) / 2
        if _canonical_x_stalled(next_trial, xi_prev) ||
                _canonical_x_stalled(next_trial, trial)
            break
        end

        _gengauss_debug_println(label, ": retrying at ", next_trial,
            " after failed target ", trial,
            " from last converged xi ", xi_prev)
        trial = next_trial
    end

    verbose && println("$(label): not converged for $(target) " *
        "[$(_format_solver_diagnostic(last_diagnostic))]")
    false, trial, w_prev, x_prev, last_diagnostic
end

@inline function _canonical_x_stalled(xa, xb)
    δ = abs(xb - xa)
    ref = max(abs(xa), abs(xb), oneunit(xa))
    δ <= 2 * eps(ref)
end

function _canonical_step_target(xi, sweep_end, dx, direction::Symbol)
    if direction == :left_to_right
        remaining = sweep_end - xi
        step = dx >= remaining ? remaining / 2 : dx
        xi + step
    elseif direction == :right_to_left
        remaining = xi - sweep_end
        step = dx >= remaining ? remaining / 2 : dx
        xi - step
    else
        error("Unknown sweep direction $(direction). Use :left_to_right or :right_to_left.")
    end
end

function _update_canonical_trend(label, direction, xi_prev, xi_curr,
        F_prev, F_curr, trend; continuation_hint="")
    if F_curr == F_prev
        error("$(label): next-moment residual stopped changing during " *
              "$(direction) sweep at ξ=$(xi_curr). F_prev=$(F_prev), " *
              "F_curr=$(F_curr)." * continuation_hint)
    end

    step_trend = F_curr > F_prev ? :increasing : :decreasing
    if trend !== nothing && step_trend != trend
        error("$(label): next-moment residual is not monotonic during " *
              "$(direction) sweep between ξ=$(xi_prev) and ξ=$(xi_curr). " *
              "F_prev=$(F_prev), F_curr=$(F_curr)." * continuation_hint)
    end

    step_trend
end

function _estimate_canonical_representation_by_sweep(compute_one, label,
        dict, moments, a, b, w0, x0;
        verbose=false, config::GaussRuleConfig=GaussRuleConfig(),
        sweep_direction::Union{Symbol,Nothing}=nothing,
        max_adaptive_steps::Int=32, lost_digits::Real=DEFAULT_LOST_DIGITS,
        options...)
    dir = sweep_direction !== nothing ? sweep_direction : get_direction(config)
    sweep_start = dir == :left_to_right ? a : b
    sweep_end = dir == :left_to_right ? b : a
    interval = abs(sweep_end - sweep_start)
    # MADS is substantially more expensive per solve, so start with fewer
    # continuation points and let the existing adaptive midpoint fallback
    # reduce the step only when a larger derivative-free solve fails.
    subdivisions = get(options, :solver_mode, nothing) == :mads ?
        _MADS_CANONICAL_SWEEP_INITIAL_SUBDIVISIONS :
        _CANONICAL_SWEEP_INITIAL_SUBDIVISIONS
    dx = interval / subdivisions
    dict_inner = dict[1:end-1] # -- we want to be exact for all but the last basis function & moment
    moments_inner = moments[1:end-1] # -- the last moment will be used to determine the sign change

    _gengauss_debug_println(label, ": sweep_start=", sweep_start,
        ", sweep_end=", sweep_end, ", direction=", dir)
    _gengauss_debug_println("w0: ", w0)
    _gengauss_debug_println("x0: ", x0)

    xi_prev = sweep_start
    w_prev, x_prev = w0, x0
    # -- initial moment error at previous principal representation (sweep start)
    previous = _canonical_sample(xi_prev, w_prev, x_prev, dict, moments)
    F_prev = previous.F
    iszero(F_prev) &&
        return CanonicalBracketState(previous, previous)

    trend = nothing
    samples = 0
    continuation_hint = _continuation_solver_hint(options)
    while true
        target = _canonical_step_target(xi_prev, sweep_end, dx, dir)
        if _canonical_x_stalled(xi_prev, target)
            break
        end

        #-- Try a step. May return a smaller step than we wanted if
        #   it could not converge and required some refinement.
        accepted, xi_curr, w_curr, x_curr, _ =
            _try_canonical_step(compute_one, label, dict_inner,
                moments_inner, xi_prev, target, w_prev, x_prev;
                verbose, config, max_adaptive_steps,
                lost_digits, options...)
        accepted || return nothing
        samples += 1

        F_curr = eval_moment_error(w_curr, x_curr, dict[end], moments[end])
        trend = _update_canonical_trend(label, dir, xi_prev, xi_curr,
            F_prev, F_curr, trend; continuation_hint)
        current = CanonicalSample(xi_curr, F_curr, w_curr, x_curr)
        if !same_sign(F_prev, F_curr)
            return _canonical_bracket_state(previous, current;
                continuation_hint)
        end

        #-- If it required refinement, update the step size
        actual_step = abs(xi_curr - xi_prev)
        if actual_step < dx
            dx = actual_step
        end
        previous = current
        xi_prev, w_prev, x_prev, F_prev = xi_curr, w_curr, x_curr, F_curr
    end

    verbose && println("$(label): no sign change after $(samples) " *
        "accepted samples.")
    nothing
end

function _compute_principal_with_canonical_recovery(solve_principal, principal_accepts,
        compute_canonical, principal_label, canonical_label, dict, moments,
        state::CanonicalBracketState;
        verbose=false, config::GaussRuleConfig=GaussRuleConfig(),
        max_adaptive_steps::Int=32, lost_digits::Real=DEFAULT_LOST_DIGITS,
        options...)
    last_w, last_x = nothing, nothing
    last_diag = nothing

    for refinement in 0:_MAX_PRINCIPAL_RECOVERY_REFINEMENTS
        converged, last_w, last_x, last_diag =
            solve_principal(state)
        if converged || principal_accepts(last_diag)
            return converged, last_w, last_x, last_diag
        end

        refinement == _MAX_PRINCIPAL_RECOVERY_REFINEMENTS && break
        if iszero(state.width)
            verbose && println("$(principal_label): saved canonical bracket " *
                "is already exact; no closer canonical retry seed exists.")
            break
        end

        verbose && println("$(principal_label): nonlinear solve failed; refining " *
            "saved canonical bracket before retry $(refinement + 1) of " *
            "$(_MAX_PRINCIPAL_RECOVERY_REFINEMENTS).")
        refined_state = _refine_canonical_bracket(
            compute_canonical, canonical_label, dict, moments, state;
            verbose, config, max_adaptive_steps, lost_digits,
            options...)
        refined_state === nothing && break
        state = refined_state
    end

    false, last_w, last_x, last_diag
end

function compute_upper_principal_representation(dict, moments, w0, x0;
        verbose=false, config::GaussRuleConfig=GaussRuleConfig(),
        options...)
    @assert isodd(length(dict))
    @assert length(moments) == length(dict)
    @assert length(w0) == (length(dict)>>1)+1

    rule = upper_principal_rule(dict, moments, config)
    try
        solve_system(rule, w0, x0; verbose, options...)
    catch e
        if e isa InterruptException
            rethrow()
        end
        false, w0, x0, _solver_exception_diagnostic(e)
    end
end

function compute_lower_canonical_representation(dict, moments, xi, w0, x0;
        verbose=false, config::GaussRuleConfig=GaussRuleConfig(),
        options...)
    @assert length(moments) == length(dict)

    if length(dict) == 2
        try
            a, b = config.add_endpoint == :right ?
                (xi, supportright(dict)) :
                (supportleft(dict), xi)
            return _compute_two_point_canonical_representation(
                dict, moments, a, b; options...)
        catch e
            if e isa InterruptException
                rethrow()
            end
            println("ERROR THROWN at $(xi) in computation of two-point lower canonical")
            @show e
            return false, w0, x0, _solver_exception_diagnostic(e)
        end
    end

    if isodd(length(dict))
        _gengauss_debug_println("DEBUG: compute_lower_canonical_representation, odd number of basis functions, (", length(dict), ")")
        # odd number of (n+1) basis functions (even n)
        @assert length(w0) == (length(dict)+1)>>1
        fixed_idx = [config.add_endpoint == :right ? 1 : length(w0)]
        rule = CanonicalRepresentationEven_J1(dict, xi, moments, fixed_idx)
    else
        _gengauss_debug_println("DEBUG: compute_lower_canonical_representation, even number of basis functions, (", length(dict), ")")
        # even number of (n+1) basis functions (odd n)
        @assert length(w0) == (length(dict)>>1)+1
        if config.add_endpoint == :right
            # Mirror of the :left branch below: we added the right endpoint b
            # (with weight 0) to a previous principal representation, and ξ is
            # the leftmost moving root. The canonical that includes b is the
            # K_1 canonical (Karlin–Studden, see eq. (2.12) of the paper).
            fixed_idx = [1, length(w0)]
            rule = CanonicalRepresentationOdd_K1(dict, xi, moments, fixed_idx)
        else
            # Added the left endpoint a; ξ is the rightmost moving root.
            # The canonical including a is the J_1 canonical (eq. (2.13)).
            fixed_idx = [1, length(w0)]
            rule = CanonicalRepresentationOdd_J1(dict, xi, moments, fixed_idx)
        end
    end

    try
        solve_system(rule, w0, x0; verbose, options...)
    catch e
        if e isa InterruptException
            rethrow()
        end
        println("ERROR THROWN at $(xi) in computation of lower canonical")
        @show e
        false, w0, x0, _solver_exception_diagnostic(e)
    end
end

function estimate_lower_canonical_representation(dict, moments, a, b, w0, x0;
        verbose = false, config::GaussRuleConfig=GaussRuleConfig(),
        sweep_direction::Union{Symbol,Nothing}=nothing,
        max_adaptive_steps::Int=32, lost_digits::Real=DEFAULT_LOST_DIGITS,
        options...)
    @assert length(dict) == length(moments)
    l = length(dict) >> 1
    #@assert length(w0) == l
    #@assert length(x0) == l

    verbose && println("Estimating lower canonical representation, xi between $(a) and $(b)")

    digits = max(lost_digits, _orthogonalization_lost_digits_floor(dict))

    seed = _estimate_canonical_representation_by_sweep(
        compute_lower_canonical_representation, "Lower canonical",
        dict, moments, a, b, w0, x0;
        verbose, config, sweep_direction, max_adaptive_steps,
        lost_digits=digits, options...)
    if seed === nothing
        hint = _continuation_solver_hint(options)
        error("Lower canonical sweep failed to locate a next-moment " *
              "sign change within the adaptive sweep budget. Check the basis " *
              "or increase max_adaptive_steps/lost_digits if the " *
              "nonlinear residuals are acceptable." * hint)
    end
    seed
end

function compute_lower_principal_representation(dict, moments, w0, x0;
        verbose=false, config::GaussRuleConfig=GaussRuleConfig(),
        options...)
    if isodd(length(dict))
        # odd number of (n+1) basis functions (even n)
        rule = LowerPrincipalEven(dict, moments)
    else
        # even number of (n+1) basis functions (odd n)
        rule = LowerPrincipalOdd(dict, moments)
    end
    try
        solve_system(rule, w0, x0; verbose, options...)
    catch e
        if e isa InterruptException
            rethrow()
        end
        false, w0, x0, _solver_exception_diagnostic(e)
    end
end

"""
    compute_canonical_both_ends(dict, moments, ξ, w0, x0; position_of_xi, ...)

Newton-solve the canonical of `c^{length(dict)}` (length(dict) must be odd, so
`n` is even) with both endpoints AND a third interior position fixed. The
interior fixed position is at index `position_of_xi` and takes value `ξ`. This
calls `CanonicalRepresentationEven_K1` with the appropriate `fixed_idx`:

- `position_of_xi == 2`: the K_1 form (eq. 2.16) — ξ is the 2nd-leftmost root.
- `position_of_xi == l_rule - 1`: the symmetric K_{l-1} form — ξ is the
  2nd-rightmost root.

DOFs: `l_rule = (n>>1) + 2` weights + `(l_rule - 3)` free interior positions =
`length(dict)` equations. Square Newton.
"""
function compute_canonical_both_ends(dict, moments, ξ, w0, x0;
        verbose=false, position_of_xi::Int=2, options...)
    @assert isodd(length(dict)) "compute_canonical_both_ends: length(dict) must be odd (so n is even); got $(length(dict))"
    @assert length(moments) == length(dict)
    n_inner = length(dict) - 1
    l_rule = (n_inner >> 1) + 2
    @assert length(w0) == l_rule
    @assert 2 <= position_of_xi <= l_rule - 1 "position_of_xi must be in 2..$(l_rule-1)"
    fixed_idx = [1, position_of_xi, l_rule]
    rule = CanonicalRepresentationEven_K1(dict, ξ, moments, fixed_idx)
    try
        solve_system(rule, w0, x0; verbose, options...)
    catch e
        if e isa InterruptException
            rethrow()
        end
        verbose && println("compute_canonical_both_ends: error at ξ=$(ξ): ", e)
        false, w0, x0, _solver_exception_diagnostic(e)
    end
end

function _canonical_both_ends_solver(position_of_xi::Int)
    function (dict, moments, ξ, w_seed, x_seed;
            verbose=false, config::GaussRuleConfig=GaussRuleConfig(),
            options...)
        compute_canonical_both_ends(dict, moments, ξ, w_seed, x_seed;
            verbose, position_of_xi, options...)
    end
end

"""
    estimate_canonical_both_ends(dict, moments, ξ_lo, ξ_hi, w0, x0; ...)

Sweep ξ in `(ξ_lo, ξ_hi)` to locate the root of
`F(ξ) = ⟨w(ξ), dict[end](x(ξ))⟩ - moments[end]`,
where the canonical at ξ is the both-endpoints canonical of `c^{length(dict)-1}`
(i.e. we strip the last basis function so the canonical is square, and use the
last basis function as the moment monitor — same pattern as
`estimate_lower_canonical_representation`).

Streams through ξ in `sweep_dir`, solving one canonical at a time and stopping
as soon as adjacent accepted samples bracket the residual sign change. Returns
the retained `CanonicalBracketState`, or `nothing` if no sign change is
established within the adaptive sweep budget.
"""
function estimate_canonical_both_ends(dict, moments, ξ_lo, ξ_hi, w0, x0;
        position_of_xi::Int=2, sweep_dir::Symbol=:left_to_right,
        max_adaptive_steps::Int=32, lost_digits::Real=DEFAULT_LOST_DIGITS,
        verbose=false, options...)
    digits = max(lost_digits, _orthogonalization_lost_digits_floor(dict))

    compute_one_both_ends = _canonical_both_ends_solver(position_of_xi)

    _estimate_canonical_representation_by_sweep(compute_one_both_ends,
        "Both-end canonical", dict, moments, ξ_lo, ξ_hi, w0, x0;
        verbose, sweep_direction=sweep_dir, max_adaptive_steps,
        lost_digits=digits, options...)
end

"""
    compute_lobatto_step(dict, moments, radau_w, radau_x, config; ...)

Compute the Gauss-Lobatto rule (upper principal of `c^{n_dict}`) by tracing the
K-canonical of `c^{n_dict-1}` with both endpoints fixed and one interior
position used as the continuation parameter ξ.

For `add_endpoint=:right`: ξ is the 2nd-leftmost position of the (l+1)-point
canonical. The seed is the right-Radau rule (UP of `c^{n_dict-1}`) padded with
`a` at weight 0; this seed is structurally on the K_1 canonical at the left
boundary, ξ_seed = `radau_x[1]` = `s_1` of `c^{n_dict-1}`. Bracket is
`(radau_x[1], radau_x[2])` — the seed's natural ξ as the lower bound and the
next inner Radau node as a wider upper bound.

For `add_endpoint=:left`: ξ is the 2nd-rightmost position. Seed is the
left-Radau rule (LP of `c^{n_dict-1}`) padded with `b` at weight 0; ξ_seed =
`radau_x[end]` = `t_l` of `c^{n_dict-1}` = right boundary of K_{l-1}. Bracket
is `(radau_x[end-1], radau_x[end])`.

The sweep finds adjacent ξ values where the next-moment residual `F(ξ)` changes
sign. The better endpoint seeds the final `UpperPrincipalOdd` nonlinear solve.
If that solve fails, bounded secant refinement tightens the retained
canonical bracket before retrying.

Requires `iseven(length(dict))`.

Returns `(converged, w, x, diag)` where `diag` is `nothing` if the ξ continuation
step failed early; otherwise a nonlinear-solver diagnostic like `solve_system`
(residual norms vs `ftol`, or `:error` if the solver threw). A final solve
within `lost_digits` decimal digits of `ftol` is accepted with a
warning when `verbose=true`.
"""
function compute_lobatto_step(dict, moments, radau_w, radau_x,
        config::GaussRuleConfig; verbose=false, max_adaptive_steps::Int=32,
        lost_digits::Real=DEFAULT_LOST_DIGITS,
        solver_tolerance=nothing,
        intermediate_tolerance=nothing, options...)
    n_dict = length(dict)
    @assert iseven(n_dict) "Gauss-Lobatto requires even basis length"
    @assert length(radau_x) == (n_dict >> 1) "Radau seed expected to have l = $(n_dict>>1) nodes"
    @assert length(radau_x) >= 2 "Radau-only Lobatto bracket needs at least two Radau nodes"
    T = eltype(radau_w)
    a_pt = T(supportleft(dict))
    b_pt = T(supportright(dict))
    digits = max(lost_digits, _orthogonalization_lost_digits_floor(dict))

    if config.add_endpoint == :right
        # Seed: prepend a (weight 0) to right-Radau (UP of c^{n_dict-1}).
        # Right-Radau has its rightmost node at b, so the padded seed has shape
        # [a, x_1_radau, ..., x_{l-1}_radau, b]. ξ at index 2 = first interior
        # of right-Radau = s_1 of c^{n_dict-1} = LEFT boundary of K_1.
        seed_w = vcat(zero(T), radau_w)
        seed_x = vcat(a_pt, radau_x)
        position_of_xi = 2
        ξ_lo = T(radau_x[1])
        ξ_hi = T(radau_x[2])
        sweep_dir = :left_to_right
    else
        # Mirror: append b (weight 0) to left-Radau (LP of c^{n_dict-1}).
        # Left-Radau has its leftmost node at a, so the padded seed has shape
        # [a, x_2_radau, ..., x_l_radau, b]. ξ at index l = last interior
        # of left-Radau = t_l of c^{n_dict-1} = RIGHT boundary of K_{l-1}.
        seed_w = vcat(radau_w, zero(T))
        seed_x = vcat(radau_x, b_pt)
        position_of_xi = length(seed_x) - 1
        ξ_lo = T(radau_x[end-1])
        ξ_hi = T(radau_x[end])
        sweep_dir = :right_to_left
    end

    verbose && println("  compute_lobatto_step: sweep interval [ξ_lo, ξ_hi] = [$(ξ_lo), $(ξ_hi)], sweep=$(sweep_dir), pos_of_xi=$(position_of_xi)")
    verbose && println("  compute_lobatto_step: seed_x = $(seed_x)")

    canonical_state = estimate_canonical_both_ends(dict, moments, ξ_lo, ξ_hi, seed_w, seed_x;
        position_of_xi, sweep_dir, verbose, max_adaptive_steps,
        lost_digits=digits, intermediate_tolerance,
        options...)
    if canonical_state === nothing
        verbose && println("compute_lobatto_step: canonical seed not found")
        return false, seed_w, seed_x, nothing
    end

    # Final solve on UpperPrincipalOdd: same fixed structure as the canonical
    # (a, b at indices 1 and l_rule), but now ξ is unfixed and the system is
    # solving for all 2l moments at once. The canonical seed is already very
    # close to the true Lobatto rule, so this converges quickly.
    rule = UpperPrincipalOdd(dict, moments)
    solve_lobatto = function (state)
        seed = state.best
        try
            solve_system(rule, seed.w, seed.x;
                verbose, mads_bracket=state,
                solver_tolerance,
                intermediate_tolerance=:strict,
                lost_digits=digits, options...)
        catch e
            if e isa InterruptException
                rethrow()
            end
            println("compute_lobatto_step: error in final UpperPrincipalOdd solve: ", e)
            false, seed.w, seed.x, _solver_exception_diagnostic(e)
        end
    end
    compute_one_both_ends = _canonical_both_ends_solver(position_of_xi)
    converged_final, wf, xf, diag =
        _compute_principal_with_canonical_recovery(
            solve_lobatto,
            diagnostic -> _lost_digits_accepts(diagnostic, digits),
            compute_one_both_ends, "Gauss-Lobatto final",
            "Both-end canonical", dict, moments, canonical_state;
            verbose, config, max_adaptive_steps,
            lost_digits=digits, intermediate_tolerance,
            options...)
    lost_digits_converged_final = !converged_final &&
        _lost_digits_accepts(diag, digits)
    if lost_digits_converged_final && verbose
        _warn_lost_digits_acceptance("Gauss-Lobatto final", nothing, diag,
            digits, "lost_digits")
    end
    converged_final || lost_digits_converged_final, wf, xf, diag
end


"""
    compute_gauss_rules(dict::Dictionary, moments = nothing;
        measure = nothing, verbose = false, principal = :lower,
        add_endpoint = nothing, max_adaptive_steps = 32,
        lost_digits = 2, solver_tolerance = nothing,
        intermediate_tolerance = nothing,
        differentiable = true, options...)

Compute the sequence of generalized Gaussian quadrature rules associated with
`dict`. Returns the final rule together with all intermediate principal
representations encountered during continuation.

Which final rule comes out is determined by the parity of `length(dict)` and the
`principal` keyword:

| `length(dict)`     | `principal=:lower`              | `principal=:upper`                  |
| ------------------ | ------------------------------- | ----------------------------------- |
| even (= 2l)        | l-point Gauss-Legendre (no ends) | (l+1)-point Gauss-Lobatto (both ends) |
| odd (= 2l+1)       | (l+1)-point left-Radau           | (l+1)-point right-Radau             |

`:lower` is the lower principal of c^{length(dict)}; `:upper` is the upper
principal of the same moment vector. Both are exact on all `length(dict)` basis
functions. The l-point right-Radau and l-point left-Radau rules computed at
intermediate iterations are *not* returned by `:upper`/`:lower` directly — they
are exposed only through the `xi_checkpoints` / `w_checkpoints` / `x_checkpoints`
sequences.

`add_endpoint` selects the continuation path the algorithm follows. If omitted,
it defaults to the natural pairing for the requested principal:

- `principal=:lower` -> `add_endpoint=:left`
- `principal=:upper` -> `add_endpoint=:right`

With an explicit value:

- `:right`: anchor at the right endpoint, sweeping right-to-left.
  Each iteration's Step 2 produces an upper-principal-style rule (right-Radau).
- `:left`: anchor at the left endpoint, sweeping left-to-right.
  Each iteration's Step 2 produces a lower-principal-style rule (left-Radau).

For **even** `length(dict)`, both `add_endpoint` settings reach the same final
rule for a given `principal` (a unique Gauss-Legendre rule for `:lower`,
a unique Gauss-Lobatto rule for `:upper`), assuming the basis functions are
well-defined at both endpoints; only the path differs. If one endpoint hosts a
singularity, anchor at the *other* endpoint so the seed rules remain well-defined.

For **odd** `length(dict)`, the natural product of `:right` continuation is
right-Radau and the natural product of `:left` continuation is left-Radau, so
the (principal, add_endpoint) pair must match: `:upper` requires `:right` and
`:lower` requires `:left`. The mismatched combinations error out up-front.

Returns: `(w, x, xi_checkpoints, w_checkpoints, x_checkpoints)` where the last
three are the sequence of intermediate quadrature rules.

Keyword arguments:
- `measure`: if `moments === nothing`, compute moments with
  `compute_moments(dict; measure=measure)`. Ignored when `moments` are passed
  explicitly.
- `verbose`: print progress information during the continuation.
- `max_adaptive_steps`: maximum midpoint retries per canonical continuation
  target when a nonlinear solve fails from the previous converged point.
- `lost_digits`: extra decimal digits by which canonical, principal, and final
  Gauss-Lobatto nonlinear residuals may miss `solver_tolerance` and still be
  accepted with a warning when `verbose=true`. The acceptance band is
  `min(solver_tolerance, active_tolerance) * 10^lost_digits`, so continuation
  never accepts residuals above the strict slack band when intermediate solves
  use a looser target. The default is 2, increased if needed to
  `ceil(orthogonalization_digits_lost/2)` for bases returned by
  `orthogonalize_basis`. Pass `lost_digits=...` to override it for one call.
- `solver_tolerance`: optional positive absolute residual tolerance for the
  nonlinear solves. Default `nothing` uses `10*eps(T)`, where `T` is the
  working numeric type, including `BigFloat`.
- `intermediate_tolerance`: optional positive absolute residual tolerance for
  canonical and nonterminal principal solves. The final returned rule is still
  polished with `solver_tolerance`. Checkpoints may be approximate when this is
  set. Default `nothing` uses a hybrid continuation tolerance:
  `min(max(10*sqrt(solver_tolerance), 1e-8), 1e-2)`.
  Pass `:strict` to disable the default hybrid and keep intermediate solves at
  `solver_tolerance`. An explicit positive value overrides the default directly.
- `differentiable`: when `true`, use Newton solves with analytic derivatives
  where available and centered finite differences for missing derivatives.
  Released support-boundary seeds use a second-order one-sided stencil until
  they move into the interior. When `false`, use Brent for the one-point rule
  and MADS for multidimensional solves.
- `options`: forwarded to differentiable nonlinear and scalar root solves.
  MADS uses an internal focused OrthoMADS policy.
"""
function compute_gauss_rules(dict::Dictionary, moments::Union{Nothing, Any} = nothing;
        measure = nothing, verbose = false, principal::Symbol = :lower,
        add_endpoint::Union{Nothing,Symbol} = nothing,
        max_adaptive_steps::Int = 32,
        lost_digits::Real = DEFAULT_LOST_DIGITS,
        solver_tolerance::Union{Nothing,Real} = nothing,
        intermediate_tolerance::Union{Nothing,Real,Symbol} = nothing,
        differentiable::Bool = true,
        options...)

    n = length(dict)
    add_endpoint_resolved = resolve_add_endpoint(principal, add_endpoint)
    # Automatically determine stop_at_odd_gauss based on dict length
    # If dict is even, compute full sequence (stop_at_odd_gauss = false)
    # If dict is odd, stop at the odd-length rule (stop_at_odd_gauss = true)
    stop_at_odd_gauss = isodd(n)

    # Reject mismatched (principal, add_endpoint) pairs for odd basis length.
    # For odd n_dict the natural product of `:right` continuation is the upper
    # principal of c^{n_dict} (right-Radau) and the natural product of `:left`
    # continuation is the lower principal of c^{n_dict} (left-Radau). The other
    # combinations would require a non-natural continuation path that this
    # algorithm does not implement; refuse them up-front rather than producing
    # a wrong rule silently.
    if stop_at_odd_gauss
        if principal == :upper && add_endpoint_resolved == :left
            error("compute_gauss_rules: for odd length(dict), principal=:upper requires add_endpoint=:right (the natural pairing produces right-Radau). Got principal=:upper, add_endpoint=:left.")
        end
        if principal == :lower && add_endpoint_resolved == :right
            error("compute_gauss_rules: for odd length(dict), principal=:lower requires add_endpoint=:left (the natural pairing produces left-Radau). Got principal=:lower, add_endpoint=:right.")
        end
    end

    if isnothing(moments)
        moments = compute_moments(dict; measure=measure)
    end

    config = GaussRuleConfig(; principal, add_endpoint=add_endpoint_resolved)
    steps = default_representation_steps(config.principal, config.add_endpoint, iseven(n))
    lost_digits_resolved =
        max(lost_digits, _orthogonalization_lost_digits_floor(dict))

    policy_type = promote_type(codomaintype(dict), eltype(moments))
    # Validate the public override once, including short paths that do not need
    # an intermediate solve. Individual solver calls resolve their working type.
    _resolve_solver_tolerances(policy_type, solver_tolerance,
        intermediate_tolerance)

    l = n >> 1 # equivalent to l = n / 2 integer division
    T = codomaintype(dict)

    left_support = supportleft(dict)
    right_support = supportright(dict)
    
    # xi_extractor extracts the node at the "end" of the sweep direction (where we're progressing toward)
    # For left_to_right sweep (add_endpoint == :left): extract rightmost node (last) - we're progressing rightward
    # For right_to_left sweep (add_endpoint == :right): extract leftmost node (first) - we're progressing leftward
    xi_extractor = config.add_endpoint == :left ? last : first

    # If stop_at_odd_gauss is true (dict is odd), stop when we reach length n
    # If stop_at_odd_gauss is false (dict is even), compute full sequence (stop_target_len = nothing)
    stop_target_len = stop_at_odd_gauss ? n : nothing

    # Unified checkpoints for all intermediate quadrature rules
    xi_checkpoints = T[]
    w_checkpoints = Vector{Vector{T}}()
    x_checkpoints = Vector{Vector{T}}()

    # Add initial anchor endpoint as first checkpoint
    push!(xi_checkpoints, config.add_endpoint == :left ? left_support : right_support)
    push!(w_checkpoints, T[])  # Empty placeholder (no quadrature rule yet)
    push!(x_checkpoints, T[])  # Empty placeholder (no quadrature rule yet)

    if n == 2 && config.principal == :upper
        verbose && println("Computing two-point Lobatto rule")
        w, x = compute_two_point_rule(dict[1:2], moments[1:2],
            left_support, right_support; solver_tolerance)
        verbose && println("Two point quadrature rule is: ", x, ", ", w)
        push!(xi_checkpoints, xi_extractor(x))
        push!(w_checkpoints, w)
        push!(x_checkpoints, x)
        return w, x, xi_checkpoints, w_checkpoints, x_checkpoints
    end

    # Select one nonlinear solver mode for all multidimensional continuation
    # solves. The scalar initializer uses Newton or Brent directly.
    solver_mode = _resolve_gauss_solver_mode(dict, differentiable)

    verbose && println("Computing initial one point rule")
    w, x = compute_one_point_rule(dict[1:2], moments[1:2];
        verbose, config, lost_digits=lost_digits_resolved,
        differentiable,
        solver_tolerance,
        intermediate_tolerance=n == 2 ?
            :strict : intermediate_tolerance,
        options...)
    verbose && println("One point quadrature rule is: ", x, ", ", w)

    # Add initial one-point rule as second checkpoint
    push!(xi_checkpoints, xi_extractor(x))
    push!(w_checkpoints, w)  # Store weights in w_checkpoints
    push!(x_checkpoints, x)  # Store nodes in x_checkpoints

    if n == 2
        # For length(dict) == 2 the loop body would not execute (k_max = 0).
        # :lower → the 1-point free-node rule already in (w, x).
        return w, x, xi_checkpoints, w_checkpoints, x_checkpoints
    end

    upper_principal_index = 0
    lower_principal_index = 0
    upper_canonical_state = nothing
    lower_canonical_state = nothing
    # Track the result of the most recent Step 2 (the anchored-endpoint Radau rule).
    # For add_endpoint=:right this is UP of c^{2k+1} (right-Radau) computed in the
    # :upper :principal arm; for add_endpoint=:left this is LP of c^{2k+1}
    # (left-Radau) computed in the :lower :principal arm with step.k == 1.
    # Used as the seed for the Gauss-Lobatto post-loop step when principal=:upper
    # and the basis length is even.
    last_radau_w = nothing
    last_radau_x = nothing
    lobatto_seed_ready = false

    k_max = iseven(n) ? l-1 : l
    for k = 1:k_max
        for step in steps
            _gengauss_debug_println("k = ", k, ": ", step.branch, " ", step.stage)
            tot_moments = 2*k + step.k
            if step.stage == :canonical
                # set initial rule as previous principal representation, possibly adding an endpoint
                # also set search interval for fixed xi
                if step.fixed_endpoint == :right
                    w0, x0 = [w; zero(T)], [x; right_support]
                    a, b = left_support, first(x)
                elseif step.fixed_endpoint == :left
                    w0, x0 = [zero(T); w], [left_support; x]
                    a, b = last(x), right_support
                elseif step.fixed_endpoint == :both
                    error("TODO")
                elseif isnothing(step.fixed_endpoint)
                    w0, x0 = w, x
                    if config.add_endpoint == :left
                        a, b = last(x), right_support
                    else
                        a, b = left_support, first(x)
                    end
                else
                    error("Unknown fixed_endpoint: ", step.fixed_endpoint)
                end
                _gengauss_debug_println("DEBUG: tot_moments: ", tot_moments)
                _gengauss_debug_println("DEBUG: length(dict[1:tot_moments]): ", length(dict[1:tot_moments]))
                _gengauss_debug_println("DEBUG: length(moments[1:tot_moments]): ", length(moments[1:tot_moments]))
                if step.branch == :upper
                    upper_canonical_state =
                        estimate_upper_canonical_representation(dict[1:tot_moments], moments[1:tot_moments], a, b, w0, x0; verbose, config, max_adaptive_steps, lost_digits=lost_digits_resolved, solver_tolerance, intermediate_tolerance, solver_mode, options...)
                    lower_canonical_state = nothing
                elseif step.branch == :lower
                    lower_canonical_state =
                        estimate_lower_canonical_representation(dict[1:tot_moments], moments[1:tot_moments], a, b, w0, x0; verbose, config, max_adaptive_steps, lost_digits=lost_digits_resolved, solver_tolerance, intermediate_tolerance, solver_mode, options...)
                    upper_canonical_state = nothing
                else
                    error("Unknown branch: ", step.branch)
                end
            elseif step.stage == :principal && step.branch == :upper
                @assert upper_canonical_state !== nothing
                _gengauss_debug_println("DEBUG: len moments: ", length(moments[1:tot_moments]))
                principal_label =
                    "upper principal representation $(upper_principal_index + 1)"
                # Only the rule returned to the caller needs strict polishing.
                # Even-basis Lobatto construction has its terminal solve after
                # the loop, so this main-loop checkpoint remains intermediate.
                terminal_main_solve = tot_moments == n &&
                    !(config.principal == :upper && iseven(n))
                principal_intermediate_tolerance =
                    terminal_main_solve ?
                    :strict : intermediate_tolerance
                solve_principal = function (state)
                    seed = state.best
                    compute_upper_principal_representation(
                        dict[1:tot_moments], moments[1:tot_moments],
                        seed.w, seed.x;
                        verbose, config, solver_mode,
                        mads_bracket=state,
                        solver_tolerance,
                        intermediate_tolerance=principal_intermediate_tolerance,
                        options...)
                end
                converged, w, x, principal_diag =
                    _compute_principal_with_canonical_recovery(
                        solve_principal,
                        diag -> _lost_digits_accepts(
                            diag, lost_digits_resolved),
                        compute_upper_canonical_representation,
                        principal_label, "Upper canonical",
                        dict[1:tot_moments], moments[1:tot_moments],
                        upper_canonical_state;
                        verbose, config, max_adaptive_steps,
                        lost_digits=lost_digits_resolved,
                        solver_tolerance,
                        intermediate_tolerance,
                        solver_mode, options...)
                xi = xi_extractor(x)
                upper_principal_index += 1
                _require_principal_convergence(
                    principal_label,
                    xi, converged, principal_diag, lost_digits_resolved;
                    verbose,
                    solver_tolerance,
                    intermediate_tolerance=principal_intermediate_tolerance,
                    policy_type=T)
                verbose && println("Upper principal representation ", upper_principal_index, " : xi is ", xi)
                verbose && println("    x: ", x)
                # Track Step 2 anchored-Radau rule (only set when step.k == 1, i.e.
                # when this :upper :principal step is in fact Step 2 of the loop).
                # Used as the seed for the post-loop Gauss-Lobatto step when the
                # basis length is even.
                if step.k == 1
                    last_radau_w = w
                    last_radau_x = x
                end
                # Add to unified checkpoints (always add, not conditional on has_upper)
                push!(xi_checkpoints, xi)
                push!(w_checkpoints, w)  # Store weights in w_checkpoints
                push!(x_checkpoints, x)  # Store nodes in x_checkpoints
                upper_canonical_state = nothing

                # Odd basis length: stop here when we've computed the principal
                # of c^{n_dict}. The :upper :principal step yields UP of c^{2k+1};
                # for odd n_dict = 2l+1 this matches at k = l (with step.k = 1).
                # Only the natural pairing (:upper, :right) reaches this point —
                # the mismatched (:lower, :right) pairing for odd n_dict is
                # rejected up-front in compute_gauss_rules.
                if stop_target_len !== nothing && (2*k+1 == stop_target_len)
                    return w, x, xi_checkpoints, w_checkpoints, x_checkpoints
                end
                # For even-length upper-principal requests, the final Step 2
                # Radau rule is enough to seed the post-loop Lobatto sweep. Do
                # not run the following lower canonical/principal Gauss solve.
                if config.principal == :upper && iseven(n) &&
                        k == k_max && step.k == 1
                    lobatto_seed_ready = true
                    break
                end
            elseif step.stage == :principal && step.branch == :lower
                @assert lower_canonical_state !== nothing
                _gengauss_debug_println("DEBUG: len moments: ", length(moments[1:tot_moments]))
                principal_label =
                    "lower principal representation $(lower_principal_index + 1)"
                # See the mirrored upper-principal branch above.
                terminal_main_solve = tot_moments == n &&
                    !(config.principal == :upper && iseven(n))
                principal_intermediate_tolerance =
                    terminal_main_solve ?
                    :strict : intermediate_tolerance
                solve_principal = function (state)
                    seed = state.best
                    compute_lower_principal_representation(
                        dict[1:tot_moments], moments[1:tot_moments],
                        seed.w, seed.x;
                        verbose, config, solver_mode,
                        mads_bracket=state,
                        solver_tolerance,
                        intermediate_tolerance=principal_intermediate_tolerance,
                        options...)
                end
                converged, w, x, principal_diag =
                    _compute_principal_with_canonical_recovery(
                        solve_principal,
                        diag -> _lost_digits_accepts(
                            diag, lost_digits_resolved),
                        compute_lower_canonical_representation,
                        principal_label, "Lower canonical",
                        dict[1:tot_moments], moments[1:tot_moments],
                        lower_canonical_state;
                        verbose, config, max_adaptive_steps,
                        lost_digits=lost_digits_resolved,
                        solver_tolerance,
                        intermediate_tolerance,
                        solver_mode, options...)
                xi = xi_extractor(x)
                lower_principal_index += 1
                _require_principal_convergence(
                    principal_label,
                    xi, converged, principal_diag, lost_digits_resolved;
                    verbose,
                    solver_tolerance,
                    intermediate_tolerance=principal_intermediate_tolerance,
                    policy_type=T)
                verbose && println("Lower principal representation ", lower_principal_index, " : xi is ", xi)
                verbose && println("    x: ", x)
                # Add to unified checkpoints (always add, not conditional on has_lower)
                push!(xi_checkpoints, xi)
                push!(w_checkpoints, w)  # Store weights in w_checkpoints
                push!(x_checkpoints, x)  # Store nodes in x_checkpoints
                lower_canonical_state = nothing

                # Track Step 2 anchored-Radau rule (left-Radau in the :left config).
                # Used as the seed for the post-loop Gauss-Lobatto step when the
                # basis length is even.
                if step.k == 1
                    last_radau_w = w
                    last_radau_x = x
                end

                # Odd basis length: stop at Step 2 (step.k == 1) of iteration k=l,
                # when the LP of c^{2l+1} (left-Radau) has just been computed.
                # Only the natural pairing (:lower, :left) reaches this point —
                # the mismatched (:upper, :left) pairing for odd n_dict is
                # rejected up-front in compute_gauss_rules. Step 4 (step.k == 2)
                # would access dict[2k+2] = dict[2l+2] past the end and is
                # unreachable for odd n_dict because we return here first.
                if stop_target_len !== nothing && step.k == 1 && (2*k+1 == stop_target_len)
                    return w, x, xi_checkpoints, w_checkpoints, x_checkpoints
                end
                # The left-anchored path computes its Radau seed in this lower
                # principal Step 2. Stop here for Lobatto just as the mirrored
                # right-anchored path stops in the upper-principal branch.
                if config.principal == :upper && iseven(n) &&
                        k == k_max && step.k == 1
                    lobatto_seed_ready = true
                    break
                end
            end
        end
        lobatto_seed_ready && break
    end
    # Post-loop Gauss–Lobatto step.
    #
    # For even upper-principal requests, the loop stopped after the final
    # Radau-type Step 2 for c^{n_dict-1}. That rule is padded with the missing
    # endpoint and swept directly to the Lobatto rule, so no lower-principal
    # Gauss solve is needed on the successful path.
    #
    # `compute_lobatto_step` does the continuation: it traces the both-endpoints
    # K-canonical of c^{n_dict-1} parameterized by ξ at the 2nd-leftmost (resp.
    # 2nd-rightmost) interior position, seeded by the Radau rule + missing
    # endpoint, until ξ is the value at which the next-moment residual vanishes.
    #
    # The result depends only on the moment vector c^{n_dict}, not on which
    # endpoint we anchored during continuation, so :upper, :right and :upper, :left
    # both produce the same Lobatto rule (provided neither endpoint hosts a
    # singularity that prevents one of the seed rules from being well-defined).
    #
    # For odd basis length, :upper has already been handled by the stop_target_len
    # early-return inside the loop, so we won't reach this block.
    if config.principal == :upper && iseven(n)
        @assert last_radau_w !== nothing "principal=:upper post-loop step needs the Radau rule from Step 2 of the loop."
        verbose && println("Computing Gauss-Lobatto rule (UP of c^$(n)) via post-loop step")
        verbose && println("  Radau rule (c^$(n-1)):   x = $(last_radau_x),  w = $(last_radau_w)")
        converged_lob, w_lob, x_lob, lobatto_diag =
            compute_lobatto_step(dict, moments, last_radau_w, last_radau_x, config;
                verbose, max_adaptive_steps,
                lost_digits=lost_digits_resolved,
                solver_tolerance,
                intermediate_tolerance,
                solver_mode, options...)
        if converged_lob
            verbose && println("  Lobatto rule found:     x = $(x_lob),  w = $(w_lob)")
            push!(xi_checkpoints, xi_extractor(x_lob))
            push!(w_checkpoints, w_lob)
            push!(x_checkpoints, x_lob)
            return w_lob, x_lob, xi_checkpoints, w_checkpoints, x_checkpoints
        else
            lobatto_diag_str =
                lobatto_diag === nothing ?
                "unavailable (e.g. continuation ξ seed not found before final solve)." :
                _format_solver_diagnostic(lobatto_diag)
            lost_digits_hint =
                lobatto_diag === nothing || (:error in keys(lobatto_diag)) ?
                "" :
                " Current effective lost_digits=$(lost_digits_resolved). " *
                "Increase lost_digits above this value to accept the final residual if the lost digits are acceptable."
            intermediate_hint = _continuation_solver_hint(;
                solver_tolerance, intermediate_tolerance,
                policy_type=T)
            error(
                "compute_gauss_rules: Gauss-Lobatto post-loop failed " *
                "(n=$(n), add_endpoint=$(config.add_endpoint)): " *
                lobatto_diag_str *
                lost_digits_hint *
                intermediate_hint *
                (verbose ? "" : " Re-run with verbose=true for continuation/sweep details.")
            )
        end
    end

    w, x, xi_checkpoints, w_checkpoints, x_checkpoints
end

"""
    compute_gauss_rule(dict::Dictionary, moments = nothing;
        kwargs...)

Convenience wrapper that returns only the terminal quadrature rule produced by
`compute_gauss_rules`. All keyword arguments are forwarded.
"""
function compute_gauss_rule(dict::Dictionary, moments = nothing; kwargs...)
    w, x, xi_checkpoints, w_checkpoints, x_checkpoints =
        compute_gauss_rules(dict, moments; kwargs...)
    # compute_gauss_rules already returns the terminal rule selected by
    # `principal` (either the LP of c^{n_dict} for :lower, the post-loop
    # Lobatto rule for :upper with even n_dict, or the Radau rule from the
    # stop_target_len early-return for odd n_dict).
    w, x
end
