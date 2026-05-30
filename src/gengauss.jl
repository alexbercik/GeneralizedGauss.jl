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

Fk(w, x, u2k, c2k) = apply_quad(w, x, u2k) - c2k
Gk(w, x, u2kp1, c2kp1) = apply_quad(w, x, u2kp1) - c2kp1

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

solver_tolerance(::Type{Float64}) = 10 * eps(Float64)
solver_tolerance(::Type{T}) where {T} = 10 * eps(T)
# Use one decimal digit above machine epsilon by default. This avoids treating
# harmless final-roundoff residuals as Newton failures.
solver_tolerance(::Type{BigFloat}) = BigFloat(10) * eps(BigFloat)

# Type-dispatched policy hooks. Downstream packages can extend these methods
# for their working precision while still using keyword overrides per call.
canonical_lost_digits(::Type{T}) where {T} = 2
lobatto_lost_digits(::Type{T}) where {T} = canonical_lost_digits(T)
principal_lost_digits(::Type{T}) where {T} = 0

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

function ismonotonic(values)
    if length(values) <= 1
        return true
    end
    if first(values) > last(values)
        reduce(&, values[1:end-1] .> values[2:end])
    else
        reduce(&, values[1:end-1] .< values[2:end])
    end
end

function sweep_indices(n, direction::Symbol)
    if direction == :left_to_right
        collect(1:n)
    elseif direction == :right_to_left
        collect(n:-1:1)
    else
        error("Unknown sweep direction $(direction). Use :left_to_right or :right_to_left.")
    end
end

function seeds_for_endpoint(target, sweep_start, sweep_end, start_idx, end_idx, pts, w, x, start_seed, end_seed)
    if target == sweep_start
        if start_seed === nothing
            return w[:,start_idx], x[:,start_idx]
        end
        return start_seed
    elseif target == sweep_end
        if end_seed === nothing
            return w[:,end_idx], x[:,end_idx]
        end
        return end_seed
    elseif target == pts[start_idx]
        return w[:,start_idx], x[:,start_idx]
    elseif target == pts[end_idx]
        return w[:,end_idx], x[:,end_idx]
    else
        return w[:,start_idx], x[:,start_idx]
    end
end

function refine_interval_from_sweep(a, b, pts2, w, x, order, Fvals_sweep, direction, start_seed, end_seed)
    start_idx = order[1]
    end_idx = order[end]
    start_pt = pts2[start_idx]
    end_pt = pts2[end_idx]
    start_val = Fvals_sweep[1]
    end_val = Fvals_sweep[end]
    decreasing = start_val > end_val
    sweep_start = direction == :left_to_right ? a : b
    sweep_end = direction == :left_to_right ? b : a
    if start_val > 0
        if decreasing
            a_new, b_new = sort((end_pt, sweep_end))
        else
            a_new, b_new = sort((sweep_start, start_pt))
        end
    else
        if decreasing
            a_new, b_new = sort((sweep_start, start_pt))
        else
            a_new, b_new = sort((end_pt, sweep_end))
        end
    end
    wa_new, xa_new = seeds_for_endpoint(a_new, sweep_start, sweep_end, start_idx, end_idx, pts2, w, x, start_seed, end_seed)
    wb_new, xb_new = seeds_for_endpoint(b_new, sweep_start, sweep_end, start_idx, end_idx, pts2, w, x, start_seed, end_seed)
    a_new, b_new, wa_new, xa_new, wb_new, xb_new
end

function _newton_diagnostic(rule, newton_x, tol)
    Fx = residual(rule, newton_x)
    residual_norm = maximum(abs, Fx)
    (; residual_norm, tolerance=tol, ratio=residual_norm / tol)
end

function _newton_exception_diagnostic(e)
    (; residual_norm=nothing, tolerance=nothing, ratio=nothing, error=e)
end

function _format_newton_diagnostic(diag)
    diag === nothing && return "diagnostic unavailable"
    if :error in keys(diag)
        return "diagnostic unavailable: solver threw $(typeof(diag.error))"
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

_lost_digits_policy_type(dict, moments) =
    promote_type(codomaintype(dict), eltype(moments))

function _resolve_canonical_lost_digits(dict, override,
        policy_type::Type=codomaintype(dict))
    base_digits = override === nothing ?
        canonical_lost_digits(policy_type) :
        override
    max(base_digits, _orthogonalization_lost_digits_floor(dict))
end

function _resolve_lobatto_lost_digits(dict, override,
        policy_type::Type=codomaintype(dict))
    base_digits = override === nothing ?
        lobatto_lost_digits(policy_type) :
        override
    max(base_digits, _orthogonalization_lost_digits_floor(dict))
end

function _resolve_principal_lost_digits(dict, override,
        policy_type::Type=codomaintype(dict))
    base_digits = override === nothing ?
        principal_lost_digits(policy_type) :
        override
    max(base_digits, _orthogonalization_lost_digits_floor(dict))
end

function _lost_digits_accepts(diag, lost_digits)
    diag === nothing && return false
    :error in keys(diag) && return false
    lost_digits < 0 && return false
    ratio = diag.ratio
    isfinite(ratio) || return false
    ratio <= typeof(ratio)(10)^typeof(ratio)(lost_digits)
end

function _canonical_lost_digits_accepts(diag, canonical_lost_digits)
    _lost_digits_accepts(diag, canonical_lost_digits)
end

function _lobatto_lost_digits_accepts(diag, lobatto_lost_digits)
    _lost_digits_accepts(diag, lobatto_lost_digits)
end

function _principal_lost_digits_accepts(diag, principal_lost_digits)
    _lost_digits_accepts(diag, principal_lost_digits)
end

function _warn_lost_digits_acceptance(label, location, diag, lost_digits,
        option_name)
    where = location === nothing ? "" : " for $(location)"
    println("WARNING: accepted $(label) Newton solve$(where) above ftol " *
            "[$(_format_newton_diagnostic(diag)), $(option_name)=$(lost_digits)]; " *
            "some accuracy may be lost.")
end

function _warn_canonical_lost_digits_acceptance(label, xi, diag, canonical_lost_digits)
    _warn_lost_digits_acceptance(label, xi, diag, canonical_lost_digits,
        "canonical_lost_digits")
end

function _warn_lobatto_lost_digits_acceptance(diag, lobatto_lost_digits)
    _warn_lost_digits_acceptance("Gauss-Lobatto final",
        nothing, diag, lobatto_lost_digits,
        "lobatto_lost_digits")
end

function _warn_principal_lost_digits_acceptance(label, location, diag, principal_lost_digits)
    _warn_lost_digits_acceptance(label, location, diag, principal_lost_digits,
        "principal_lost_digits")
end

function _require_principal_convergence(label, location, converged, diag,
        principal_lost_digits; verbose=false)
    converged && return true
    if _principal_lost_digits_accepts(diag, principal_lost_digits)
        verbose && _warn_principal_lost_digits_acceptance(label, location,
            diag, principal_lost_digits)
        return true
    end

    where = location === nothing ? "" : " for $(location)"
    println("ERROR: $(label) Newton solve$(where) failed " *
            "[$(_format_newton_diagnostic(diag)), " *
            "principal_lost_digits=$(principal_lost_digits)]; stopping.")
    error("$(label) Newton solve failed. Consider increasing principal_lost_digits if residual is still acceptable.")
end

function _has_first_derivatives(dict::Dictionary)
    x_probe = (supportleft(dict) + supportright(dict)) / 2
    all(i -> maybe_funeval_deriv(dict, i, x_probe, 1) !== nothing,
        1:length(dict))
end

function _resolve_gauss_solver(dict::Dictionary, solver::Symbol)
    if solver == :newton && !_has_first_derivatives(dict)
        @warn("No analytic first derivatives were provided for this basis. " *
              "Falling back to the derivative-free solver scaffold; only " *
              "the one-point rule is implemented so far, and the " *
              "multidimensional fallback is TBD.")
        return :fallback
    end
    solver
end

function _require_multidimensional_solver_available(solver::Symbol, n::Int)
    n <= 2 && return nothing
    solver == :newton && return nothing
    error("compute_gauss_rules: solver=$(solver) is implemented only for the one-point rule so far. " *
          "The derivative-free multidimensional continuation fallback is TBD.")
end

function solve_system_with_diagnostics(rule, w0, x0; verbose=false, options...)
    x_init = quad_to_newton(rule, w0, x0)
    F!(Fx, x) = residual!(Fx, rule, x)
    J!(Jx, x) = jacobian!(Jx, rule, x)

    tol = solver_tolerance(eltype(x_init))

    r = nlsolve(F!, J!, x_init; ftol = tol, options...)
    w, x = newton_to_quad(rule, r.zero)
    converged(r), w, x, _newton_diagnostic(rule, r.zero, tol)
end

function solve_system(rule, w0, x0; verbose=false, options...)
    ok, w, x, _ = solve_system_with_diagnostics(rule, w0, x0; verbose, options...)
    ok, w, x
end

function supportleft(dict)
    t = leftendpoint(support(dict))
    isinf(t) ? -one(t)*10 : t
end
supportright(dict) = rightendpoint(support(dict))

function compute_one_point_rule(dict, moments; verbose=false,
        config::GaussRuleConfig=GaussRuleConfig(),
        principal_lost_digits::Real=0, solver::Symbol=:newton, options...)
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
    base_tol = solver_tolerance(typeof(x0))
    x_tol = base_tol * max(one(x0), abs(a), abs(b))
    f_tol = base_tol * max(one(x0), maximum(abs, moments))

    if solver == :newton
        df = x -> c1 * funeval_deriv(dict, 2, x) -
                  c2 * funeval_deriv(dict, 1, x)
        converged, x, fx = _scalar_newton_root(f, df, a, b, x0;
            x_tol, f_tol, verbose, options...)
        if !converged
            verbose && println("WARNING: one-point scalar safeguarded Newton did not " *
                "converge; falling back to Brent root finding.")
            converged, x, fx = _brent_root_on_interval(f, a, b, x;
                x_tol, f_tol, options...)
        end
    else
        converged, x, fx = _brent_root_on_interval(f, a, b, x0;
            x_tol, f_tol, options...)
    end

    if !converged
        diag = (; residual_norm=abs(fx), tolerance=f_tol,
                ratio=abs(fx) / f_tol)
        _require_principal_convergence("one-point scalar", x,
            false, diag, principal_lost_digits; verbose)
    end

    f1x = funeval(dict, 1, x)
    if f1x == zero(f1x)
        error("Degenerate one-point rule: f₁(x) is zero at x=$(x), so the weight is undefined.")
    end
    w = moments[1] / f1x
    [w], [x]
end

function compute_two_point_rule(dict, moments; options...)
    @assert length(dict) == 2
    @assert length(moments) == 2

    a = supportleft(dict)
    b = supportright(dict)
    x = [a, b]

    u0a = funeval(dict, 1, a)
    u0b = funeval(dict, 1, b)
    u1a = funeval(dict, 2, a)
    u1b = funeval(dict, 2, b)

    denom = u0a * u1b - u0b * u1a

    tol = solver_tolerance(typeof(denom))
    if abs(denom) < tol
        error("Degenerate system: u₀(a)u₁(b) - u₀(b)u₁(a) ≈ 0.\n" *
              "The functions are linearly dependent on {a,b}, so a unique 2-point rule does not exist.")
    end

    wa = (moments[1] * u1b - moments[2] * u0b) / denom
    wb = (moments[2] * u0a - moments[1] * u1a) / denom
    w = [wa, wb]
    w, x
end


function compute_upper_canonical_representation(dict, moments, xi, w0, x0;
        verbose=false, config::GaussRuleConfig=GaussRuleConfig(),
        diagnostics::Bool=false, options...)
    @assert length(moments) == length(dict)

    _gengauss_debug_println("DEBUG: compute_upper_canonical_representation, even number of basis functions, (", length(dict), ")")

    if iseven(length(dict))
        # even number of (n+1) basis functions (odd n)
        @assert length(w0) == (length(dict)>>1)+1
        fixed_idx = config.add_endpoint == :right ? [1,length(w0)] : [length(w0)-1,length(w0)]
        rule = CanonicalRepresentationOdd_K1(dict, xi, moments, fixed_idx)
    else
        error("TODO")
    end
    try
        if diagnostics
            solve_system_with_diagnostics(rule, w0, x0; verbose, options...)
        else
            solve_system(rule, w0, x0; verbose, options...)
        end
    catch e
        if e isa InterruptException
            rethrow()
        end
        println("ERROR THROWN at $(xi) in computation of upper canonical")
        @show e
        diagnostics ? (false, w0, x0, _newton_exception_diagnostic(e)) : (false, w0, x0)
    end
end


function compute_many_upper_canonical_representation(dict, moments, w0, x0, pts;
        verbose=false, config::GaussRuleConfig=GaussRuleConfig(),
        sweep_direction::Union{Symbol,Nothing}=nothing,
        max_adaptive_steps::Int=32, initial_xi=nothing,
        canonical_lost_digits=nothing, options...)

    initial_xi_resolved = initial_xi === nothing ?
        (config.add_endpoint == :right ? first(x0) : x0[end-1]) :
        initial_xi
    lost_digits = _resolve_canonical_lost_digits(dict,
        canonical_lost_digits, _lost_digits_policy_type(dict, moments))
    _adaptive_canonical_sweep(compute_upper_canonical_representation,
        "upper canonical", dict, moments, w0, x0, pts, initial_xi_resolved;
        verbose, config, sweep_direction, max_adaptive_steps,
        canonical_lost_digits=lost_digits, options...)
end

function estimate_upper_canonical_representation(dict, moments, a, b, w0, x0;
        verbose=false, config::GaussRuleConfig=GaussRuleConfig(),
        sweep_direction::Union{Symbol,Nothing}=nothing,
        max_adaptive_steps::Int=32, canonical_lost_digits=nothing,
        options...)
    @assert isodd(length(dict))
    @assert length(dict) == length(moments)
    l = (length(dict)-1) >> 1
    @assert length(w0) == l+1
    @assert length(x0) == l+1

    verbose && println("Estimating upper canonical representation, xi between $(a) and $(b)")
    n = 8
    estimate_upper_canonical_representation(dict, moments, a, b, w0, x0, n;
        verbose, config, sweep_direction, max_adaptive_steps,
        canonical_lost_digits, options...)
end

function interpolate_starting_values(a, b, p, w_left, x_left, w_right, x_right)
    θ = (p-a)/(b-a)
    w0 = w_left + θ * (w_right-w_left)
    x0 = x_left + θ * (x_right-x_left)
    w0, x0
end

switches_sign(values) = ! (all(values .> 0) || all(values .< 0))

function _canonical_midpoint(x_good, x_failed)
    x_good + (x_failed - x_good) / 2
end

function _push_canonical_sample!(pts_done::Vector{T},
        w_done::Vector{Vector{T}}, x_done::Vector{Vector{T}},
        xi, w, x) where {T}
    push!(pts_done, xi)
    push!(w_done, T.(w))
    push!(x_done, T.(x))
    nothing
end

function _canonical_sample_matrix(cols::Vector{Vector{T}}, nrows::Int,
        perm::Vector{Int}) where {T}
    M = zeros(T, nrows, length(perm))
    for (j, idx) in enumerate(perm)
        M[:,j] = cols[idx]
    end
    M
end

function _adaptive_canonical_sweep(compute_one, label, dict, moments, w0, x0,
        pts, initial_xi;
        verbose=false, config::GaussRuleConfig=GaussRuleConfig(),
        sweep_direction::Union{Symbol,Nothing}=nothing,
        max_adaptive_steps::Int=32,
        canonical_lost_digits::Real=2, options...)
    n = length(pts)
    T = promote_type(eltype(w0), eltype(x0), eltype(pts))
    pts_done = T[]
    w_done = Vector{Vector{T}}()
    x_done = Vector{Vector{T}}()
    dir = sweep_direction !== nothing ? sweep_direction : get_direction(config)
    order = sweep_indices(n, dir)

    _gengauss_debug_println("w0: ", w0)
    _gengauss_debug_println("x0: ", x0)
    _gengauss_debug_println("pts: ", pts)
    _gengauss_debug_println("order: ", order)
    _gengauss_debug_println("initial_xi: ", initial_xi)

    xi_prev = initial_xi
    w_prev, x_prev = w0, x0
    for i in order
        target = pts[i]
        trial = target
        adaptive_steps = 0
        reached_target = false
        last_diagnostic = nothing

        while true
            converged, w1, x1, last_diagnostic =
                compute_one(dict, moments, trial, w_prev, x_prev;
                    verbose, config, diagnostics=true, options...)

            lost_digits_converged = !converged &&
                _canonical_lost_digits_accepts(last_diagnostic,
                    canonical_lost_digits)
            if lost_digits_converged && verbose
                _warn_canonical_lost_digits_acceptance(label, trial,
                    last_diagnostic, canonical_lost_digits)
            end

            if converged || lost_digits_converged
                _push_canonical_sample!(pts_done, w_done, x_done, trial, w1, x1)
                w_prev, x_prev = w1, x1
                xi_prev = trial
                if trial == target
                    reached_target = true
                    break
                end
                trial = target
                continue
            end

            adaptive_steps += 1
            if adaptive_steps > max_adaptive_steps
                break
            end

            next_trial = _canonical_midpoint(xi_prev, trial)
            if next_trial == xi_prev || next_trial == trial
                break
            end

            _gengauss_debug_println(label, ": retrying at ", next_trial,
                " after failed target ", trial,
                " from last converged xi ", xi_prev)
            trial = next_trial
        end

        if !reached_target && verbose
            println("Many $(label): not converged for $(target) [$(_format_newton_diagnostic(last_diagnostic))]")
        end
        reached_target || break
    end

    if isempty(pts_done)
        return false, zeros(T, length(w0), 0), zeros(T, length(x0), 0), T[]
    end

    perm = sortperm(pts_done)
    w = _canonical_sample_matrix(w_done, length(w0), perm)
    x = _canonical_sample_matrix(x_done, length(x0), perm)
    true, w, x, pts_done[perm]
end

function estimate_upper_canonical_representation(dict, moments, a, b, w0, x0, n;
        verbose=false, config::GaussRuleConfig=GaussRuleConfig(),
        sweep_direction::Union{Symbol,Nothing}=nothing,
        max_adaptive_steps::Int=32, canonical_lost_digits=nothing,
        options...)
    if n > 1024
        println(
            "estimate_upper_canonical_representation: canonical interval refinement is too small ",
            "(adaptive sweep grid parameter n=", n,
            " exceeds maximum 1024 without locating a suitable bracket).",
        )
        println(
            "Either check that the quadrature basis is correct, or increase ",
            "the canonical_lost_digits parameter.",
        )
        if !verbose
            println(
                "Re-run with verbose=true (e.g. compute_gauss_rule(...; verbose=true)) ",
                "to print Newton and sweep diagnostics.",
            )
        end
    end
    @assert n <= 1024

    dir = sweep_direction !== nothing ? sweep_direction : get_direction(config)

    pts = collect(range(a, b, length=n+2)[2:end-1])
    # end_seed will be computed from refine_interval_from_sweep results if needed
    end_seed = nothing

    initial_xi = dir == :left_to_right ? a : b
    lost_digits = _resolve_canonical_lost_digits(dict,
        canonical_lost_digits, _lost_digits_policy_type(dict, moments))
    some_converged, w, x, pts2 = compute_many_upper_canonical_representation(
        dict[1:end-1], moments[1:end-1], w0, x0, pts;
        verbose, config, sweep_direction, max_adaptive_steps, initial_xi,
        canonical_lost_digits=lost_digits, options...)
    if some_converged
        if verbose && length(pts2) != length(pts)
            println("Upper canonical: collected $(length(pts2)) converged samples for $(length(pts)) requested grid points")
        end
        Fvals = [Fk(w[:,i],x[:,i],dict[end], moments[end]) for i in 1:size(w,2)]
        order = sweep_indices(length(Fvals), dir)
        Fvals_sweep = Fvals[order]
        if verbose && !ismonotonic(Fvals_sweep)
            println("Upper canonical: function Fk is not monotonic along $(dir) sweep but should be.")
        end
        @assert ismonotonic(Fvals_sweep) "Upper canonical Fk sweep $(dir) should be monotonic.\nFvals_sweep = $(Fvals_sweep)\npts = $(pts2)\nIf the above Fvals are close to monotonic, this may indicate insufficient\nBigFloat precision for the current basis and degree.\nConsider increasing extra_digits or using a better-conditioned basis (e.g. Chebyshev).\nIf they are very bad, this may indicate something wrong with the basis or the moments."
        if switches_sign(Fvals_sweep)
            if first(Fvals_sweep) > 0
                I = findlast(Fvals_sweep .> 0)
            else
                I = findlast(Fvals_sweep .< 0)
            end
            left_idx = order[I]
            right_idx = order[I+1]
            pts2[left_idx], pts2[right_idx], w[:,left_idx], x[:,left_idx], w[:,right_idx], x[:,right_idx]
        else
            a_new, b_new, wa_new, xa_new, wb_new, xb_new =
                refine_interval_from_sweep(a, b, pts2, w, x, order, Fvals_sweep, dir, (w0, x0), end_seed)
            verbose && println("Upper canonical: refining from $((a,b)) to $((a_new,b_new)) in direction $(dir)")
            # Select the appropriate seed based on sweep direction:
            # the seed should come from the START of the sweep (where we have
            # the known good solution).
            if sweep_direction !== nothing
                w0_new = dir == :left_to_right ? wa_new : wb_new
                x0_new = dir == :left_to_right ? xa_new : xb_new
            else
                w0_new = config.add_endpoint == :left ? wa_new : wb_new
                x0_new = config.add_endpoint == :left ? xa_new : xb_new
            end
            estimate_upper_canonical_representation(dict, moments, a_new, b_new, w0_new, x0_new, n; verbose, config, sweep_direction, max_adaptive_steps, canonical_lost_digits=lost_digits, options...)
        end
    else
        verbose && println("Upper canonical: increasing n to $(2n)")
        estimate_upper_canonical_representation(dict, moments, a, b, w0, x0, 2n; verbose, config, sweep_direction, max_adaptive_steps, canonical_lost_digits=lost_digits, options...)
    end
end

function compute_upper_principal_representation(dict, moments, w0, x0;
        verbose=false, config::GaussRuleConfig=GaussRuleConfig(),
        diagnostics::Bool=false, options...)
    @assert isodd(length(dict))
    @assert length(moments) == length(dict)
    @assert length(w0) == (length(dict)>>1)+1

    rule = upper_principal_rule(dict, moments, config)
    try
        if diagnostics
            solve_system_with_diagnostics(rule, w0, x0; verbose, options...)
        else
            solve_system(rule, w0, x0; verbose, options...)
        end
    catch e
        if e isa InterruptException
            rethrow()
        end
        diagnostics ? (false, w0, x0, _newton_exception_diagnostic(e)) : (false, w0, x0)
    end
end

function compute_lower_canonical_representation(dict, moments, xi, w0, x0;
        verbose=false, config::GaussRuleConfig=GaussRuleConfig(),
        diagnostics::Bool=false, options...)
    @assert length(moments) == length(dict)

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
        if diagnostics
            solve_system_with_diagnostics(rule, w0, x0; verbose, options...)
        else
            solve_system(rule, w0, x0; verbose, options...)
        end
    catch e
        if e isa InterruptException
            rethrow()
        end
        println("ERROR THROWN at $(xi) in computation of lower canonical")
        @show e
        diagnostics ? (false, w0, x0, _newton_exception_diagnostic(e)) : (false, w0, x0)
    end
end

function compute_many_lower_canonical_representation(dict, moments, w0, x0, pts;
        verbose=false, config::GaussRuleConfig=GaussRuleConfig(),
        sweep_direction::Union{Symbol,Nothing}=nothing,
        max_adaptive_steps::Int=32, initial_xi=nothing,
        canonical_lost_digits=nothing, options...)

    initial_xi_resolved = initial_xi === nothing ?
        (config.add_endpoint == :right ? first(x0) : last(x0)) :
        initial_xi
    lost_digits = _resolve_canonical_lost_digits(dict,
        canonical_lost_digits, _lost_digits_policy_type(dict, moments))
    _adaptive_canonical_sweep(compute_lower_canonical_representation,
        "lower canonical", dict, moments, w0, x0, pts, initial_xi_resolved;
        verbose, config, sweep_direction, max_adaptive_steps,
        canonical_lost_digits=lost_digits, options...)
end

function estimate_lower_canonical_representation(dict, moments, a, b, w0, x0;
        verbose = false, config::GaussRuleConfig=GaussRuleConfig(),
        sweep_direction::Union{Symbol,Nothing}=nothing,
        max_adaptive_steps::Int=32, canonical_lost_digits=nothing,
        options...)
    @assert length(dict) == length(moments)
    l = length(dict) >> 1
    #@assert length(w0) == l
    #@assert length(x0) == l

    verbose && println("Estimating lower canonical representation, xi between $(a) and $(b)")
    n = 8
    estimate_lower_canonical_representation(dict, moments, a, b, w0, x0, n;
        verbose, config, sweep_direction, max_adaptive_steps,
        canonical_lost_digits, options...)
end

function estimate_lower_canonical_representation(dict, moments, a, b, w0, x0, n;
        verbose=false, config::GaussRuleConfig=GaussRuleConfig(),
        sweep_direction::Union{Symbol,Nothing}=nothing,
        max_adaptive_steps::Int=32, canonical_lost_digits=nothing,
        options...)
    if n > 1024
        println(
            "estimate_lower_canonical_representation: canonical interval refinement is too small ",
            "(adaptive sweep grid parameter n=", n,
            " exceeds maximum 1024 without locating a suitable bracket).",
        )
        println(
            "Either check that the quadrature basis is correct, or increase ",
            "the canonical_lost_digits parameter.",
        )
        if !verbose
            println(
                "Re-run with verbose=true (e.g. compute_gauss_rule(...; verbose=true)) ",
                "to print Newton and sweep diagnostics.",
            )
        end
    end
    @assert n <= 1024

    dir = sweep_direction !== nothing ? sweep_direction : get_direction(config)

    pts = collect(range(a, b, length=n+2)[2:end-1])
    # end_seed will be computed from refine_interval_from_sweep results if needed
    end_seed = nothing
    initial_xi = dir == :left_to_right ? a : b
    lost_digits = _resolve_canonical_lost_digits(dict,
        canonical_lost_digits, _lost_digits_policy_type(dict, moments))
    someconverged, w, x, pts2 = compute_many_lower_canonical_representation(
        dict[1:end-1], moments[1:end-1], w0, x0, pts;
        verbose, config, sweep_direction, max_adaptive_steps, initial_xi,
        canonical_lost_digits=lost_digits, options...)
    if someconverged
        if verbose && length(pts2) != length(pts)
            println("Lower canonical: collected $(length(pts2)) converged samples for $(length(pts)) requested grid points")
        end
        Fvals = [Fk(w[:,i],x[:,i],dict[end], moments[end]) for i in 1:size(w,2)]
        order = sweep_indices(length(Fvals), dir)
        Fvals_sweep = Fvals[order]
        if verbose && !ismonotonic(Fvals_sweep)
            println("Lower canonical: function Fk is not monotonic along $(dir) sweep but should be.")
        end
        @assert ismonotonic(Fvals_sweep) "Lower canonical Fk sweep $(dir) should be monotonic.\nFvals_sweep = $(Fvals_sweep)\npts = $(pts2)\nIf the above Fvals are close to monotonic, this may indicate insufficient\nBigFloat precision for the current basis and degree.\nConsider increasing extra_digits or using a better-conditioned basis (e.g. Chebyshev).\nIf they are very bad, this may indicate something wrong with the basis or the moments."
        if switches_sign(Fvals_sweep)
            _gengauss_debug_println("DEBUG: Detected sign change in lower canonical")
            if first(Fvals_sweep) > 0
                I = findlast(Fvals_sweep .> 0)
            else
                I = findlast(Fvals_sweep .< 0)
            end
            left_idx = order[I]
            right_idx = order[I+1]
            _gengauss_debug_println("DEBUG: Sign change detected at indices $(left_idx) and $(right_idx)")
            pts2[left_idx], pts2[right_idx], w[:,left_idx], x[:,left_idx], w[:,right_idx], x[:,right_idx]
        else
            _gengauss_debug_println("DEBUG: No sign change detected, refining interval")
            a_new, b_new, wa_new, xa_new, wb_new, xb_new =
                refine_interval_from_sweep(a, b, pts2, w, x, order, Fvals_sweep, dir, (w0, x0), end_seed)
            verbose && println("Lower canonical: refining from $((a,b)) to $((a_new,b_new)) in direction $(dir)")
            # Select the appropriate boundary based on add_endpoint
            if sweep_direction !== nothing
                w0_new = dir == :left_to_right ? wa_new : wb_new
                x0_new = dir == :left_to_right ? xa_new : xb_new
            else
                w0_new = config.add_endpoint == :left ? wa_new : wb_new
                x0_new = config.add_endpoint == :left ? xa_new : xb_new
            end
            estimate_lower_canonical_representation(dict, moments, a_new, b_new, w0_new, x0_new, 8;
                verbose, config, sweep_direction, max_adaptive_steps,
                canonical_lost_digits=lost_digits, options...)
        end
    else
        verbose && println("Lower canonical: no convergence on grid, increasing to n=$(2n)")
        estimate_lower_canonical_representation(dict, moments, a, b, w0, x0, 2n;
            verbose, config, sweep_direction, max_adaptive_steps,
            canonical_lost_digits=lost_digits, options...)
    end
end

function compute_lower_principal_representation(dict, moments, w0, x0;
        verbose=false, config::GaussRuleConfig=GaussRuleConfig(),
        diagnostics::Bool=false, options...)
    if isodd(length(dict))
        # odd number of (n+1) basis functions (even n)
        rule = LowerPrincipalEven(dict, moments)
    else
        # even number of (n+1) basis functions (odd n)
        rule = LowerPrincipalOdd(dict, moments)
    end
    try
        if diagnostics
            solve_system_with_diagnostics(rule, w0, x0; verbose, options...)
        else
            solve_system(rule, w0, x0; verbose, options...)
        end
    catch e
        if e isa InterruptException
            rethrow()
        end
        diagnostics ? (false, w0, x0, _newton_exception_diagnostic(e)) : (false, w0, x0)
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
        verbose=false, position_of_xi::Int=2, diagnostics::Bool=false,
        options...)
    @assert isodd(length(dict)) "compute_canonical_both_ends: length(dict) must be odd (so n is even); got $(length(dict))"
    @assert length(moments) == length(dict)
    n_inner = length(dict) - 1
    l_rule = (n_inner >> 1) + 2
    @assert length(w0) == l_rule
    @assert 2 <= position_of_xi <= l_rule - 1 "position_of_xi must be in 2..$(l_rule-1)"
    fixed_idx = [1, position_of_xi, l_rule]
    rule = CanonicalRepresentationEven_K1(dict, ξ, moments, fixed_idx)
    try
        if diagnostics
            solve_system_with_diagnostics(rule, w0, x0; verbose, options...)
        else
            solve_system(rule, w0, x0; verbose, options...)
        end
    catch e
        if e isa InterruptException
            rethrow()
        end
        verbose && println("compute_canonical_both_ends: error at ξ=$(ξ): ", e)
        diagnostics ? (false, w0, x0, _newton_exception_diagnostic(e)) : (false, w0, x0)
    end
end

"""
    compute_many_canonical_both_ends(dict, moments, w0, x0, pts, sweep_dir; ...)

Sweep over `pts` in the order given by `sweep_dir` (`:left_to_right` or
`:right_to_left`), solving the both-endpoints canonical at each ξ. Each step
warm-seeds from the previous *converged* solution. If Newton fails for a target,
retry by bisecting between the last converged ξ and the failed trial; if the
target still cannot be reached, stop the sweep rather than jumping farther away.
"""
function compute_many_canonical_both_ends(dict, moments, w0, x0, pts, sweep_dir::Symbol;
        verbose=false, position_of_xi::Int=2, max_adaptive_steps::Int=32,
        initial_xi=nothing, canonical_lost_digits=nothing, options...)
    initial_xi_resolved = initial_xi === nothing ? x0[position_of_xi] : initial_xi
    lost_digits = _resolve_canonical_lost_digits(dict,
        canonical_lost_digits, _lost_digits_policy_type(dict, moments))
    compute_one_both_ends = function (dict, moments, ξ, w_seed, x_seed;
            verbose=false, config::GaussRuleConfig=GaussRuleConfig(),
            diagnostics::Bool=false, options...)
        compute_canonical_both_ends(dict, moments, ξ, w_seed, x_seed;
            verbose, position_of_xi, diagnostics, options...)
    end
    _adaptive_canonical_sweep(compute_one_both_ends,
        "both-end canonical", dict, moments, w0, x0, pts, initial_xi_resolved;
        verbose, sweep_direction=sweep_dir, max_adaptive_steps,
        canonical_lost_digits=lost_digits, options...)
end

"""
    estimate_canonical_both_ends(dict, moments, ξ_lo, ξ_hi, w0, x0; ...)

Bisect ξ in `(ξ_lo, ξ_hi)` to bracket the root of
`F(ξ) = ⟨w(ξ), dict[end](x(ξ))⟩ - moments[end]`,
where the canonical at ξ is the both-endpoints canonical of `c^{length(dict)-1}`
(i.e. we strip the last basis function so the canonical is square, and use the
last basis function as the moment monitor — same pattern as
`estimate_lower_canonical_representation`).

Mirrors the recursive sample-and-refine logic of
`estimate_lower_canonical_representation`. Returns `(ξ_left, ξ_right, w_left,
x_left, w_right, x_right)` bracketing the sign change, or `nothing` if the
bracket cannot be established within `max_n_samples`.
"""
function estimate_canonical_both_ends(dict, moments, ξ_lo, ξ_hi, w0, x0;
        position_of_xi::Int=2, sweep_dir::Symbol=:left_to_right,
        n_samples::Int=8, max_n_samples::Int=1024,
        max_adaptive_steps::Int=32, canonical_lost_digits=nothing,
        verbose=false, options...)
    @assert n_samples <= max_n_samples
    pts = collect(range(ξ_lo, ξ_hi; length=n_samples+2)[2:end-1])
    dict_inner = dict[1:end-1]
    moments_inner = moments[1:end-1]
    initial_xi = sweep_dir == :left_to_right ? ξ_lo : ξ_hi
    lost_digits = _resolve_canonical_lost_digits(dict,
        canonical_lost_digits, _lost_digits_policy_type(dict, moments))
    some_ok, w, x, pts2 = compute_many_canonical_both_ends(dict_inner, moments_inner,
        w0, x0, pts, sweep_dir; position_of_xi, verbose,
        max_adaptive_steps, initial_xi,
        canonical_lost_digits=lost_digits, options...)
    if !some_ok
        if 2 * n_samples > max_n_samples
            verbose && println("estimate_canonical_both_ends: no convergence on $(n_samples) samples; max reached")
            return nothing
        end
        verbose && println("estimate_canonical_both_ends: no convergence; doubling to $(2*n_samples)")
        return estimate_canonical_both_ends(dict, moments, ξ_lo, ξ_hi, w0, x0;
            position_of_xi, sweep_dir, n_samples=2*n_samples, max_n_samples,
            max_adaptive_steps, canonical_lost_digits=lost_digits,
            verbose, options...)
    end
    if verbose && length(pts2) < length(pts)
        println("estimate_canonical_both_ends: $(length(pts2))/$(length(pts)) samples converged")
    end
    Fvals = [Fk(w[:, i], x[:, i], dict[end], moments[end]) for i in 1:size(w, 2)]
    order = sweep_indices(length(Fvals), sweep_dir)
    Fvals_sweep = Fvals[order]
    if verbose && !ismonotonic(Fvals_sweep)
        println("estimate_canonical_both_ends: F not monotonic in sweep direction (may indicate samples outside K_1)")
    end
    if switches_sign(Fvals_sweep)
        j = first(Fvals_sweep) > 0 ? findlast(Fvals_sweep .> 0) : findlast(Fvals_sweep .< 0)
        left_i = order[j]
        right_i = order[j + 1]
        return pts2[left_i], pts2[right_i], w[:, left_i], x[:, left_i], w[:, right_i], x[:, right_i]
    else
        if 2 * n_samples > max_n_samples
            verbose && println("estimate_canonical_both_ends: no sign change at $(n_samples) samples; max reached")
            return nothing
        end
        verbose && println("estimate_canonical_both_ends: no sign change; doubling samples to $(2*n_samples)")
        return estimate_canonical_both_ends(dict, moments, ξ_lo, ξ_hi, w0, x0;
            position_of_xi, sweep_dir, n_samples=2*n_samples, max_n_samples,
            max_adaptive_steps, canonical_lost_digits=lost_digits,
            verbose, options...)
    end
end

"""
    compute_lobatto_step(dict, moments, lp_w, lp_x, radau_w, radau_x, config; ...)

Compute the Gauss-Lobatto rule (upper principal of `c^{n_dict}`) by tracing the
K-canonical of `c^{n_dict-1}` with both endpoints fixed and one interior
position used as the continuation parameter ξ.

For `add_endpoint=:right`: ξ is the 2nd-leftmost position of the (l+1)-point
canonical. The seed is the right-Radau rule (UP of `c^{n_dict-1}`) padded with
`a` at weight 0; this seed is structurally on the K_1 canonical at the left
boundary, ξ_seed = `radau_x[1]` = `s_1` of `c^{n_dict-1}`. Bracket is
`(radau_x[1], lp_x[2])` — the seed's natural ξ as the lower bound and the
second LP node of `c^{n_dict}` as the upper, which contains the Lobatto rule's
ξ value (= `s_2` of `c^{n_dict}`).

For `add_endpoint=:left`: ξ is the 2nd-rightmost position. Seed is the
left-Radau rule (LP of `c^{n_dict-1}`) padded with `b` at weight 0; ξ_seed =
`radau_x[end]` = `t_l` of `c^{n_dict-1}` = right boundary of K_{l-1}. Bracket
is `(lp_x[end-1], radau_x[end])`.

The bisection finds ξ where the next-moment residual `F(ξ)` changes sign; the
better of the two bracket endpoints (the one with smaller `|F|`) is then handed
to one final `UpperPrincipalOdd` Newton solve to refine to the exact Lobatto rule.

Requires `iseven(length(dict))`.

Returns `(converged, w, x, diag)` where `diag` is `nothing` if the ξ bracket
step failed early; otherwise a Newton diagnostic like `solve_system_with_diagnostics`
(residual norms vs `ftol`, or `:error` if the solver threw). A final solve
within `lobatto_lost_digits` decimal digits of `ftol` is accepted with a
warning when `verbose=true`.
"""
function compute_lobatto_step(dict, moments, lp_w, lp_x, radau_w, radau_x,
        config::GaussRuleConfig; verbose=false, max_adaptive_steps::Int=32,
        canonical_lost_digits=nothing,
        lobatto_lost_digits=nothing, options...)
    n_dict = length(dict)
    @assert iseven(n_dict) "Gauss-Lobatto requires even basis length"
    @assert length(lp_x) == (n_dict >> 1) "lp rule expected to have l = $(n_dict>>1) nodes"
    T = eltype(lp_w)
    a_pt = T(supportleft(dict))
    b_pt = T(supportright(dict))
    lobatto_digits =
        _resolve_lobatto_lost_digits(dict, lobatto_lost_digits,
            _lost_digits_policy_type(dict, moments))

    if config.add_endpoint == :right
        # Seed: prepend a (weight 0) to right-Radau (UP of c^{n_dict-1}).
        # Right-Radau has its rightmost node at b, so the padded seed has shape
        # [a, x_1_radau, ..., x_{l-1}_radau, b]. ξ at index 2 = first interior
        # of right-Radau = s_1 of c^{n_dict-1} = LEFT boundary of K_1.
        seed_w = vcat(zero(T), radau_w)
        seed_x = vcat(a_pt, radau_x)
        position_of_xi = 2
        # Bracket: (s_1 of c^{n_dict-1}, t_2 of c^{n_dict}).
        # Lower = radau_x[1] = the seed's natural ξ (the K_1 left boundary).
        # Upper = lp_x[2] (or lp_x[end] when l=2). Lobatto's position 2 value
        # is s_2 of c^{n_dict} ∈ (t_1, t_2) of c^{n_dict} = (lp_x[1], lp_x[2]),
        # which lies inside this bracket. Sweeping left-to-right keeps Newton
        # close to a converged previous solution at every step.
        ξ_lo = T(radau_x[1])
        ξ_hi = T(length(lp_x) >= 2 ? lp_x[2] : lp_x[end])
        sweep_dir = :left_to_right
    else
        # Mirror: append b (weight 0) to left-Radau (LP of c^{n_dict-1}).
        # Left-Radau has its leftmost node at a, so the padded seed has shape
        # [a, x_2_radau, ..., x_l_radau, b]. ξ at index l = last interior
        # of left-Radau = t_l of c^{n_dict-1} = RIGHT boundary of K_{l-1}.
        seed_w = vcat(radau_w, zero(T))
        seed_x = vcat(radau_x, b_pt)
        position_of_xi = length(seed_x) - 1
        # Upper = radau_x[end] = the seed's natural ξ (right boundary of
        # K_{l-1}). Lower = lp_x[end-1] (or lp_x[1] when l=2).
        ξ_lo = T(length(lp_x) >= 2 ? lp_x[end-1] : lp_x[1])
        ξ_hi = T(radau_x[end])
        sweep_dir = :right_to_left
    end

    verbose && println("  compute_lobatto_step: bracket [ξ_lo, ξ_hi] = [$(ξ_lo), $(ξ_hi)], sweep=$(sweep_dir), pos_of_xi=$(position_of_xi)")
    verbose && println("  compute_lobatto_step: seed_x = $(seed_x)")

    bracket = estimate_canonical_both_ends(dict, moments, ξ_lo, ξ_hi, seed_w, seed_x;
        position_of_xi, sweep_dir, verbose, max_adaptive_steps,
        canonical_lost_digits, options...)
    if bracket === nothing
        verbose && println("compute_lobatto_step: bracket not found; falling back to seed")
        return false, seed_w, seed_x, nothing
    end

    _ξl, _ξr, w_left, x_left, w_right, x_right = bracket
    F_left = Fk(w_left, x_left, dict[end], moments[end])
    F_right = Fk(w_right, x_right, dict[end], moments[end])
    seed_w_final, seed_x_final = abs(F_left) <= abs(F_right) ? (w_left, x_left) : (w_right, x_right)

    # Final Newton on UpperPrincipalOdd: same fixed structure as the canonical
    # (a, b at indices 1 and l_rule), but now ξ is unfixed and the system is
    # solving for all 2l moments at once. The bracket-end seed is already very
    # close to the true Lobatto rule, so this converges quickly.
    rule = UpperPrincipalOdd(dict, moments)
    try
        converged_final, wf, xf, diag =
            solve_system_with_diagnostics(rule, seed_w_final, seed_x_final;
                verbose, options...)
        lost_digits_converged_final = !converged_final &&
            _lobatto_lost_digits_accepts(diag, lobatto_digits)
        if lost_digits_converged_final && verbose
            _warn_lobatto_lost_digits_acceptance(diag, lobatto_digits)
        end
        return converged_final || lost_digits_converged_final, wf, xf, diag
    catch e
        if e isa InterruptException
            rethrow()
        end
        println("compute_lobatto_step: error in final UpperPrincipalOdd Newton: ", e)
        return false, seed_w_final, seed_x_final, _newton_exception_diagnostic(e)
    end
end


"""
    compute_gauss_rules(dict::Dictionary, moments = nothing;
        measure = nothing, verbose = false, principal = :lower,
        add_endpoint = nothing, max_adaptive_steps = 32,
        canonical_lost_digits = nothing,
        principal_lost_digits = nothing,
        lobatto_lost_digits = nothing, solver = :newton, options...)

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
- `max_adaptive_steps`: maximum midpoint retries per canonical grid target when
  Newton fails from the previous converged continuation point.
- `canonical_lost_digits`: extra decimal digits by which intermediate
  canonical Newton residuals may miss `ftol` and still be accepted with a
  warning when `verbose=true`. Default `nothing` uses
  `canonical_lost_digits(T)`, where `T` is the promoted type of `dict` and
  `moments`, increased if needed to `ceil(orthogonalization_digits_lost/2)`
  for bases returned by
  `orthogonalize_basis`.
- `lobatto_lost_digits`: same lost-digit allowance for the final
  Gauss-Lobatto Newton solve, with a warning when `verbose=true`. Default
  `nothing` uses the same policy as `lobatto_lost_digits(T)`.
- `principal_lost_digits`: same lost-digit allowance for non-Lobatto principal
  Newton solves. Default `nothing` uses `principal_lost_digits(T)`, which is 0
  unless overridden, then applies the orthogonalization floor.
- `solver`: nonlinear solver selector. `:newton` is the default and current
  multidimensional implementation. If no first derivatives are available, the
  code warns and switches to the derivative-free fallback scaffold. At present
  that fallback is implemented only for the initial one-point rule.
- `options`: forwarded to the nonlinear solver used at every refinement step.
"""
function compute_gauss_rules(dict::Dictionary, moments::Union{Nothing, Any} = nothing;
        measure = nothing, verbose = false, principal::Symbol = :lower,
        add_endpoint::Union{Nothing,Symbol} = nothing,
        max_adaptive_steps::Int = 32,
        canonical_lost_digits = nothing,
        principal_lost_digits = nothing,
        lobatto_lost_digits = nothing,
        solver::Symbol = :newton,
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
    canonical_lost_digits_resolved =
        _resolve_canonical_lost_digits(dict, canonical_lost_digits,
            _lost_digits_policy_type(dict, moments))
    principal_lost_digits_resolved =
        _resolve_principal_lost_digits(dict, principal_lost_digits,
            _lost_digits_policy_type(dict, moments))
    lobatto_lost_digits_resolved =
        _resolve_lobatto_lost_digits(dict, lobatto_lost_digits,
            _lost_digits_policy_type(dict, moments))

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
        w, x = compute_two_point_rule(dict[1:2], moments[1:2])
        verbose && println("Two point quadrature rule is: ", x, ", ", w)
        push!(xi_checkpoints, xi_extractor(x))
        push!(w_checkpoints, w)
        push!(x_checkpoints, x)
        return w, x, xi_checkpoints, w_checkpoints, x_checkpoints
    end

    # -- Can we use derivatives? If yes, use a Newton solver.
    solver_resolved = _resolve_gauss_solver(dict, solver)
    _require_multidimensional_solver_available(solver_resolved, n)

    verbose && println("Computing initial one point rule")
    w, x = compute_one_point_rule(dict[1:2], moments[1:2];
        verbose, config, principal_lost_digits=principal_lost_digits_resolved,
        solver=solver_resolved, options...)
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
                    _, _, _, _, w2, x2 =
                        estimate_upper_canonical_representation(dict[1:tot_moments], moments[1:tot_moments], a, b, w0, x0; verbose, config, max_adaptive_steps, canonical_lost_digits=canonical_lost_digits_resolved, options...)
                        upper_canonical_state = (w2, x2)
                        lower_canonical_state = nothing
                elseif step.branch == :lower
                    _, _, _, _, w2, x2 =
                        estimate_lower_canonical_representation(dict[1:tot_moments], moments[1:tot_moments], a, b, w0, x0; verbose, config, max_adaptive_steps, canonical_lost_digits=canonical_lost_digits_resolved, options...)
                        upper_canonical_state = nothing
                        lower_canonical_state = (w2, x2)
                else
                    error("Unknown branch: ", step.branch)
                end
            elseif step.stage == :principal && step.branch == :upper
                @assert upper_canonical_state !== nothing
                w2, x2 = upper_canonical_state
                _gengauss_debug_println("DEBUG: x2: ", x2)
                _gengauss_debug_println("DEBUG: w2: ", w2)
                _gengauss_debug_println("DEBUG: len moments: ", length(moments[1:tot_moments]))
                converged, w, x, principal_diag =
                    compute_upper_principal_representation(
                        dict[1:tot_moments], moments[1:tot_moments], w2, x2;
                        verbose, config, diagnostics=true, options...)
                if !converged
                    converged, w, x, principal_diag =
                        compute_upper_principal_representation(
                            dict[1:tot_moments], moments[1:tot_moments], w, x;
                            verbose, config, diagnostics=true, options...)
                end
                xi = xi_extractor(x)
                upper_principal_index += 1
                _require_principal_convergence(
                    "upper principal representation $(upper_principal_index)",
                    xi, converged, principal_diag, principal_lost_digits_resolved;
                    verbose)
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
            elseif step.stage == :principal && step.branch == :lower
                @assert lower_canonical_state !== nothing
                w2, x2 = lower_canonical_state
                _gengauss_debug_println("DEBUG: x2: ", x2)
                _gengauss_debug_println("DEBUG: w2: ", w2)
                _gengauss_debug_println("DEBUG: len moments: ", length(moments[1:tot_moments]))
                converged, w, x, principal_diag =
                    compute_lower_principal_representation(
                        dict[1:tot_moments], moments[1:tot_moments], w2, x2;
                        verbose, config, diagnostics=true, options...)
                if !converged
                    converged, w, x, principal_diag =
                        compute_lower_principal_representation(
                            dict[1:tot_moments], moments[1:tot_moments], w, x;
                            verbose, config, diagnostics=true, options...)
                end
                xi = xi_extractor(x)
                lower_principal_index += 1
                _require_principal_convergence(
                    "lower principal representation $(lower_principal_index)",
                    xi, converged, principal_diag, principal_lost_digits_resolved;
                    verbose)
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
            end
        end
    end
    # Post-loop Gauss–Lobatto step.
    #
    # When the main loop runs to completion for even basis length, (w, x) holds
    # the lower principal of c^{n_dict} — the l-point Gauss-Legendre rule, which
    # we use BOTH as the basis for the bisection bracket and as a verification
    # of the moment match. principal=:upper for even basis length means we want
    # the Gauss-Lobatto rule = upper principal of the SAME moment vector:
    # an (l+1)-point rule with both endpoints fixed.
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
        verbose && println("  GL rule (LP of c^$(n)):  x = $(x),  w = $(w)")
        verbose && println("  Radau rule (c^$(n-1)):   x = $(last_radau_x),  w = $(last_radau_w)")
        converged_lob, w_lob, x_lob, lobatto_diag =
            compute_lobatto_step(dict, moments, w, x, last_radau_w, last_radau_x, config;
                verbose, max_adaptive_steps,
                canonical_lost_digits=canonical_lost_digits_resolved,
                lobatto_lost_digits=lobatto_lost_digits_resolved,
                options...)
        if converged_lob
            verbose && println("  Lobatto rule found:     x = $(x_lob),  w = $(w_lob)")
            push!(xi_checkpoints, xi_extractor(x_lob))
            push!(w_checkpoints, w_lob)
            push!(x_checkpoints, x_lob)
            return w_lob, x_lob, xi_checkpoints, w_checkpoints, x_checkpoints
        else
            lobatto_diag_str =
                lobatto_diag === nothing ?
                "unavailable (e.g. continuation ξ bracket not found before final Newton)." :
                _format_newton_diagnostic(lobatto_diag)
            lobatto_lost_digits_hint =
                lobatto_diag === nothing || (:error in keys(lobatto_diag)) ?
                "" :
                " Current effective lobatto_lost_digits=$(lobatto_lost_digits_resolved). " *
                "Increase lobatto_lost_digits above this value to accept the final residual if the lost digits are acceptable."
            println(
                "WARNING: compute_gauss_rules: Gauss-Lobatto post-loop failed ",
                "(n=$(n), add_endpoint=$(config.add_endpoint)): ",
                lobatto_diag_str,
                lobatto_lost_digits_hint,
                verbose ? "" : " Re-run with verbose=true for continuation/sweep details.",
            )
            println("Returning Gauss rule instead.")
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
