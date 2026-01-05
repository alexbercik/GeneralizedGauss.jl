
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

# Helper function to get direction from add_endpoint
get_direction(config::GaussRuleConfig) = config.add_endpoint == :right ? :right_to_left : :left_to_right

endpoint_value(dict, config::GaussRuleConfig) =
    config.add_endpoint == :right ? supportright(dict) : supportleft(dict)

pivot_value(x, config::GaussRuleConfig) =
    config.add_endpoint == :right ? x[1] : x[end]


# TODO: delete this too?
function upper_principal_rule(dict, moments, config::GaussRuleConfig)
    config.add_endpoint == :right ?
        UpperPrincipalEven(dict, moments) :
        LowerPrincipalEven(dict, moments)
end

default_threshold(dict::Dictionary) = default_threshold(codomaintype(dict))
default_threshold(::Type{T}) where {T <: AbstractFloat} = sqrt(eps(T))/100
default_threshold(::Type{BigFloat}) = big(1e-20)

solver_tolerance(::Type{Float64}) = 1e-8
solver_tolerance(::Type{T}) where {T} = sqrt(eps(T))
solver_tolerance(::Type{BigFloat}) = BigFloat(1e-30)

struct RepresentationStep
    branch::Symbol
    stage::Symbol
    fixed_endpoint::Union{Symbol,Nothing}
    k::Int
end

function default_representation_steps(principal::Symbol, add_endpoint::Symbol, even_basis_length::Bool)

    if principal == :lower && add_endpoint == :right && even_basis_length
        # Start from one-point LG rule, follow:
        # Add (fixed) right endpoint and explore upper canonical representations
        # Pop into upper principal representation for next moment (right-Radau)
        # Slide right endpoint down to explore lower canonical representations
        # Pop into lower principal representation for next moment (LG)
        steps = RepresentationStep[
        RepresentationStep(:upper, :canonical, :right, 1),
        RepresentationStep(:upper, :principal, :right, 1),
        RepresentationStep(:lower, :canonical, nothing, 2),
        RepresentationStep(:lower, :principal, nothing, 2),
        ]
    elseif principal == :lower && add_endpoint == :left && even_basis_length
        # Start from one-point LG rule, follow:
        # Add (fixed) left endpoint and explore lower canonical representations
        # Pop into lower principal representation for next moment (left-Radau)
        # Slide right endpoint down to explore lower canonical representations
        # Pop into lower principal representation for next moment (LG)
        steps = RepresentationStep[
        RepresentationStep(:lower, :canonical, :left, 1),
        RepresentationStep(:lower, :principal, :left, 1),
        RepresentationStep(:lower, :canonical, nothing, 2),
        RepresentationStep(:lower, :principal, nothing, 2),
        ]
    else
        error("TODO: Unknown combination of principal, add_endpoint, and even_basis_length")
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

function solve_system(rule, w0, x0; verbose=false, options...)
    x_init = quad_to_newton(rule, w0, x0)
    F!(Fx, x) = residual!(Fx, rule, x)
    J!(Jx, x) = jacobian!(Jx, rule, x)

    tol = solver_tolerance(eltype(x_init))
    r = nlsolve(F!, J!, x_init; ftol = tol, options...)
    w, x = newton_to_quad(rule, r.zero)
    converged(r), w, x
end

function supportleft(dict)
    t = leftendpoint(support(dict))
    isinf(t) ? -one(t)*10 : t
end
supportright(dict) = rightendpoint(support(dict))

function compute_one_point_rule(dict, moments; config::GaussRuleConfig=GaussRuleConfig(), options...)
    @assert length(dict) == 2
    @assert length(moments) == 2

    x0 = 1/2 * (supportleft(dict) + supportright(dict))
    w0 = moments[1] / eval_element(dict, 1, x0)
    # For the one-point rule, we want to always use LowerPrincipalOdd (free point, not fixed endpoints)
    # regardless of config.add_endpoint, so create a temporary config that forces this
    one_point_config = GaussRuleConfig(; principal=config.principal, add_endpoint=:right)
    converged, w, x = compute_lower_principal_representation(dict, moments, w0, x0; config=one_point_config, options...)
    @assert converged
    w, x
end

function compute_two_point_rule(dict, moments; tol = 1e-12, options...)
    @assert length(dict) == 2
    @assert length(moments) == 2

    a = supportleft(dict)
    b = supportright(dict)
    x = [a, b]

    u0a = eval_element(dict, 1, a)
    u0b = eval_element(dict, 1, b)
    u1a = eval_element(dict, 2, a)
    u1b = eval_element(dict, 2, b)

    denom = u0a * u1b - u0b * u1a

    if isapprox(denom, 0.0; atol = tol, rtol = 0.0)
        error("Degenerate system: u₀(a)u₁(b) - u₀(b)u₁(a) ≈ 0.\n" *
              "The functions are linearly dependent on {a,b}, so a unique 2-point rule does not exist.")
    end

    wa = (moments[1] * u1b - moments[2] * u0b) / denom
    wb = (moments[2] * u0a - moments[1] * u1a) / denom
    w = [wa, wb]
    w, x
end


function compute_upper_canonical_representation(dict, moments, xi, w0, x0;
        verbose=false, config::GaussRuleConfig=GaussRuleConfig(), options...)
    @assert length(moments) == length(dict)

    println("DEBUG: compute_upper_canonical_representation, even number of basis functions, (", length(dict), ")")

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
        false, w0, x0
    end
end


function compute_many_upper_canonical_representation(dict, moments, w0, x0, pts;
        verbose=false, config::GaussRuleConfig=GaussRuleConfig(), options...)

    n = length(pts)
    T = eltype(w0)
    w = zeros(T,length(w0),n)
    x = zeros(T,length(w0),n)
    hasconverged = zeros(Bool,n)
    order = sweep_indices(n, get_direction(config))
    prev_idx = nothing
    println("w0: ", w0)
    println("x0: ", x0)
    println("pts: ", pts)
    println("order: ", order)
    w_prev, x_prev = w0, x0
    for i in order
        xi = pts[i]
        if prev_idx !== nothing
            w_prev = w[:,prev_idx]
            x_prev = x[:,prev_idx]
        end
        converged, w1, x1 = compute_upper_canonical_representation(dict, moments, xi, w_prev, x_prev; verbose, config, options...)
        if !converged && verbose
            println("Many upper canonical: not converged for $(xi)")
        end
        w[:,i] = w1
        x[:,i] = x1
        println("w1: ", w1)
        println("x1: ", x1)
        hasconverged[i] = converged
        prev_idx = i
    end
    I = findall(hasconverged)
    any(hasconverged), w[:,I], x[:,I], pts[I]
end

function estimate_upper_canonical_representation(dict, moments, a, b, w0, x0;
        verbose=false, config::GaussRuleConfig=GaussRuleConfig(), options...)
    @assert isodd(length(dict))
    @assert length(dict) == length(moments)
    l = (length(dict)-1) >> 1
    @assert length(w0) == l+1
    @assert length(x0) == l+1

    verbose && println("Estimating upper canonical representation, xi between $(a) and $(b)")
    n = 8
    estimate_upper_canonical_representation(dict, moments, a, b, w0, x0, n;
        verbose, config, options...)
end

function interpolate_starting_values(a, b, p, w_left, x_left, w_right, x_right)
    θ = (p-a)/(b-a)
    w0 = w_left + θ * (w_right-w_left)
    x0 = x_left + θ * (x_right-x_left)
    w0, x0
end

switches_sign(values) = ! (all(values .> 0) || all(values .< 0))

function estimate_upper_canonical_representation(dict, moments, a, b, w0, x0, n;
        verbose=false, config::GaussRuleConfig=GaussRuleConfig(), options...)
    @assert n <= 1024

    pts = collect(range(a, b, length=n+2)[2:end-1])
    # end_seed will be computed from refine_interval_from_sweep results if needed
    end_seed = nothing

    some_converged, w, x, pts2 = compute_many_upper_canonical_representation(dict[1:end-1], moments[1:end-1], w0, x0, pts; verbose, config, options...)
    if some_converged
        if verbose && length(pts2) < length(pts)
            println("Upper canonical: converged for $(length(pts2)) out of $(length(pts)) points")
        end
        Fvals = [Fk(w[:,i],x[:,i],dict[end], moments[end]) for i in 1:size(w,2)]
        order = sweep_indices(length(Fvals), get_direction(config))
        Fvals_sweep = Fvals[order]
        if verbose && !ismonotonic(Fvals_sweep)
            println("Upper canonical: function Fk is not monotonic along $(get_direction(config)) sweep but should be.")
        end
        @assert ismonotonic(Fvals_sweep) "Upper canonical Fk sweep $(get_direction(config)) should be monotonic."
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
                refine_interval_from_sweep(a, b, pts2, w, x, order, Fvals_sweep, get_direction(config), (w0, x0), end_seed)
            verbose && println("Upper canonical: refining from $((a,b)) to $((a_new,b_new)) in direction $(get_direction(config))")
            # Select the appropriate boundary based on add_endpoint
            w0_new = config.add_endpoint == :left ? wa_new : wb_new
            x0_new = config.add_endpoint == :left ? xa_new : xb_new
            estimate_upper_canonical_representation(dict, moments, a_new, b_new, w0_new, x0_new, n; verbose, config, options...)
        end
    else
        verbose && println("Upper canonical: increasing n to $(2n)")
        estimate_upper_canonical_representation(dict, moments, a, b, w0, x0, 2n; verbose, config, options...)
    end
end

function compute_upper_principal_representation(dict, moments, w0, x0;
        verbose=false, config::GaussRuleConfig=GaussRuleConfig(), options...)
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
        println("ERROR THROWN in computation of upper principal")
        @show e
        false, w0, x0
    end
end

function compute_lower_canonical_representation(dict, moments, xi, w0, x0;
        verbose=false, config::GaussRuleConfig=GaussRuleConfig(), options...)
    @assert length(moments) == length(dict)

    if isodd(length(dict))
        println("DEBUG: compute_lower_canonical_representation, odd number of basis functions, (", length(dict), ")")
        # odd number of (n+1) basis functions (even n)
        @assert length(w0) == (length(dict)+1)>>1
        fixed_idx = [config.add_endpoint == :right ? 1 : length(w0)]
        rule = CanonicalRepresentationEven_J1(dict, xi, moments, fixed_idx)
    else
        println("DEBUG: compute_lower_canonical_representation, even number of basis functions, (", length(dict), ")")
        # even number of (n+1) basis functions (odd n)
        @assert length(w0) == (length(dict)>>1)+1
        if config.add_endpoint == :right
            error("TODO")
        else
            fixed_idx = [1,length(w0)]
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
        false, w0, x0
    end
end

function compute_many_lower_canonical_representation(dict, moments, w0, x0, pts;
        verbose=false, config::GaussRuleConfig=GaussRuleConfig(), options...)
    
    n = length(pts)
    T = eltype(w0)
    w = zeros(T,length(w0),n)
    x = zeros(T,length(w0),n)
    hasconverged = zeros(Bool,n)
    order = sweep_indices(n, get_direction(config))
    prev_idx = nothing
    println("w0: ", w0)
    println("x0: ", x0)
    println("pts: ", pts)
    println("order: ", order)
    w_prev, x_prev = w0, x0
    for i in order
        xi = pts[i]
        if prev_idx !== nothing
            w_prev = w[:,prev_idx]
            x_prev = x[:,prev_idx]
        end
        converged, w1, x1 =
            compute_lower_canonical_representation(dict, moments, xi, w_prev, x_prev;
                verbose, config, options...)
        if !converged && verbose
            println("Many lower canonical: not converged for $(xi)")
        end
        w[:,i] = w1
        x[:,i] = x1
        println("w1: ", w1)
        println("x1: ", x1)
        hasconverged[i] = converged
        prev_idx = i
    end
    I = findall(hasconverged)
    any(hasconverged), w[:,I], x[:,I], pts[I]
end

function estimate_lower_canonical_representation(dict, moments, a, b, w0, x0;
        verbose = false, config::GaussRuleConfig=GaussRuleConfig(), options...)
    @assert length(dict) == length(moments)
    l = length(dict) >> 1
    #@assert length(w0) == l
    #@assert length(x0) == l

    verbose && println("Estimating lower canonical representation, xi between $(a) and $(b)")
    n = 8
    estimate_lower_canonical_representation(dict, moments, a, b, w0, x0, n; verbose, config, options...)
end

function estimate_lower_canonical_representation(dict, moments, a, b, w0, x0, n;
        verbose=false, config::GaussRuleConfig=GaussRuleConfig(), options...)
    @assert n <= 1024

    pts = collect(range(a, b, length=n+2)[2:end-1])
    # end_seed will be computed from refine_interval_from_sweep results if needed
    end_seed = nothing
    someconverged, w, x, pts2 = compute_many_lower_canonical_representation(dict[1:end-1], moments[1:end-1], w0, x0, pts; verbose, config, options...)
    if someconverged
        if verbose && length(pts2)<length(pts)
            println("Lower canonical: converged for $(length(pts2)) out of $(length(pts)) points")
        end
        Fvals = [Fk(w[:,i],x[:,i],dict[end], moments[end]) for i in 1:size(w,2)]
        order = sweep_indices(length(Fvals), get_direction(config))
        Fvals_sweep = Fvals[order]
        if verbose && !ismonotonic(Fvals_sweep)
            println("Lower canonical: function Fk is not monotonic along $(get_direction(config)) sweep but should be.")
        end
        @assert ismonotonic(Fvals_sweep) "Lower canonical Fk sweep $(get_direction(config)) should be monotonic."
        if switches_sign(Fvals_sweep)
            println("DEBUG: Detected sign change in lower canonical")
            if first(Fvals_sweep) > 0
                I = findlast(Fvals_sweep .> 0)
            else
                I = findlast(Fvals_sweep .< 0)
            end
            left_idx = order[I]
            right_idx = order[I+1]
            println("DEBUG: Sign change detected at indices $(left_idx) and $(right_idx)")
            pts2[left_idx], pts2[right_idx], w[:,left_idx], x[:,left_idx], w[:,right_idx], x[:,right_idx]
        else
            println("DEBUG: No sign change detected, refining interval")
            a_new, b_new, wa_new, xa_new, wb_new, xb_new =
                refine_interval_from_sweep(a, b, pts2, w, x, order, Fvals_sweep, get_direction(config), (w0, x0), end_seed)
            verbose && println("Lower canonical: refining from $((a,b)) to $((a_new,b_new)) in direction $(get_direction(config))")
            # Select the appropriate boundary based on add_endpoint
            w0_new = config.add_endpoint == :left ? wa_new : wb_new
            x0_new = config.add_endpoint == :left ? xa_new : xb_new
            estimate_lower_canonical_representation(dict, moments, a_new, b_new, w0_new, x0_new, 8; verbose, config, options...)
        end
    else
        verbose && println("Lower canonical: no convergence on grid, increasing to n=$(2n)")
        estimate_lower_canonical_representation(dict, moments, a, b, w0, x0, 2n; verbose, config, options...)
    end
end

function compute_lower_principal_representation(dict, moments, w0, x0;
        verbose=false, config::GaussRuleConfig=GaussRuleConfig(), options...)
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
        println("ERROR THROWN in computation of lower principal")
        @show e
        false, w0, x0
    end
end


"""
    compute_gauss_rules(dict::Dictionary, moments = compute_moments(dict);
        verbose = false, principal = :lower, add_endpoint = :right, config = nothing, options...)

Compute the sequence of generalized Gaussian quadrature rules associated with
`dict`. The function returns the final Gauss rule together with all intermediate
principal representations. Keyword arguments:

- `verbose`: print progress information during the continuation process.
- `principal`: which principal representation to return as the final rule.
  - `:lower` (default): returns the lower principal representation (e.g. Gauss-Lobatto rule or right-anchored Radau).
  - `:upper`: returns the upper principal representation (e.g. Gauss rule or left-anchored Radau).
- `add_endpoint`: which endpoint to anchor during the computation.
  - `:right` (default): anchor at the right endpoint.
  - `:left`: anchor at the left endpoint.
- `options`: forwarded to the nonlinear solver used at every refinement step.

The function automatically determines whether to stop at an odd-length rule based on
the length of `dict`: if `dict` is even, it computes the full sequence; if `dict` is
odd, it stops at the odd-length rule.
"""
function compute_gauss_rules(dict::Dictionary, moments::Union{Nothing, Any} = nothing;
        verbose = false, principal::Symbol = :lower, add_endpoint::Symbol = :right,
        options...)
    n = length(dict)
    # Automatically determine stop_at_odd_gauss based on dict length
    # If dict is even, compute full sequence (stop_at_odd_gauss = false)
    # If dict is odd, stop at the odd-length rule (stop_at_odd_gauss = true)
    stop_at_odd_gauss = isodd(n)

    if isnothing(moments)
        moments = compute_moments(dict)
    end

    config = GaussRuleConfig(; principal, add_endpoint)
    steps = default_representation_steps(config.principal, config.add_endpoint, iseven(n))

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

    # TODO: Will need to adjust this when finding LGL rules
    verbose && println("Computing initial one point rule")
    w, x = compute_one_point_rule(dict[1:2], moments[1:2]; verbose, config, options...)
    verbose && println("One point quadrature rule is: ", x, ", ", w)

    # Add initial one-point rule as second checkpoint
    push!(xi_checkpoints, xi_extractor(x))
    push!(w_checkpoints, w)  # Store weights in w_checkpoints
    push!(x_checkpoints, x)  # Store nodes in x_checkpoints
    
    if n == 2
        return w, x, xi_checkpoints, w_checkpoints, x_checkpoints
    end

    upper_principal_index = 0
    lower_principal_index = 0
    upper_canonical_state = nothing
    lower_canonical_state = nothing
    last_upper_w = nothing
    last_upper_x = nothing

    k_max = iseven(n) ? l-1 : l
    for k = 1:k_max
        for step in steps
            println("k = ", k, ": ", step.branch, " ", step.stage)
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
                println("DEBUG: tot_moments: ", tot_moments)
                println("DEBUG: length(dict[1:tot_moments]): ", length(dict[1:tot_moments]))
                println("DEBUG: length(moments[1:tot_moments]): ", length(moments[1:tot_moments]))
                if step.branch == :upper
                    _, _, _, _, w2, x2 =
                        estimate_upper_canonical_representation(dict[1:tot_moments], moments[1:tot_moments], a, b, w0, x0; verbose, config, options...)
                        upper_canonical_state = (w2, x2)
                        lower_canonical_state = nothing
                elseif step.branch == :lower
                    _, _, _, _, w2, x2 =
                        estimate_lower_canonical_representation(dict[1:tot_moments], moments[1:tot_moments], a, b, w0, x0; verbose, config, options...)
                        upper_canonical_state = nothing
                        lower_canonical_state = (w2, x2)
                else
                    error("Unknown branch: ", step.branch)
                end
            elseif step.stage == :principal && step.branch == :upper
                @assert upper_canonical_state !== nothing
                w2, x2 = upper_canonical_state
                println("DEBUG: x2: ", x2)
                println("DEBUG: w2: ", w2)
                println("DEBUG: len moments: ", length(moments[1:tot_moments]))
                converged, w, x = compute_upper_principal_representation(dict[1:tot_moments], moments[1:tot_moments], w2, x2; verbose, config, options...)
                xi = xi_extractor(x)
                upper_principal_index += 1
                verbose && println("Upper principal representation ", upper_principal_index, " : xi is ", xi)
                verbose && println("    x: ", x)
                # Track last upper principal for final selection
                last_upper_w = w
                last_upper_x = x
                # Add to unified checkpoints (always add, not conditional on has_upper)
                push!(xi_checkpoints, xi)
                push!(w_checkpoints, w)  # Store weights in w_checkpoints
                push!(x_checkpoints, x)  # Store nodes in x_checkpoints
                upper_canonical_state = nothing

                if stop_target_len !== nothing && (2*k+1 == stop_target_len)
                    return w, x, xi_checkpoints, w_checkpoints, x_checkpoints
                end

                if config.principal == :upper && iseven(n) && k == l-1
                    return w, x, xi_checkpoints, w_checkpoints, x_checkpoints
                end
            elseif step.stage == :principal && step.branch == :lower
                @assert lower_canonical_state !== nothing
                w2, x2 = lower_canonical_state
                println("DEBUG: x2: ", x2)
                println("DEBUG: w2: ", w2)
                println("DEBUG: len moments: ", length(moments[1:tot_moments]))
                converged, w, x = compute_lower_principal_representation(dict[1:tot_moments], moments[1:tot_moments], w2, x2; verbose, config, options...)
                xi = xi_extractor(x)
                lower_principal_index += 1
                verbose && println("Lower principal representation ", lower_principal_index, " : xi is ", xi)
                verbose && println("    x: ", x)
                # Add to unified checkpoints (always add, not conditional on has_lower)
                push!(xi_checkpoints, xi)
                push!(w_checkpoints, w)  # Store weights in w_checkpoints
                push!(x_checkpoints, x)  # Store nodes in x_checkpoints
                lower_canonical_state = nothing
            end
        end
    end
    # If principal is :upper, return the last upper principal rule
    if config.principal == :upper && last_upper_w !== nothing
        w = last_upper_w
        x = last_upper_x
    end
    w, x, xi_checkpoints, w_checkpoints, x_checkpoints
end

"""
    compute_gauss_rule(dict::Dictionary, moments = compute_moments(dict);
        kwargs...)

Convenience wrapper that returns only the terminal quadrature rule produced by
`compute_gauss_rules`. All keyword arguments are forwarded.
"""
function compute_gauss_rule(dict::Dictionary, moments = compute_moments(dict); kwargs...)
    w, x, xi_checkpoints, w_checkpoints, x_checkpoints =
        compute_gauss_rules(dict, moments; kwargs...)
    # The final rule is already returned as w, x from compute_gauss_rules
    # (it handles config.principal == :upper selection internally)
    w, x
end
