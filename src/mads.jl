_node_variable_index(rule::QuadRuleFreePoints, node_index::Int) =
    quadlength(rule) + node_index

function _node_variable_index(rule::QuadRuleFixedPoints, node_index::Int)
    free_index = findfirst(==(node_index), rule.free_idxs)
    free_index === nothing &&
        error("MADS cannot bound fixed node index $(node_index).")
    quadlength(rule) + free_index
end

function _mads_support_bounds(rule)
    a = leftendpoint(support(basis(rule)))
    b = rightendpoint(support(basis(rule)))
    isfinite(a) && isfinite(b) ||
        error("MADS requires a finite basis support interval.")
    a < b || error("MADS requires a non-empty basis support interval.")
    isfinite(Float64(a)) && isfinite(Float64(b)) ||
        error("MADS requires support bounds representable as Float64.")
    a, b
end

function _mads_rule_is_valid(rule, w, x)
    all(isfinite, w) && all(isfinite, x) || return false
    a, b = _mads_support_bounds(rule)
    a <= first(x) && last(x) <= b || return false
    all(i -> x[i] < x[i + 1], 1:(length(x) - 1))
end

function _mads_objective(rule, w, x)
    value = zero(eltype(w))
    for j in 1:length(basis(rule))
        moment_error = -moments(rule)[j]
        for i in eachindex(w)
            moment_error += w[i] * funeval(basis(rule), j, x[i])
        end
        value += abs2(moment_error)
    end
    value
end

function _mads_canonical_initial_mesh(rule, w, dx)
    a, b = _mads_support_bounds(rule)
    support_width = Float64(b - a)
    isfinite(support_width) && support_width > 0 ||
        error("MADS requires a support width representable as Float64.")
    node_mesh = max(abs(Float64(dx)), eps(Float64))
    weight_reference = max(maximum(abs, Float64.(w)), eps(Float64))
    weight_mesh = max(node_mesh / support_width * weight_reference, eps(Float64))
    isfinite(node_mesh) && isfinite(weight_mesh) ||
        error("MADS requires initial mesh sizes representable as Float64.")
    vcat(fill(weight_mesh, length(w)),
        fill(node_mesh, dofs(rule) - length(w)))
end

function _mads_initial_mesh(rule, w, dx, bracket)
    bracket === nothing && return _mads_canonical_initial_mesh(rule, w, dx)

    left = quad_to_newton(rule, bracket.left.w, bracket.left.x)
    right = quad_to_newton(rule, bracket.right.w, bracket.right.x)
    mesh = abs.(Float64.(right .- left))
    fallback = _mads_canonical_initial_mesh(rule, w, bracket.width)
    for i in eachindex(mesh)
        if !isfinite(mesh[i]) || mesh[i] <= eps(Float64)
            mesh[i] = fallback[i]
        end
        mesh[i] = max(mesh[i], eps(Float64))
    end
    mesh
end

function _mads_variable_bounds(rule, bracket)
    _mads_support_bounds(rule)
    lower = fill(-Inf, dofs(rule))
    upper = fill(Inf, dofs(rule))

    # Only the released xi node gets a hard bound. Support and strict ordering
    # remain extreme-barrier constraints because they apply to every node and
    # ordering couples neighboring nodes.
    if bracket !== nothing && !iszero(bracket.width)
        variable_index = _node_variable_index(rule, bracket.xi_index)
        xi_lower = Float64(bracket.left.xi)
        xi_upper = Float64(bracket.right.xi)
        # A BigFloat bracket can collapse after conversion. In that case the
        # Float64 backend cannot enforce a meaningful hard bound.
        if xi_lower < xi_upper
            lower[variable_index] = xi_lower
            upper[variable_index] = xi_upper
        end
    end
    lower, upper
end

_mads_solver_tolerance(::Type{T}) where {T} =
    max(solver_tolerance(T), T(10) * T(eps(Float64)))

function _mads_trial_outputs(rule, newton_x_float, ::Type{T};
        objective_scale::Float64=1.0) where {T}
    newton_x = T.(newton_x_float)
    w, x = newton_to_quad(rule, newton_x)
    if !_mads_rule_is_valid(rule, w, x)
        return true, true, [Inf, 1.0]
    end

    # NOMAD coordinates and outputs are Float64. Evaluate the basis and direct
    # sum-of-squares objective in the rule's working type before conversion.
    # A constant scale keeps small objectives visible to NOMAD without changing
    # the minimizer or the ordering of candidate rules.
    true, true, [objective_scale * Float64(_mads_objective(rule, w, x)), -1.0]
end

function _solve_system_mads(rule, w0, x0; dx=nothing, bracket=nothing,
        verbose=false, max_bb_eval::Int=5000,
        min_mesh_size::Float64=eps(Float64))
    T = promote_type(eltype(w0), eltype(x0))
    x_init = quad_to_newton(rule, w0, x0)
    tol = _mads_solver_tolerance(T)
    w_init, x_quad_init = newton_to_quad(rule, x_init)

    # Continuation often supplies an already acceptable rule. Avoid invoking
    # NOMAD in that case, especially when the canonical bracket has collapsed.
    if _mads_rule_is_valid(rule, w_init, x_quad_init)
        diag = _solver_diagnostic(rule, x_init, tol)
        diag.residual_norm <= tol && return true, w_init, x_quad_init, diag
    end

    initial_mesh = _mads_initial_mesh(rule, w0,
        dx === nothing ? (_mads_support_bounds(rule)[2] -
                          _mads_support_bounds(rule)[1]) /
                         _CANONICAL_SWEEP_INITIAL_SUBDIVISIONS : dx,
        bracket)
    physical_lower, physical_upper =
        _mads_variable_bounds(rule, bracket)
    callback_error = Ref{Any}(nothing)
    eval_count = Ref(0)
    objective_scale = inv(Float64(tol))^2
    isfinite(objective_scale) ||
        error("MADS requires a finite Float64 objective scale.")
    center = Float64.(x_init)
    all(isfinite, center) ||
        error("MADS requires initial coordinates representable as Float64.")
    physical_mesh = initial_mesh
    last_w, last_x, last_diag = w_init, x_quad_init,
        _solver_diagnostic(rule, x_init, tol)

    # NOMAD's mesh is expressed in local coordinates around the latest feasible
    # rule. The first physical mesh is exactly the continuation mesh above.
    # Re-centering lets the Float64 backend polish the same free variables near
    # a solution while sharing one evaluation budget across all local scales.
    for _ in 1:6
        remaining_evals = max_bb_eval - eval_count[]
        remaining_evals > 0 || break
        local_lower = (physical_lower .- center) ./ physical_mesh
        local_upper = (physical_upper .- center) ./ physical_mesh

        function blackbox(local_x_float)
            eval_count[] += 1
            newton_x_float = center .+ physical_mesh .* local_x_float
            try
                return _mads_trial_outputs(rule, newton_x_float, T;
                    objective_scale)
            catch e
                callback_error[] === nothing && (callback_error[] = e)
                return false, true, [Inf, 1.0]
            end
        end

        problem = NOMAD.NomadProblem(dofs(rule), 2, ["OBJ", "EB"], blackbox;
            lower_bound=local_lower, upper_bound=local_upper,
            initial_mesh_size=ones(dofs(rule)),
            min_mesh_size=fill(min_mesh_size, dofs(rule)))
        problem.options.display_degree = verbose ? 2 : 0
        problem.options.max_bb_eval = remaining_evals
        problem.options.direction_type = "ORTHO N+1 NEG"
        problem.options.eval_queue_sort = "DIR_LAST_SUCCESS"
        problem.options.quad_model_search = false
        problem.options.sgtelib_model_search = false
        problem.options.speculative_search = false
        problem.options.nm_search = false
        problem.options.seed = 0

        result = NOMAD.solve(problem, zeros(dofs(rule)))
        callback_error[] === nothing || throw(callback_error[])
        result.x_best_feas === nothing &&
            error("NOMAD did not return a feasible quadrature rule.")

        center = center .+ physical_mesh .* result.x_best_feas
        newton_x = T.(center)
        last_w, last_x = newton_to_quad(rule, newton_x)
        _mads_rule_is_valid(rule, last_w, last_x) ||
            error("NOMAD returned an invalid quadrature rule.")
        last_diag = _solver_diagnostic(rule, newton_x, tol)
        last_diag.residual_norm <= tol &&
            return true, last_w, last_x, last_diag
        physical_mesh ./= 1000
    end

    false, last_w, last_x, last_diag
end
