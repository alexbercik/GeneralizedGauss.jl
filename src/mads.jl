_node_variable_index(rule::QuadRuleFreePoints, node_index::Int) =
    node_index

function _node_variable_index(rule::QuadRuleFixedPoints, node_index::Int)
    free_index = findfirst(==(node_index), rule.free_idxs)
    free_index === nothing &&
        error("MADS cannot bound fixed node index $(node_index).")
    free_index
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

function _mads_nodes_are_valid(x, a, b)
    for xi in x
        isfinite(xi) || return false
    end
    a <= first(x) && last(x) <= b || return false
    for i in 1:(length(x) - 1)
        x[i] < x[i + 1] || return false
    end
    true
end

function _mads_rule_is_valid(w, x, a, b)
    for wi in w
        isfinite(wi) || return false
    end
    _mads_nodes_are_valid(x, a, b)
end

function _mads_node_variables(rule::QuadRuleFreePoints, x)
    # NOMAD coordinates contain only nodes; weights are projected separately.
    copy(x)
end

function _mads_node_variables(rule::QuadRuleFixedPoints, x)
    x[rule.free_idxs]
end

function _mads_nodes_to_quad!(rule::QuadRuleFreePoints, x, node_variables)
    copyto!(x, node_variables)
end

function _mads_nodes_to_quad!(rule::QuadRuleFixedPoints, x, node_variables)
    for (j, i) in enumerate(rule.free_idxs)
        x[i] = node_variables[j]
    end
    for (i, xi) in zip(rule.fixed_idxs, rule.fixed_pts)
        x[i] = xi
    end
    x
end

function _mads_project_weights!(w, basis_matrix, rule, x)
    # For fixed nodes the moment equations are linear in the weights. Project
    # each MADS node trial onto its least-squares optimal weights so NOMAD only
    # has to search the genuinely nonlinear node variables.
    for j in 1:length(basis(rule))
        for i in eachindex(x)
            basis_matrix[j, i] = funeval(basis(rule), j, x[i])
        end
    end
    copyto!(w, basis_matrix \ moments(rule))

    objective = zero(eltype(w))
    for j in 1:length(basis(rule))
        moment_error = -moments(rule)[j]
        for i in eachindex(w)
            moment_error += w[i] * basis_matrix[j, i]
        end
        objective += abs2(moment_error)
    end
    objective
end

function _mads_canonical_initial_mesh(rule, dx)
    node_mesh = max(abs(Float64(dx)), eps(Float64))
    isfinite(node_mesh) ||
        error("MADS requires initial mesh sizes representable as Float64.")
    fill(node_mesh, dofs(rule) - quadlength(rule))
end

function _mads_initial_mesh(rule, dx, bracket)
    bracket === nothing && return _mads_canonical_initial_mesh(rule, dx)

    left = _mads_node_variables(rule, bracket.left.x)
    right = _mads_node_variables(rule, bracket.right.x)
    mesh = abs.(Float64.(right .- left))
    fallback = _mads_canonical_initial_mesh(rule, bracket.width)
    for i in eachindex(mesh)
        if !isfinite(mesh[i]) || mesh[i] <= eps(Float64)
            mesh[i] = fallback[i]
        end
        mesh[i] = max(mesh[i], eps(Float64))
    end
    mesh
end

function _mads_variable_bounds(rule, bracket)
    node_variable_count = dofs(rule) - quadlength(rule)
    lower = fill(-Inf, node_variable_count)
    upper = fill(Inf, node_variable_count)

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
    first(_resolve_solver_tolerances(T;
        tolerance_floor=T(10) * T(eps(Float64))))

const _MADS_LOCAL_MIN_MESH_SIZE = 1e-3

function _mads_trial_outputs!(outputs, rule, node_variables_float,
        node_variables, w, x, basis_matrix, support_bounds;
        objective_scale::Float64=1.0)
    copyto!(node_variables, node_variables_float)
    _mads_nodes_to_quad!(rule, x, node_variables)
    # Reject invalid nodes before evaluating any basis function. Weights are
    # generated only after this extreme-barrier check.
    if !_mads_nodes_are_valid(x, support_bounds...)
        outputs[1] = Inf
        outputs[2] = 1.0
        return true, true, outputs
    end

    # NOMAD coordinates and outputs are Float64. Evaluate the basis and direct
    # sum-of-squares objective in the rule's working type before conversion.
    # A constant scale keeps small objectives visible to NOMAD without changing
    # the minimizer or the ordering of candidate rules.
    objective = _mads_project_weights!(w, basis_matrix, rule, x)
    if !_mads_rule_is_valid(w, x, support_bounds...) || !isfinite(objective)
        outputs[1] = Inf
        outputs[2] = 1.0
        return true, true, outputs
    end
    outputs[1] = objective_scale * Float64(objective)
    outputs[2] = -1.0
    true, true, outputs
end

function _mads_trial_outputs(rule, node_variables_float, ::Type{T};
        objective_scale::Float64=1.0) where {T}
    node_variables = Vector{T}(undef, length(node_variables_float))
    w = Vector{T}(undef, quadlength(rule))
    x = similar(w)
    basis_matrix = Matrix{T}(undef, length(basis(rule)), quadlength(rule))
    outputs = Vector{Float64}(undef, 2)
    _mads_trial_outputs!(outputs, rule, node_variables_float, node_variables,
        w, x, basis_matrix, _mads_support_bounds(rule); objective_scale)
end

function _solve_system_mads(rule, w0, x0; dx=nothing, bracket=nothing,
        verbose=false, max_bb_eval::Int=5000,
        min_mesh_size::Float64=_MADS_LOCAL_MIN_MESH_SIZE,
        intermediate_tolerance=nothing)
    0 < min_mesh_size < 1 ||
        error("MADS local minimum mesh size must be between zero and one.")
    T = promote_type(eltype(w0), eltype(x0))
    x_init = quad_to_newton(rule, w0, x0)
    strict_tolerance, active_tolerance =
        _resolve_solver_tolerances(T, intermediate_tolerance;
            tolerance_floor=T(10) * T(eps(Float64)))
    w_init, x_quad_init = newton_to_quad(rule, x_init)
    support_bounds = _mads_support_bounds(rule)

    # Continuation often supplies an already acceptable rule. Avoid invoking
    # NOMAD in that case, especially when the canonical bracket has collapsed.
    if _mads_rule_is_valid(w_init, x_quad_init, support_bounds...)
        diag = _solver_diagnostic(rule, x_init, strict_tolerance,
            active_tolerance)
        diag.residual_norm <= active_tolerance &&
            return true, w_init, x_quad_init, diag
    end

    # Some Lobatto canonical steps fix every node. Variable projection reduces
    # those systems to a linear weight solve, so there is nothing for NOMAD to
    # search after the initial continuation rule has been checked.
    if dofs(rule) == quadlength(rule)
        basis_matrix = Matrix{T}(undef, length(basis(rule)), quadlength(rule))
        _mads_nodes_are_valid(x_quad_init, support_bounds...) ||
            error("MADS received invalid fixed quadrature nodes.")
        _mads_project_weights!(w_init, basis_matrix, rule, x_quad_init)
        _mads_rule_is_valid(w_init, x_quad_init, support_bounds...) ||
            error("MADS projected invalid quadrature weights.")
        projected_x = quad_to_newton(rule, w_init, x_quad_init)
        diag = _solver_diagnostic(rule, projected_x, strict_tolerance,
            active_tolerance)
        return diag.residual_norm <= active_tolerance, w_init, x_quad_init, diag
    end

    initial_mesh = _mads_initial_mesh(rule,
        dx === nothing ? (support_bounds[2] - support_bounds[1]) /
                         _CANONICAL_SWEEP_INITIAL_SUBDIVISIONS : dx,
        bracket)
    physical_lower, physical_upper =
        _mads_variable_bounds(rule, bracket)
    callback_error = Ref{Any}(nothing)
    eval_count = Ref(0)
    objective_scale = inv(Float64(active_tolerance))^2
    isfinite(objective_scale) ||
        error("MADS requires a finite Float64 objective scale.")
    center = Float64.(_mads_node_variables(rule, x0))
    all(isfinite, center) ||
        error("MADS requires initial coordinates representable as Float64.")
    physical_mesh = initial_mesh
    last_w, last_x, last_diag = w_init, x_quad_init,
        _solver_diagnostic(rule, x_init, strict_tolerance, active_tolerance)
    node_variables_float = zeros(Float64, length(center))
    node_variables = Vector{T}(undef, length(center))
    trial_w = Vector{T}(undef, quadlength(rule))
    trial_x = similar(trial_w)
    basis_matrix = Matrix{T}(undef, length(basis(rule)), quadlength(rule))
    trial_outputs = Vector{Float64}(undef, 2)

    # NOMAD's mesh is expressed in local coordinates around the latest feasible
    # rule. The first physical mesh is exactly the continuation mesh above.
    # Re-centering lets the Float64 backend polish the free nodes near a solution
    # while sharing one evaluation budget across all local scales.
    for _ in 1:6
        remaining_evals = max_bb_eval - eval_count[]
        remaining_evals > 0 || break
        local_lower = (physical_lower .- center) ./ physical_mesh
        local_upper = (physical_upper .- center) ./ physical_mesh

        function blackbox(local_x_float)
            eval_count[] += 1
            @. node_variables_float = center + physical_mesh * local_x_float
            try
                return _mads_trial_outputs!(trial_outputs, rule,
                    node_variables_float, node_variables, trial_w, trial_x,
                    basis_matrix, support_bounds; objective_scale)
            catch e
                callback_error[] === nothing && (callback_error[] = e)
                trial_outputs[1] = Inf
                trial_outputs[2] = 1.0
                return false, true, trial_outputs
            end
        end

        problem = NOMAD.NomadProblem(length(center), 2, ["OBJ", "EB"], blackbox;
            lower_bound=local_lower, upper_bound=local_upper,
            initial_mesh_size=ones(length(center)),
            min_mesh_size=fill(min_mesh_size, length(center)))
        problem.options.display_degree = verbose ? 2 : 0
        problem.options.max_bb_eval = remaining_evals
        problem.options.direction_type = "ORTHO N+1 NEG"
        problem.options.eval_queue_sort = "DIR_LAST_SUCCESS"
        problem.options.quad_model_search = false
        problem.options.sgtelib_model_search = false
        problem.options.speculative_search = false
        problem.options.nm_search = false
        problem.options.seed = 0

        result = NOMAD.solve(problem, zeros(length(center)))
        callback_error[] === nothing || throw(callback_error[])
        result.x_best_feas === nothing &&
            error("NOMAD did not return a feasible quadrature rule.")

        center = center .+ physical_mesh .* result.x_best_feas
        selected_nodes = T.(center)
        _mads_nodes_to_quad!(rule, last_x, selected_nodes)
        _mads_project_weights!(last_w, basis_matrix, rule, last_x)
        _mads_rule_is_valid(last_w, last_x, support_bounds...) ||
            error("NOMAD returned an invalid quadrature rule.")
        selected_x = quad_to_newton(rule, last_w, last_x)
        last_diag = _solver_diagnostic(rule, selected_x, strict_tolerance,
            active_tolerance)
        last_diag.residual_norm <= active_tolerance &&
            return true, last_w, last_x, last_diag
        # Continue at the physical resolution reached by this local solve.
        # This avoids asking NOMAD to polish every continuation step all the
        # way to Float64 epsilon in a single local coordinate system.
        physical_mesh .*= min_mesh_size
    end

    false, last_w, last_x, last_diag
end
