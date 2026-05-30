@inline same_sign(a, b) =
    (a > zero(a) && b > zero(b)) || (a < zero(a) && b < zero(b))

@inline function interior(a, b, x)
    # Move x into the strict interior (a,b).
    # This avoids ever evaluating f(a) or f(b).
    x = min(max(x, a), b)
    x <= a && return nextfloat(a)
    x >= b && return prevfloat(b)
    return x
end

@inline side_dx(a, b, x0, n, side::Symbol) =
    side === :left ? (x0 - a)/(n + 1) : (b - x0)/(n + 1)

@inline side_step(x0, dx, side::Symbol, k) =
    side === :left ? x0 - k*dx : x0 + k*dx

@inline other_side(side::Symbol) = side === :left ? :right : :left

function try_side(f, a, b, x0, fx0, n, side::Symbol)
    # Search only one side of x0 for a sign change.
    # side = :left  searches from x0 toward a.
    # side = :right searches from x0 toward b.
    dx = side_dx(a, b, x0, n, side)

    xprev, fprev = x0, fx0

    for k in 1:n
        x = side_step(x0, dx, side, k)

        fx = f(x)
        fx == zero(fx) && return (:root, x, fx)

        if !same_sign(fprev, fx)
            return side === :left ?
                (:bracket, (x, fx, xprev, fprev)) :
                (:bracket, (xprev, fprev, x, fx))
        end

        xprev, fprev = x, fx
    end

    return nothing # -- no sign change found
end

"""
Find a sign-changing bracket on the interval (a, b) containing the initial guess x0.
# Arguments
- `f`: The callable scalar function to find the root of.
- `a`: The left endpoint of the interval.
- `b`: The right endpoint of the interval.
- `x0`: The initial guess for the root. Assumed to be in the interior of the interval (a, b).
- `n_samples`: The number of samples to use to find the bracket.
"""
function _find_interior_sign_bracket(
    f, a, b, x0;
    n_samples::Int = 48,
)
    a < b || throw(ArgumentError("requires a < b"))
    n_samples >= 1 || throw(ArgumentError("n_samples must be positive"))

    fx0 = f(x0)
    fx0 == zero(fx0) && return (:root, x0, fx0)

    # Cheap direction check.
    # We assume f is monotone, but do not know whether it is increasing
    # or decreasing. Probe the first try_side step on each side.
    xL = side_step(x0, side_dx(a, b, x0, n_samples, :left), :left, 1)
    xR = side_step(x0, side_dx(a, b, x0, n_samples, :right), :right, 1)

    fL = f(xL)
    fR = f(xR)

    fL == zero(fL) && return (:root, xL, fL)
    fR == zero(fR) && return (:root, xR, fR)

    # -- if we already have a sign change, return the bracket
    !same_sign(fL, fx0) && return (:bracket, (xL, fL, x0, fx0))
    !same_sign(fx0, fR) && return (:bracket, (x0, fx0, xR, fR))

    # -- determine the primary side to search on
    primary = if fL < fR # -- increasing function
        fx0 > zero(fx0) ? :left : :right # -- if fx0 is positive, the root is to the left
    elseif fL > fR       # -- decreasing function
        fx0 < zero(fx0) ? :left : :right # -- if fx0 is negative, the root is to the left
    else
        :left # -- default to left if we cannot determine the monotonicity
    end

    for side in (primary, other_side(primary))
        result = try_side(f, a, b, x0, fx0, n_samples, side)
        result !== nothing && return result
    end

    error("Could not find an interior sign-changing bracket on ($a, $b).")
end

function _update_bracket(bracket, x1, f1, x2, f2)
    if bracket === nothing
        # Need opposite signs to establish a root bracket initially.
        same_sign(f1, f2) && return nothing
        xa, xb = x1 < x2 ? (x1, x2) : (x2, x1)
        fa, fb = x1 < x2 ? (f1, f2) : (f2, f1)
        return (xa, fa, xb, fb)
    end
    # Once bracketed, tighten using the new trial point alone: same sign as an
    # endpoint replaces that endpoint even if consecutive iterates share a sign.
    bx_a, bf_a, bx_b, bf_b = bracket
    if same_sign(bf_a, f2)
        # -- if the new f2 has the same sign as the left endpoint, replace the left endpoint
        return (x2, f2, bx_b, bf_b)
    else
        # -- otherwise, there is a sign change, so replace the right endpoint
        return (bx_a, bf_a, x2, f2)
    end
end

"""
A scalar root solver that uses the Brent method to find a root of a function.
# Arguments
- `f`: The callable scalar function to find the root of.
- `a`: The left endpoint of the interval.
- `b`: The right endpoint of the interval.
- `fa`: The function value at the left endpoint. Assumed to be non-zero.
- `fb`: The function value at the right endpoint. Assumed to be non-zero.
- `x_tol`: The tolerance for the root.
- `f_tol`: The tolerance for the function value.
- `maxiter`: The maximum number of iterations.
"""
function _brent_root(f, a, b, fa, fb; x_tol, f_tol, maxiter::Int=500, kwargs...)
    fa == zero(fa) && return true, a, fa
    fb == zero(fb) && return true, b, fb

    isfinite(fa) && isfinite(fb) ||
        error("Brent solver requires finite endpoint values.")

    same_sign(fa, fb) &&
        error("Brent solver requires a sign-changing bracket.")

    # Brent keeps b as the current best estimate.
    # The interval [b,c] is the active sign-changing bracket.
    c = a
    fc = fa

    # d is the proposed step, e is the previous accepted step.
    d = b - a
    e = d

    for _ in 1:maxiter
        # Ensure that b and c continue to bracket the root.
        if same_sign(fb, fc)
            c = a
            fc = fa
            d = b - a
            e = d
        end

        # Make b the endpoint with the smaller residual.
        if abs(fc) < abs(fb)
            a = b
            b = c
            c = a

            fa = fb
            fb = fc
            fc = fa
        end

        # Stopping tolerance in x.
        #
        # eps(b) is the local spacing near b, so this works naturally for
        # Float64, BigFloat, etc., without explicitly specifying a type.
        tol = 2 * eps(b) + x_tol / 2
        xm = (c - b) / 2

        if abs(fb) <= f_tol
            return true, b, fb
        end
        if abs(xm) <= tol
            return false, b, fb
        end

        if abs(e) >= tol && abs(fa) > abs(fb)
            # Attempt inverse quadratic interpolation.
            # If a == c, this reduces to the secant method.
            s = fb / fa

            if a == c
                p = 2 * xm * s
                q = one(s) - s
            else
                q = fa / fc
                r = fb / fc
                p = s * (2 * xm * q * (q - r) - (b - a) * (r - one(r)))
                q = (q - one(q)) * (r - one(r)) * (s - one(s))
            end

            # Make p positive. The sign is carried by q.
            if p > zero(p)
                q = -q
            else
                p = -p
            end

            # Accept interpolation only if it is sufficiently safe.
            if 2 * p < min(3 * xm * q - abs(tol * q), abs(e * q))
                e = d
                d = p / q
            else
                d = xm
                e = d
            end
        else
            # Fall back to bisection.
            d = xm
            e = d
        end

        # Move b, but by at least tol.
        a = b
        fa = fb

        b += abs(d) > tol ? d : (xm >= zero(xm) ? tol : -tol)
        fb = f(b)

        isfinite(fb) || error("Brent solver encountered a non-finite function value.")
    end

    return false, b, fb
end

function _brent_root_on_interval(f, a, b, x0; x_tol, f_tol, maxiter::Int=500,
        bracket=nothing, n_samples::Int=48, kwargs...)
    if bracket === nothing
        result = _find_interior_sign_bracket(f, a, b, x0; n_samples)
        if result[1] == :root # -- got lucky and found a root immediately
            _, x, fx = result
            return true, x, fx
        end
        bracket = result[2]
    end
    xa, fa, xb, fb = bracket
    return _brent_root(f, xa, xb, fa, fb; x_tol, f_tol, maxiter)
end

function _warn_scalar_newton_brent_fallback(verbose, reason)
    verbose || return
    println("WARNING: scalar Newton $(reason); falling back to Brent root finding.")
end

"""
A scalar root solver that uses a safeguarded Newton method to find a root of a function.

# Arguments
- `f`: The callable scalar function to find the root of.
- `df`: The callable scalar derivative of f w.r.t. x.
- `a`: The left endpoint of the interval.
- `b`: The right endpoint of the interval.
- `x`: The initial guess for the root.
- `x_tol`: The tolerance for the root.
- `f_tol`: The tolerance for the function value.
- `maxiter`: The maximum number of iterations.
- `verbose`: print warnings when falling back to Brent.

# Returns
- `(converged, x, fx)`: A tuple containing a boolean indicating whether the solver converged, the root, and the function value at the root.
"""
function _scalar_newton_root(f, df, a, b, x; x_tol, f_tol,
        maxiter::Int=500, verbose=false, kwargs...)
    fx = f(x)
    # -- Check if it is already a root
    if fx == zero(fx) || abs(fx) <= f_tol
        return true, x, fx
    end

    bracket = nothing # -- initialize to nothing to avoid evaluating f at the endpoints
    newton_left, newton_right = a, b # -- physical bracket for Newton iterations

    for _ in 1:maxiter
        dfx = df(x)
        if dfx == zero(dfx) || !isfinite(dfx)
            _warn_scalar_newton_brent_fallback(verbose,
                "encountered a zero or non-finite derivative at x=$(x)")
            return _brent_root_on_interval(f, a, b, x;
                x_tol, f_tol, maxiter, bracket)
        end

        x_new = x - fx / dfx # Newton step
        if !(newton_left < x_new < newton_right)
            _warn_scalar_newton_brent_fallback(verbose,
                "step x_new=$(x_new) left bracket ($(newton_left), $(newton_right))")
            return _brent_root_on_interval(f, a, b, x;
                x_tol, f_tol, maxiter, bracket)
        end

        f_new = f(x_new)
        bracket = _update_bracket(bracket, x, fx, x_new, f_new)
        if bracket !== nothing
            newton_left, newton_right = bracket[1], bracket[3]
        end
        if abs(f_new) > abs(fx)
            _warn_scalar_newton_brent_fallback(verbose,
                "residual worsened (|f(x_new)|=$(abs(f_new)), |f(x)|=$(abs(fx)))")
            return _brent_root_on_interval(f, a, b, x;
                x_tol, f_tol, maxiter, bracket)
        end

        # -- Check for convergence
        if abs(f_new) <= f_tol
            return true, x_new, f_new
        end
        if abs(x_new - x) <= x_tol
            return false, x_new, f_new
        end

        x = x_new
        fx = f_new
    end

    # -- brent fallback if the maximum number of iterations is reached without convergence
    _warn_scalar_newton_brent_fallback(verbose,
        "reached maxiter=$(maxiter) without converging")
    return _brent_root_on_interval(f, a, b, x;
        x_tol, f_tol, maxiter, bracket)
end
