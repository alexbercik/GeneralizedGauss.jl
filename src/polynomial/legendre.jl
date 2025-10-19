
using BasisFunctions: PolynomialDegree

inversechristoffelfunction(onb) = sum(phi^2 for phi in onb)

"""
    extremal_polynomial(t, onb)

For a given orthonormal basis, the extremal polynomial is the expansion
```
K_t(x) = sum_k phi_k(t) phi_(x).
```
"""
extremal_polynomial(t, onb) = Expansion(onb, onb(t))

"""
    extremal_roots(t, onb)

For a given orthonormal basis, compute the roots of the extremal polynomial.
"""
extremal_roots(t, onb) = roots(extremal_polynomial(t, onb))

"""
The Gauss points associated with a fixed point `t` and an orthonormal basis, is
a set containing `t` as well as the roots of the extremal polynomial associated with `t`.
"""
gausspoints(t, onb) = sort([t; extremal_roots(t, onb)])

function legendre_principal_reps(n::Int)
    m = PolynomialDegree(n >> 1)
    @show m
    if iseven(n)
        Qlower = Jacobi(m, 0, 1)
        Qupper = Jacobi(m, 1, 0)
        xup = [roots(Qupper[m]); 1]
        xlo = [-1; roots(Qlower[m])]
    else
        Plower = Jacobi(m+1, 0, 0)
        Pupper = Jacobi(m, 1, 1)
        xup = [-1; roots(Pupper[m]); 1]
        xlo = roots(Plower[m+1])
    end

    kern = inversechristoffelfunction(normalize(Legendre(m)))
    wup = 1 ./ kern.(xup)
    wlo = 1 ./ kern.(xlo)

    if isodd(n)
        M = value(m)
        wup = wup * (M+1)/(M+2)
    end
    (wlo, xlo), (wup, xup)
end

function legendre_canonical_rep(n::Int, xi)
    m = PolynomialDegree(n >> 1)
    (wlo, xlo), (wup, xup) = legendre_principal_reps(n)
    # We distinguish between J and K intervals, see Karlin and Studden, Ch2, Sect5
    if iseven(n)
        idx = findlast(xi .>= xlo)
        if xi <= xup[idx]
            # the root xi is in a J-interval
            P_lower = normalize(Jacobi(m, 0, 0))
            kern = inversechristoffelfunction(P_lower)
            x = gausspoints(xi, P_lower)
            w = 1 ./ kern.(x)
        else
            # the root xi is in a K-interval
            P_upper = normalize(Jacobi(m-1, 1, 1))
            kern = inversechristoffelfunction(P_upper)
            x1 = gausspoints(xi, P_upper)
            w1 = 1 ./ (kern.(x1) .* (1 .+ x1) .* (1 .- x1))
            # wa and wb are found by requiring sum(w) = 2 and sum(x.*w) = 0
            wb = (2 - sum(w1) - sum(x1.*w1)) / 2
            wa = 2 - sum(w1) - wb
            x = [-1; x1; 1]
            w = [wa; w1; wb]
        end
    else
        idx = findlast(xi .>= xup)
        if idx <= length(xlo) && xi <= xlo[idx]
            # the root xi is in a K-interval
            Q_upper = normalize(Jacobi(m, 1, 0))
            kern = inversechristoffelfunction(Q_upper)
            x1 = gausspoints(xi, Q_upper)
            w1 = 1 ./ (kern.(x1) .* (1 .- x1))
            wb = 2 - sum(w1)
            x = [x1; 1]
            w = [w1; wb]
        else
            # the root xi is in a J-interval
            Q_lower = normalize(Jacobi(m, 0, 1))
            kern = inversechristoffelfunction(Q_lower)
            x1 = gausspoints(xi, Q_lower)
            w1 = 1 ./ (kern.(x1) .* (1 .+ x1))
            wa = 2 - sum(w1)
            x = [-1; x1]
            w = [wa; w1]
        end
    end
    w, x
end

function jacobi_principal_reps(n::Int, α, β)
    m = PolynomialDegree(n >> 1)
    if iseven(n)
        Qlower = Jacobi(m, α, β+1)
        Qupper = Jacobi(m, α+1, β)
        xup = [roots(Qupper[m]); 1]
        xlo = [-1; roots(Qlower[m])]
    else
        Plower = Jacobi(m+1, α, β)
        Pupper = Jacobi(m, α+1, β+1)
        xup = [-1; roots(Pupper[m]); 1]
        xlo = roots(Plower[m+1])
    end

    kern = inversechristoffelfunction(normalize(Jacobi(m, α, β)))
    wup = 1 ./ kern.(xup)
    wlo = 1 ./ kern.(xlo)

    if isodd(n)
        M = value(m)
        wup = wup * (M+1)/(M+2)
    end
    (wlo, xlo), (wup, xup)
end

function jacobi_canonical_rep(n::Int, α, β, xi)
    m = PolynomialDegree(n >> 1)
    (wlo, xlo), (wup, xup) = jacobi_principal_reps(n, α, β)
    μ0 = sum(wlo)
    μ1 = sum(wlo .* xlo)
    # We distinguish between J and K intervals, see Karlin and Studden, Ch2, Sect5
    if iseven(n)
        idx = findlast(xi .>= xlo)
        if xi <= xup[idx]
            # the root xi is in a J-interval
            P_lower = normalize(Jacobi(m, α, β))
            kern = inversechristoffelfunction(P_lower)
            x = gausspoints(xi, P_lower)
            w = 1 ./ kern.(x)
        else
            # the root xi is in a K-interval
            P_upper = normalize(Jacobi(m-1, α+1, β+1))
            kern = inversechristoffelfunction(P_upper)
            x1 = gausspoints(xi, P_upper)
            w1 = 1 ./ (kern.(x1) .* (1 .+ x1) .* (1 .- x1))
            # wa and wb are found by requiring sum(w) = μ0 and sum(x.*w) = μ1
            wb = (μ1 + μ0 - sum(w1) - sum(x1.*w1)) / 2
            wa = μ0 - sum(w1) - wb
            x = [-1; x1; 1]
            w = [wa; w1; wb]
        end
    else
        idx = findlast(xi .>= xup)
        if idx <= length(xlo) && xi <= xlo[idx]
            # the root xi is in a K-interval
            Q_upper = normalize(Jacobi(m, α+1, β))
            kern = inversechristoffelfunction(Q_upper)
            x1 = gausspoints(xi, Q_upper)
            w1 = 1 ./ (kern.(x1) .* (1 .- x1))
            wb = μ0 - sum(w1)
            x = [x1; 1]
            w = [w1; wb]
        else
            # the root xi is in a J-interval
            Q_lower = normalize(Jacobi(m, α, β+1))
            kern = inversechristoffelfunction(Q_lower)
            x1 = gausspoints(xi, Q_lower)
            w1 = 1 ./ (kern.(x1) .* (1 .+ x1))
            wa = μ0 - sum(w1)
            x = [-1; x1]
            w = [wa; w1]
        end
    end
    w, x
end
