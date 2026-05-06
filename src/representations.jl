
"Generate the non-linear system for a lower principal representation for odd n."
function LowerPrincipalOdd(s::Dictionary, moments = compute_moments(s))
    n = length(s) - 1
    @assert isodd(n)
    l = (n+1) >> 1
    QuadRuleFreePoints(QuadRuleData(s, l, moments))
end

"Generate the non-linear system for an upper principal representation for odd n."
function UpperPrincipalOdd(s::Dictionary, moments = compute_moments(s))
    n = length(s) - 1
    @assert isodd(n)
    l = ((n+1) >> 1) + 1
    QuadRuleFixedPoints(QuadRuleData(s, l, moments), [1, l], [supportleft(s), supportright(s)])
end

"Generate the non-linear system for a lower principal representation for even n."
function LowerPrincipalEven(s::Dictionary, moments = compute_moments(s))
    n = length(s) - 1
    @assert iseven(n)
    l = (n >> 1) + 1
    QuadRuleFixedPoints(QuadRuleData(s, l, moments), [1], [supportleft(s)])
end

"Generate the non-linear system for an upper principal representation for even n."
function UpperPrincipalEven(s::Dictionary, moments = compute_moments(s))
    n = length(s) - 1
    @assert iseven(n)
    l = (n >> 1) + 1
    QuadRuleFixedPoints(QuadRuleData(s, l, moments), [l], [supportright(s)])
end

# The fixed root is in K1 = [a,t1].
function CanonicalRepresentationOdd_K1(s::Dictionary, xstar, moments = compute_moments(s), fixed_idx = nothing)
    n = length(s) - 1
    @assert isodd(n)
    l = ((n+1) >> 1) + 1
    if fixed_idx === nothing
        fixed_idx = [1, l]
    end
    @assert all(idx -> idx in 1:l, fixed_idx) "fixed_idx=$fixed_idx must all be in 1:$l"
    QuadRuleFixedPoints(QuadRuleData(s, l, moments), fixed_idx, [xstar, supportright(s)])
end

# The fixed root is in J1 = [t1,s2].
function CanonicalRepresentationOdd_J1(s::Dictionary, xstar, moments = compute_moments(s), fixed_idx = nothing)
    n = length(s) - 1
    @assert isodd(n)
    l = ((n+1) >> 1) + 1
    if fixed_idx === nothing
        fixed_idx = [1, 2]
    end
    @assert all(idx -> idx in 1:l, fixed_idx) "fixed_idx=$fixed_idx must all be in 1:$l"
    QuadRuleFixedPoints(QuadRuleData(s, l, moments), fixed_idx, [supportleft(s), xstar])
end

# The fixed root is in J1 = [a,s1].
function CanonicalRepresentationEven_J1(s::Dictionary, xstar, moments = compute_moments(s), fixed_idx = nothing)
    n = length(s) - 1
    @assert iseven(n)
    l = (n >> 1) + 1
    if fixed_idx === nothing
        fixed_idx = [1]
    end
    @assert all(idx -> idx in 1:l, fixed_idx) "fixed_idx=$fixed_idx must all be in 1:$l"
    QuadRuleFixedPoints(QuadRuleData(s, l, moments), fixed_idx, [xstar])
end

# The fixed root is in K1 = [s1,t2].
#
# This canonical has BOTH endpoints fixed at a and b, plus a third fixed
# position at xstar. The default `fixed_idx = [1, 2, l]` puts xstar at the
# 2nd-leftmost position (the K_1 form, eq. (2.16) of the paper).
#
# Passing `fixed_idx = [1, l-1, l]` instead puts xstar at the 2nd-rightmost
# position — the symmetric "K_{l-1}" form used by Gauss-Lobatto continuation
# from a left-anchored sweep. fixed_pts is always [a, xstar, b], so caller
# must pick fixed_idx so that the middle entry (= xstar) is correctly placed.
function CanonicalRepresentationEven_K1(s::Dictionary, xstar, moments = compute_moments(s), fixed_idx = nothing)
    n = length(s) - 1
    @assert iseven(n)
    l = (n >> 1) + 2
    if fixed_idx === nothing
        fixed_idx = [1, 2, l]
    end
    @assert all(idx -> idx in 1:l, fixed_idx) "fixed_idx=$fixed_idx must all be in 1:$l"
    QuadRuleFixedPoints(QuadRuleData(s, l, moments), fixed_idx, [supportleft(s), xstar, supportright(s)])
end
