# Minimal working example: compute a generalized Gaussian rule for monomials on
# [0, 1] using BigFloat arithmetic.
using GeneralizedGauss

setprecision(BigFloat, 120; base=10)

# Use moments for monomials through degree 2p + 1, which produces a p + 1 point
# Gaussian rule in the default lower-principal path.
p = 5
basis_funs = [x -> x^i for i in 0:2p+1]
basis_derivs = vcat(x -> zero(x), [x -> i*x^(i-1) for i in 1:2p+1])
a, b = BigFloat(0), BigFloat(1)
basis = quadbasis(basis_funs, basis_derivs, a, b)
w, x = compute_gauss_rule(basis, verbose=true, principal=:lower)

println("\n")
println("x: ", x)
println("w: ", w)
