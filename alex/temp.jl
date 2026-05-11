# minimal working example to get a quadrature rule for a polynomial basis
using GeneralizedGauss

setprecision(BigFloat, 120; base=10)

p = 5
basis_funs = [x -> x^i for i in 0:2p+1]
basis_derivs = vcat(x -> zero(x), [x -> i*x^(i-1) for i in 1:2p+1])
a, b = BigFloat(0), BigFloat(1)
basis = quadbasis(basis_funs, basis_derivs, a, b)
w, x = compute_gauss_rule(basis, verbose=true, principal=:lower)

println("\n")
println("x: ", x)
println("w: ", w)