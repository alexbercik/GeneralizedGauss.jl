# Minimal mixed-basis example: polynomials, x^i exp(x), and exp(2x) on [-1, 1].
using GeneralizedGauss

setprecision(BigFloat, 20; base=10) do
    p = 2
    qfuncs_poly  = [let k=k; x -> x^k end for k in 0:(2p - 1)]
    qderivs_poly = [let k=k; k == 0 ? (x -> zero(x)) : (x -> k * x^(k-1)) end for k in 0:(2p - 1)]
    qfuncs_exp = [x -> x^i * exp(x) for i in 0:p]
    qderivs_exp = vcat(exp,
                    [x -> (i * x^(i - 1) + x^i) * exp(x) for i in 1:p])
    qfuncs_exp2 = x -> exp(2 * x)
    qderivs_exp2 = x -> 2 * exp(2 * x)
    qfuncs = vcat(qfuncs_poly, qfuncs_exp, qfuncs_exp2)
    qderivs = vcat(qderivs_poly, qderivs_exp, qderivs_exp2)
    quad_basis = quadbasis(qfuncs, qderivs, BigFloat(-1), BigFloat(1))
    quad_basis, _ = orthogonalize_basis(quad_basis)
    # The late principal and final Lobatto solves are ill-conditioned, so this
    # driver allows a small per-call lost-digits acceptance window.
    w, x = compute_gauss_rule(quad_basis;
        principal=:upper, verbose=true, lost_digits=5)
    println("\nx: ", x)
    println("w: ", w)
end
