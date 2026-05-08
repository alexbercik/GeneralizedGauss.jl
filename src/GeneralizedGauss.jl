module GeneralizedGauss

using BasisFunctions, LinearAlgebra, NLsolve, Random

import BasisFunctions: moment

export quadbasis,
    compute_moments,
    compute_gauss_rule,
    compute_gauss_rules,
    orthogonalize_basis,
    check_ECT_system,
    check_T_system,
    gauss_legendre

import Base:
    eltype,
    length,
    size

include("basis.jl")
include("orthogonalization.jl")
include("basis_checks.jl")
include("quadrule.jl")
include("representations.jl")
include("gengauss.jl")

end # module
