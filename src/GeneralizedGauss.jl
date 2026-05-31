module GeneralizedGauss

using BasisFunctions, LinearAlgebra, NLsolve, Random
import NOMAD

import BasisFunctions: dict_moment

export quadbasis,
    compute_moments,
    compute_gauss_rule,
    compute_gauss_rules,
    orthogonalize_basis,
    check_ECT_system,
    check_T_system,
    gauss_legendre,
    legendre_functions,
    legendre_basis

import Base:
    eltype,
    length,
    size

include("basis.jl")
include("polynomials.jl")
include("orthogonalization.jl")
include("basis_checks.jl")
include("root_solvers.jl")
include("quadrule.jl")
include("representations.jl")
include("gengauss.jl")
include("mads.jl")

end # module
