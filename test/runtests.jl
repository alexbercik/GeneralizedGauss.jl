using GeneralizedGauss
using BasisFunctions
using Test

include("test_helpers.jl")

include("fast_polynomial_quadratures.jl")
include("fast_solver_paths.jl")
include("fast_bigfloat_precision.jl")
include("fast_nonpolynomial_quadrature.jl")
include("ill_conditioned_basis.jl")
include("fast_basis_diagnostics.jl")
include("polynomials.jl")
include("orthogonalization.jl")
