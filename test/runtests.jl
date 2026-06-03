using GeneralizedGauss
using BasisFunctions
using Test

const RUN_SLOW_TESTS = lowercase(get(ENV, "GENGAUSS_RUN_SLOW_TESTS", "")) in
                       ("1", "true", "yes", "y")

include("test_helpers.jl")

include("fast_polynomial_quadratures.jl")
include("fast_solver_paths.jl")
include("fast_bigfloat_precision.jl")
include("fast_nonpolynomial_quadrature.jl")
include("ill_conditioned_basis.jl")
include("fast_basis_diagnostics.jl")

if RUN_SLOW_TESTS
    @testset "Slow diagnostic suites" begin
        include("polynomials.jl")
        include("orthogonalization.jl")
        include("basis_checks.jl")
        include("bigfloat_precision.jl")
    end
end
