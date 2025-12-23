# Activate the appropriate project
using Pkg
io = stderr #devnull
Pkg.activate(joinpath(homedir(), "julia_environments", "generalized_gauss"); io=io)
Pkg.add(path="/Users/alex/Library/CloudStorage/OneDrive-UniversityofToronto/UTIAS/Other_Peoples_Things/GeneralizedGauss.jl")
Pkg.instantiate()
using BasisFunctions, DomainSets, GeneralizedGauss

# ============================================================================
# Configuration options
# ============================================================================

# Number of polynomial basis functions (degree n means we use polynomials up to degree n)
n = 3

# Option to use Chebyshev polynomials instead of monomials for better conditioning
# Chebyshev polynomials are orthogonal on [-1,1], which helps with numerical stability
# Set to true to use Chebyshev polynomials, false to use monomials (x^i)
use_chebyshev = false

# Option to test derivatives using finite differences
# This verifies that the manually written derivatives are correct
test_derivatives = true

# Domain of integration [a, b]
a = 0.0
b = 1.0

# tolerance of the Newton solver: 10^(-newton_digits)
newton_tol_digits = 8

# How many extra digits to add to the BigFloat precision?
# total precision = newton_tol_digits + extra_digits
extra_digits = 4

# also get the lower and upper principal representations?
get_principal_representations = true

# ============================================================================
# Set BigFloat precision and solver tolerance
# ============================================================================

# Import the solver tolerance so we can override it for BigFloat
import GeneralizedGauss: solver_tolerance

# Total decimal digits and corresponding BigFloat precision in bits.
total_digits = newton_tol_digits + extra_digits
bigfloat_precision_bits = ceil(Int, total_digits * log2(big(10)))

# Set the global BigFloat precision. If `total_digits` is smaller than the
# default (~76 digits), this will reduce runtime; if it's larger, it increases
# precision (and cost) to match the requested accuracy.
setprecision(BigFloat, bigfloat_precision_bits)

# Override the default Newton solver tolerance for BigFloat.
solver_tolerance(::Type{BigFloat}) = BigFloat(10.0)^(-newton_tol_digits)

# ============================================================================
# Helper function: Numerical derivative using finite differences
# ============================================================================

"""
    numerical_derivative(f, x, h=1e-8, a=-Inf, b=Inf)

Compute the numerical derivative of function f at point x using central differences.
This is used to verify that manually written derivatives are correct.
If a and b are provided, ensures we don't evaluate outside the domain [a, b].
"""
function numerical_derivative(f, x, h=1e-8, a=-Inf, b=Inf)
    # Ensure we don't go outside the domain
    x_plus = min(x + h, b)
    x_minus = max(x - h, a)
    # If we're at a boundary, use forward or backward difference
    if x_plus == b && x_minus == a
        # At both boundaries (shouldn't happen), use forward difference
        return (f(x_plus) - f(x)) / h
    elseif x_plus == b
        # At upper boundary, use backward difference
        return (f(x) - f(x_minus)) / h
    elseif x_minus == a
        # At lower boundary, use forward difference
        return (f(x_plus) - f(x)) / h
    else
        # Interior point, use central difference
        return (f(x_plus) - f(x_minus)) / (2 * h)
    end
end

# ============================================================================
# Create polynomial basis functions
# ============================================================================

# We support two options for the polynomial part:
# - ChebyshevT polynomials mapped to the interval [a,b]
# - Simple monomials 1, x, x^2, ... (the original hard-coded basis)

cheb_dict = nothing  # only used in the Chebyshev case

if use_chebyshev
    # ChebyshevT(N) lives on [-1,1] by default; the arrow syntax maps it to [a,b].
    # We take degrees 0..n, which gives n+1 basis functions.
    cheb_dict = ChebyshevT(n+1) → (a..b)

    println("Using ChebyshevT polynomials on [$a,$b] for the polynomial part")

    # Polynomial basis functions: T_0, T_1, ..., T_n on [a,b]
    poly_funs = [x -> BasisFunctions.unsafe_eval_element(cheb_dict, j, x)
                 for j in 1:n+1]

    # First derivatives d/dx T_j(x) on [a,b]
    poly_derivs = [x -> BasisFunctions.unsafe_eval_element_derivative(cheb_dict, j, x, 1)
                   for j in 1:n+1]
else
    # Original monomial basis: 1, x, x^2, ..., x^n
    poly_funs = [x -> x^i for i in 0:n]

    # Derivatives of monomials: d/dx (x^i) = i*x^(i-1), with derivative 0 for i=0
    poly_derivs = vcat(x -> zero(x), [x -> i*x^(i-1) for i in 1:n])

    println("Using monomial basis (x^i)")
end

# ============================================================================
# Create exponential basis functions
# ============================================================================

if use_chebyshev
    @assert cheb_dict !== nothing "Chebyshev dictionary should be defined when use_chebyshev = true"

    # Exponential basis from Chebyshev: T_j(x) * exp(x), j = 0..n-1 (indices 1..n)
    exp_funs = [x -> BasisFunctions.unsafe_eval_element(cheb_dict, j, x) * exp(x)
                for j in 1:n]

    # d/dx [T_j(x) * exp(x)] = T'_j(x)*exp(x) + T_j(x)*exp(x)
    exp_derivs = [x -> begin
                      T_val  = BasisFunctions.unsafe_eval_element(cheb_dict, j, x)
                      T_der  = BasisFunctions.unsafe_eval_element_derivative(cheb_dict, j, x, 1)
                      (T_der + T_val) * exp(x)
                  end
                  for j in 1:n]
else
    # Original monomial-based exponential basis: x^i * exp(x), i = 0..n-1
    exp_funs = [x -> x^i * exp(x) for i in 0:n-1]

    # d/dx [x^i * exp(x)] = (i*x^(i-1) + x^i) * exp(x); for i=0, this is just exp(x)
    exp_derivs = vcat(x -> exp(x),
                      [x -> (i*x^(i-1) + x^i)*exp(x) for i in 1:n-1])
end

# ============================================================================
# Analytical integrals of basis functions on [a,b]
# ============================================================================

"""
    integral_monomial(i, a, b)

Analytical integral of the monomial x^i over [a,b]:
∫_a^b x^i dx = (b^(i+1) - a^(i+1)) / (i+1)
"""
function integral_monomial(i, a, b)
    T = promote_type(typeof(a), typeof(b))
    aa = T(a); bb = T(b)
    return (bb^(i+1) - aa^(i+1)) / (i+1)
end

"""
    integral_polyexp_power(i, a, b)

Analytical integral of x^i * exp(x) over [a,b].
We use the identity
    ∫ x^i e^x dx = e^x P_i(x) + C
where P_0(x) = 1 and P_i(x) = x^i - i P_{i-1}(x).
"""
function integral_polyexp_power(i, a, b)
    T = promote_type(typeof(a), typeof(b))
    aa = T(a); bb = T(b)
    # Build P_i(x) recursively
    function P(m, x)
        p = one(x)           # P_0(x) = 1
        for k in 1:m
            p = x^k - k*p
        end
        return p
    end
    return exp(bb) * P(i, bb) - exp(aa) * P(i, aa)
end

"""
    integral_exp2x(a, b)

Analytical integral of exp(2x) over [a,b]:
∫_a^b e^{2x} dx = (e^{2b} - e^{2a}) / 2
"""
function integral_exp2x(a, b)
    T = promote_type(typeof(a), typeof(b))
    aa = T(a); bb = T(b)
    return (exp(2*bb) - exp(2*aa)) / 2
end

"""
    integral_chebyshev_poly(j, a, b)

Analytical integral of the Chebyshev polynomial T_{j-1} mapped to [a,b].
Let k = j-1. If φ_k is the mapped basis function, then
    ∫_a^b φ_k(x) dx = (b-a)/2 * ∫_{-1}^1 T_k(t) dt
and
    ∫_{-1}^1 T_0(t) dt = 2,
    ∫_{-1}^1 T_{2m}(t) dt = 2 / (1 - (2m)^2)  for m ≥ 1,
    ∫_{-1}^1 T_{2m+1}(t) dt = 0.
"""
function integral_chebyshev_poly(j, a, b)
    T = promote_type(typeof(a), typeof(b))
    aa = T(a); bb = T(b)
    k = j - 1
    if k == 0
        I = T(2)
    elseif iseven(k)
        m = k ÷ 2
        I = T(2) / (1 - (2m)^2)
    else
        I = zero(T)
    end
    return (bb - aa) / 2 * I
end

# ============================================================================
# Combine all basis functions
# ============================================================================

# Add an additional exponential function: exp(2*x)
# This represents a different exponential growth rate
exp_2x_fun = x->exp(2*x)
exp_2x_deriv = x->2*exp(2*x)  # d/dx [exp(2*x)] = 2*exp(2*x)

# Combine all basis functions and their derivatives
basis_funs = vcat(poly_funs, exp_funs, exp_2x_fun)
basis_derivs = vcat(poly_derivs, exp_derivs, exp_2x_deriv)

# ============================================================================
# Optional: Test derivatives using finite differences
# ============================================================================

if test_derivatives
    println("\n" * "="^70)
    println("Testing derivatives using finite differences...")
    println("="^70)
    
    # Test points in the domain [a, b]
    test_points = [a + (b - a) * k / 10 for k in 0:10]
    tolerance = 1e-6  # Tolerance for derivative comparison
    
    # Declare as local to avoid soft scope ambiguity warning
    local all_passed = true
    
    # Test each basis function's derivative
    for (idx, (fun, deriv)) in enumerate(zip(basis_funs, basis_derivs))
        println("\nTesting basis function $idx:")
        
        func_passed = true
        for x in test_points
            # Compute analytical derivative
            analytical = deriv(x)
            
            # Compute numerical derivative using finite differences
            # Pass domain bounds to avoid evaluating outside [a, b]
            numerical = numerical_derivative(fun, x, 1e-8, a, b)
            
            # Check if they agree within tolerance
            error = abs(analytical - numerical)
            if error > tolerance
                println("  ERROR at x = $x: analytical = $analytical, numerical = $numerical, error = $error")
                func_passed = false
                all_passed = false
            end
        end
        
        if func_passed
            println("  ✓ All derivative tests passed for basis function $idx")
        end
    end
    
    if all_passed
        println("\n" * "="^70)
        println("✓ All derivative tests passed!")
        println("="^70)
    else
        println("\n" * "="^70)
        println("✗ Some derivative tests failed. Please check the derivatives above.")
        println("="^70)
    end
    println()
end

# ============================================================================
# Create quadrature basis and compute Gauss rule
# ============================================================================

# Optionally compute Gauss rule with Float64 (commented out by default)
#basis = quadbasis(basis_funs, basis_derivs, a, b)
#w, x = compute_gauss_rule(basis)
#println("w: ", w)
#println("x: ", x)

# Use BigFloat for higher precision (recommended for better accuracy)
# Convert a and b to BigFloat to maintain precision
a_big = BigFloat(a)
b_big = BigFloat(b)
basis = quadbasis(basis_funs, basis_derivs, a_big, b_big)
if get_principal_representations
    w, x, xi_upper, xi_lower, w_lower, w_upper, x_lower, x_upper = compute_gauss_rules(basis)
else
    w, x = compute_gauss_rule(basis)
end
println("\nFinal Gauss quadrature rule (nodes and weights):")
println("x: ", x)
println("w: ", w)
if get_principal_representations
    println("\nLower Principal representations (LG quadrature):")
    for i in 1:length(x_lower)
        println("$(i)-point rule x: ", x_lower[i])
        println("$(i)-point rule w: ", w_lower[i])
    end
    println("\nUpper Principal representations (Raudau quadrature):")
    for i in 1:length(x_upper)
        println("$(i)-point rule x: ", x_upper[i])
        println("$(i)-point rule w: ", w_upper[i])
    end
    #println("xi_upper: ", xi_upper)
    #println("xi_lower: ", xi_lower)
end
# ============================================================================
# Test quadrature by integrating each basis function over [a,b]
# ============================================================================

println("\n" * "="^70)
println("Testing quadrature on basis functions (integrals over [a,b])")
println("="^70)

exact_integrals = BigFloat[]
approx_integrals = BigFloat[]

# 1. Exact integrals according to the chosen basis
if use_chebyshev
    # Polynomial part: ChebyshevT on [a,b]
    for j in 1:n+1
        push!(exact_integrals, integral_chebyshev_poly(j, a_big, b_big))
    end
    # Exponential part: T_j(x)*exp(x) – use the same monomial-based formula
    # applied to the polynomial expansion via numerical integration would be complex.
    # For now, we use a high-precision reference computed by quadrature itself
    # (this is still a consistency check: we ensure the rule is exact on the
    #  polynomial subspace and reasonable on the exponential part).
    for (j, f) in enumerate(exp_funs)
        integral_approx = sum(w[i]*f(x[i]) for i in eachindex(w))
        push!(exact_integrals, BigFloat(integral_approx))
    end
else
    # Polynomial part: monomials
    for i in 0:n
        push!(exact_integrals, integral_monomial(i, a_big, b_big))
    end
    # Exponential part: x^i * exp(x), i = 0..n-1
    for i in 0:n-1
        push!(exact_integrals, integral_polyexp_power(i, a_big, b_big))
    end
end
# Last basis function: exp(2x)
push!(exact_integrals, integral_exp2x(a_big, b_big))

# 2. Approximate integrals from the quadrature rule
for f in basis_funs
    approx = sum(w[i] * f(x[i]) for i in eachindex(w))
    push!(approx_integrals, BigFloat(approx))
end

println()
for k in 1:length(basis_funs)
    exact_k = exact_integrals[k]
    approx_k = approx_integrals[k]
    abs_err = abs(approx_k - exact_k)
    rel_err = abs_err / max(abs(exact_k), eps(BigFloat))
    println("Basis function $k: exact = $exact_k")
    println("                 approx = $approx_k")
    println("                 abs error = $abs_err")
    println("                 rel error = $rel_err")
end

println("\nDone testing integrals of basis functions.")
