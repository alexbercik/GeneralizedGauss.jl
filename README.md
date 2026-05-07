# GeneralizedGauss.jl

A package for the computation of generalized Gaussian quadrature rules. For a description of the algorithm, see the paper [On the computation of Gaussian quadrature rules for Chebyshev sets of linearly independent functions](https://arxiv.org/abs/1710.11244).
This repository is forked from Daan Huybrechs (https://github.com/daanhb/GeneralizedGauss.jl)

This package computes generalized Gaussian quadrature rules from:
- a basis `dict`,
- and the moments of that basis with respect to some measure `dμ`.

The main entry points are:

```julia
compute_gauss_rule(dict::Dictionary, moments = nothing; measure = nothing, kwargs...)
compute_gauss_rules(dict::Dictionary, moments = nothing; measure = nothing, kwargs...)
compute_moments(dict::Dictionary; measure = nothing)
```

Short version:
- `dict` defines the function space `{ϕ_i(x)}`.
- `moments[i]` must equal `∫ ϕ_i(x) dμ(x)` for the `i`-th basis function.
- If you do not pass `moments`, they are computed automatically.
- If you pass `measure=μ`, the automatic path uses `moment(dict, i; measure=μ)`.
- If you pass both `moments` and `measure`, the explicit `moments` win.
- If you do not pass `add_endpoint`, it defaults to `:left` for
  `principal=:lower` and `:right` for `principal=:upper`.

## 1) What you pass in

### Basis

`dict` is a `BasisFunctions.Dictionary`. Its support defines the integration
domain, and its elements define the exactness conditions for the quadrature rule.

You can use:
- built-in `BasisFunctions` dictionaries such as `ChebyshevT`, `Legendre`,
  `Jacobi`, `Laguerre`, `Hermite`, and combinations of them,
- or a custom basis via `quadbasis(funs, fun_derivs, a, b)`.

### Moments

If you pass `moments`, they must satisfy

```julia
moments[i] = ∫ ϕ_i(x) dμ(x)
```

in the same order as the dictionary elements.

This is the safest option when:
- you already know the moments,
- you want full control over the measure,
- you want to minimize numerical error (exactness of moments),
- or your basis is custom.

### Measure

If `moments === nothing`, the package computes them automatically.

- `compute_gauss_rule(dict; measure=μ)` uses `moment(dict, i; measure=μ)`.
- `compute_gauss_rule(dict)` uses the default `moment(dict, i)`, i.e. Lebesgue measure

Note that for the manual `quadbasis(...)` type in this repository, `measure(dict)` is not defined.

## 2) Which quadrature rule do you get?

The final rule depends on the basis length and the keyword arguments 
`principal` and `add_endpoint`.

If `add_endpoint` is omitted, the natural default is:
- `principal=:lower` -> `add_endpoint=:left`
- `principal=:upper` -> `add_endpoint=:right`

### Even number of basis functions

If `length(dict) = 2l`:

- `principal = :lower` gives the `l`-point LG-like rule
  (Gauss-Legendre-like, no endpoints).
- `principal = :upper` gives the `(l+1)`-point LGL-like rule
  (Gauss-Lobatto-like, both endpoints included).

For even basis length, `add_endpoint=:left` and `:right` follow different
continuation paths but reach the same final rule for a given `principal`.

### Odd number of basis functions

If `length(dict) = 2l+1`:

- `principal = :lower, add_endpoint = :left` gives a left-Radau rule
  containing the left endpoint.
- `principal = :upper, add_endpoint = :right` gives a right-Radau rule
  containing the right endpoint.

These are the natural pairings implemented by the algorithm. The crossed
pairings are rejected.

### Endpoint singularities

This matters for bases or integrands with endpoint singularities:

- LG-like rules (`principal=:lower`, even basis) avoid both endpoints.
- LGL-like rules (`principal=:upper`, even basis) include both endpoints, so the
  basis must be well-defined there.
- Radau rules include one endpoint, so choose the side that avoids forcing the
  singular endpoint into the rule.
- For LG-like and Radau rules, choose `add_endpoint` appropriately so the 
  continuation algorithm is valid, avoiding the endpoint singularity.

## 3) Automatic vs explicit moments

The following two approaches are equivalent, as they compute the moments
numerically (possibly with smart evaluation, depending on the basis):
(Internally, this calls the same function `compute_moments`, or `moment`.)

```julia
moments = [moment(dict, i; measure=μ) for i in eachindex(dict)]
w, x = compute_gauss_rule(dict, moments)
```

or

```julia
w, x = compute_gauss_rule(dict; measure=μ)
```

`BasisFunctions` and `DomainIntegrals` handle the actual moment evaluation.
For known basis/measure combinations this may use specialized formulas; otherwise
it falls back to numerical integration. Alternatively, you can create your own
vector of exact moments corresponding to each basis function.

## 4) Defining your own basis

Use:

```julia
basis = quadbasis(funs, fun_derivs, a, b)
```

where:
- `funs[i](x)` evaluates the `i`-th basis function,
- `fun_derivs[i](x)` evaluates its first derivative,
- `[a,b]` is the support interval.

For custom bases:
- moments can still be calculated through the generic `BasisFunctions` 
  integration fallback,
- but for weighted problems it is necessary to pass either `measure=μ` 
- it may be helpful to explicitly pass `moments` to ensure accuracy.

## 5) Basis orthogonalization

For ill-conditioned bases (e.g. monomials past degree ~7), `orthogonalize_basis`
orthonormalizes the basis before computing the quadrature rule. It uses unpivoted
Householder QR factorization of the weighted evaluation matrix
A_{qi} = √w_q · φ_i(x_q), giving T = R⁻ᵀ (lower-triangular).

Two design choices are important here:

- **Triangular (not symmetric/Löwdin):** The continuation algorithm slices the
  basis as `dict[1:k]` and requires every prefix {ψ_1, ..., ψ_k} to be a
  Chebyshev system. A lower-triangular T ensures ψ_i depends only on
  φ_1, ..., φ_i, preserving the graded structure. A symmetric transform (Löwdin)
  would mix all basis functions into every ψ_i, breaking this property.

- **QR of A (not Cholesky of G = AᵀA):** Forming G squares the condition number
  (κ(G) = κ(A)²), so Cholesky loses twice as many digits as QR. Since Householder
  QR works directly on A, it avoids this squaring while producing the same
  triangular factor (Rᵀ = L, the Cholesky factor of G).

```julia
orth_basis, T_mat = orthogonalize_basis(basis)
orth_moments = T_mat * moments
```

A custom weight for the inner product can be passed via the `measure` keyword:

```julia
orth_basis, T_mat = orthogonalize_basis(basis; measure=x -> exp(x))
```

The quadrature order for the evaluation matrix defaults to `max(2N, 8)`, which
is sufficient for polynomial and smooth bases. Override with `quad_order` for
oscillatory or singular weight functions that need more points.

## 6) Arbitrary precision with `BigFloat`

The continuation algorithm also works in arbitrary precision. The important
point is that the *basis data itself* should be built in `BigFloat`, not
just the final call to `compute_gauss_rule`. See the following example:

```julia
using BasisFunctions, DomainSets, GeneralizedGauss
import GeneralizedGauss: solver_tolerance

newton_tol_digits = 30 # digits of precision for Newton solves 
extra_digits = 10 # additional digits of precision to work in

total_digits = newton_tol_digits + extra_digits # total precision 
bigfloat_precision_bits = ceil(Int, total_digits * log2(big(10)))
setprecision(BigFloat, bigfloat_precision_bits)

# Optional: tighten or loosen the nonlinear solver tolerance explicitly.
solver_tolerance(::Type{BigFloat}) = BigFloat(10)^(-newton_tol_digits)

a = BigFloat(0)
b = BigFloat(1)

n = 10
funs = [x -> x^i for i in 0:n-1]
fun_derivs = vcat(x -> zero(x), [x -> i*x^(i-1) for i in 1:n-1])

basis = quadbasis(funs, fun_derivs, a, b)
w, x = compute_gauss_rule(basis)
```

Practical notes:
- Use `BigFloat` endpoints when mapping a basis to an interval. For example,
  `ChebyshevT(10) → (0.0..1.0)` keeps the support in `Float64`, which can cap
  basis evaluation accuracy near `Float64` precision. The correct approach is
  `ChebyshevT(10) → (a..b)`.
- HOWEVER, the default bases `ChebyshevT`, `Lagrange`, etc. often do not 
  natively support `BigFloat`. Therefore, I suggest always defining a custom
  basis when working with a desired precision.
- If you pass explicit `moments`, those moments should also be computed/stored
  in `BigFloat`. Otherwise, the automatically computed moments should preserve
  `BigFloat` precision (as long as you use a custom basis).
- If a computation becomes fragile at higher degree, increase
  `setprecision(BigFloat, ...)` and consider a better-conditioned basis.
  You can orthogonalize your basis with `orthogonalize_basis` (see above),
  which preserves the `BigFloat` precision.

By default, `GeneralizedGauss` uses `sqrt(eps(BigFloat))` as the Newton solver
tolerance, so overriding `solver_tolerance(::Type{BigFloat})` is optional.

## 7) Examples

### A standard LG-like rule

```julia
using BasisFunctions, GeneralizedGauss

basis = ChebyshevT(10)
w, x = compute_gauss_rule(basis)   # 5-point LG rule
```

### The LGL-like rule on the same space

```julia
using BasisFunctions, GeneralizedGauss

basis = ChebyshevT(10)
w, x = compute_gauss_rule(basis; principal=:upper)   # 6-point LGL rule
```

### A right-Radau rule

```julia
using BasisFunctions, GeneralizedGauss

basis = ChebyshevT(11)
w, x = compute_gauss_rule(basis; principal=:upper) # 6-point right-Radau rule
```

### A logarithmic endpoint-singular space on `[0,1]`

```julia
using BasisFunctions, DomainSets, GeneralizedGauss

N = 6
cheb = ChebyshevT(N) → 0..1
basis = cheb ⊕ log*cheb

# ensure we add the endpoints on the right to avoid the left singularity
w, x = compute_gauss_rule(basis; principal=:lower, add_endpoint=:right)
```


### Automatic weighted moments using the basis' natural measure

```julia
using BasisFunctions, GeneralizedGauss

basis = ChebyshevT(6)
mu = measure(basis)

w, x = compute_gauss_rule(basis; measure=mu)
```

### Explicit weighted moments using the same measure

```julia
using BasisFunctions, GeneralizedGauss

basis = ChebyshevT(6)
mu = measure(basis)
moments = [moment(basis, i; measure=mu) for i in eachindex(basis)]

w, x = compute_gauss_rule(basis, moments)
```

### Directly specifying a classical measure

```julia
using BasisFunctions, GeneralizedGauss

basis = ChebyshevT(6)
mu = ChebyshevTWeight()

w, x = compute_gauss_rule(basis; measure=mu)
```

Other useful measures include:
- `LegendreWeight()`
- `JacobiWeight(α, β)`
- `LaguerreWeight(α)`
- `HermiteWeight()`

These come from `DomainIntegrals.jl` and are re-exported by `BasisFunctions.jl`.

### A custom continuous weight

```julia
using BasisFunctions, DomainSets, GeneralizedGauss

basis = Legendre(6)
mu = BasisFunctions.GenericWeight(support(basis), x -> exp(x))

w, x = compute_gauss_rule(basis; measure=mu)
```

You can also inspect individual moments and inner products:

```julia
moment(basis, 3; measure=mu)
innerproduct(basis[2], basis[4], mu)
```

### A manual polynomial basis (orthogonalized)

```julia
using GeneralizedGauss

n = 10
funs = [x -> x^i for i in 0:n-1]
fun_derivs = vcat(x -> zero(x), [x -> i*x^(i-1) for i in 1:n-1])

basis = quadbasis(funs, fun_derivs, -1.0, 1.0)

orth_basis, T_mat = orthogonalize_basis(basis)
orth_moments = T_mat * moments
# Note: explicitly computing the orth_moments is unecessary
# if one instead wishes to rely upon the automatically computed moments

w, x = compute_gauss_rule(orth_basis, orth_moments)
```

### A manual basis with a custom weighted measure

```julia
using GeneralizedGauss, BasisFunctions, DomainSets

n = 6
funs = [x -> x^i for i in 0:n-1]
fun_derivs = vcat(x -> zero(x), [x -> i*x^(i-1) for i in 1:n-1])

basis = quadbasis(funs, fun_derivs, -1.0, 1.0)
mu = BasisFunctions.GenericWeight(support(basis), x -> exp(x))
# Note: if we want to explicitly compute the moments, we can use:
# moments = [moment(basis, i; measure=mu) for i in eachindex(basis)]

w, x = compute_gauss_rule(basis; measure=mu)
```

### Getting all intermediate rules

```julia
using BasisFunctions, GeneralizedGauss

basis = ChebyshevT(10)
w, x, xi_checkpoints, w_checkpoints, x_checkpoints =
    compute_gauss_rules(basis; verbose=true)
```

This returns the final rule together with the sequence of intermediate principal
representations encountered during continuation.
