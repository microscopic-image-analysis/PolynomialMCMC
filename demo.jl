"""
Demonstration of involutive MCMC on the polynomial regression task.
This file is meant to be run interactively (like a notebook).
"""

using CairoMakie  # For plotting
using Random

include("main.jl")

######################################################################
#          Some functions for data generation and plotting
######################################################################
"""
    gen_data_and_true_coefs(n, degree, xmin, xmax, σ)

Generate `n` data points sampled around a polynomial of fixed `degree`
with uniform distribution of x values between (`xmin`, `xmax`) and a
(gaussian) noise level of `σ`.
"""
function gen_data_and_true_coefs(n, degree, xmin, xmax, σ)
    x = rand(n) * (xmax - xmin) .- xmax
    X = design_matrix(x, degree)
    QR = qr(X)
    # isotropic gaussian coefficients in *orthogonal basis*
    Rβ = randn(degree + 1)
    # coefficients in standard basis
    β = inv(QR.R) * Rβ
    # same as y = X * β + σ * randn(n)
    y = QR.Q * Rβ + σ * randn(n)
    return x, y, β
end

"""
    plot_data_and_truth(x, y, β)

Generate a scatter plot of `x`-`y` and overlay polynomial with
coefficients `β` (in Monomial basis).
"""
function plot_data_and_truth(x, y, β)
    x_plot = range(extrema(x)...; length=100)
    X_plot = design_matrix(x_plot, length(β) - 1)
    y_plot = X_plot * β
    lines(x_plot, y_plot; label="Truth")
    scatter!(x, y; label="Data")
    current_figure()
end

"""
    plot_samples!(p, ..., βs, ...)

Add plots of polynomials (given as vector of polynomial coefficients `βs`)
on top of existing plot `p`.
"""
function plot_samples!(
    p,
    xmin,
    xmax,
    βs::AbstractVector{<:AbstractVector};
    color=:red,
    maxsamples=length(βs),
    label=nothing,
)
    x = range(xmin, xmax; length=100)
    y_mean = zeros(100)
    idxs = shuffle(eachindex(βs))
    meanlabel = isnothing(label) ? nothing : "mean $label"
    n_plotted = 0
    for i in idxs
        β = βs[i]
        X = design_matrix(x, length(β) - 1)
        y = X * β
        y_mean .+= y
        if n_plotted < maxsamples
            lines!(p[1, 1], x, y; color, alpha=0.2, label)
            label = nothing
            n_plotted += 1
        end
    end
    y_mean ./= length(βs)
    lines!(p[1, 1], x, y_mean; color, label=meanlabel)
    p
end

######################################################################

# Some constants

degree = 4  # True polynomial degree
n = 100  # Number of data points
σ = 0.1  # Stddev of noise
xmin, xmax = -5, 5  # Range of x-values

# Generate random polynomial of specified degree
# and data that is normally distributed around that polynomial
x, y, β = gen_data_and_true_coefs(n, degree, xmin, xmax, σ)

# Plot data
p = plot_data_and_truth(x, y, β)

######################################################################
#                  Assume polynomial degree is known
######################################################################

# Description of joint distribution:
joint = Joint(
    x,
    Poisson(123456),  # irrelevant for now, see next step
    1.0,  # prior variance for coefficients
    σ^2,  # true noise value
)
# Condition joint distribution on y and true polynomial degree
posterior = condition(joint, degree, y)

# Generate random samples from posterior 
Rβs = [rand(posterior) for _ in 1:1000]  # Random polynomial coefficients

# These coefficients are coefficients for an orthogonal polynomial basis,
# not the standard basis given by the Vandermonde-matrix.
# Transform back to standard basis: 
Q, R = qr(design_matrix(x, degree))
R_inv = inv(R)
βs = [R_inv * Rβ for Rβ in Rβs]

# Add samples to plot
plot_samples!(p, extrema(x)..., βs; color=:red, label="samples", maxsamples=20)

######################################################################
#                  "Wrong" polynomial degree
######################################################################
# Fresh plot
p = plot_data_and_truth(x, y, β)
assumed_degree = 3
posterior = condition(joint, assumed_degree, y)

# Generate random samples from posterior 
Rβs = [rand(posterior) for _ in 1:1000]  # Random polynomial coefficients

# These coefficients are coefficients for an orthogonal polynomial basis,
# not the standard basis given by the Vandermonde-matrix.
# Transform back to standard basis: 
Q, R = qr(design_matrix(x, assumed_degree))
R_inv = inv(R)
βs = [R_inv * Rβ for Rβ in Rβs]

# Add samples to plot
plot_samples!(p, extrema(x)..., βs; color=:red, label="samples", maxsamples=20)

######################################################################
#         Use involutive MCMC to sample the polynomial degree
######################################################################
# Fresh plot
p = plot_data_and_truth(x, y, β)
# Joint distribution, now with sensible prior for degree
joint = Joint(
    x,
    Poisson(4),  # Also try prior not centered around true value
    1.0,  # prior variance for coefficients
    σ^2,  # true noise value
)

# Instead of conditioning on an assumed degree, we use Involutive MC to sample
# over the polynomial degree
βs, Rβs, degrees, accepteds = sample(
    joint, y, SimpleAuxiliaryDistribution, push_pop_involution!, 1000
)
plot_samples!(p, extrema(x)..., βs; color=:red, label="samples", maxsamples=20)
hist(degrees; bins=((minimum(degrees) - 1):maximum(degrees)) .+ 0.5)

######################################################################
#                       Different initial states
######################################################################

# Fresh plot
p = plot_data_and_truth(x, y, β)
βs, Rβs, degrees, accepteds = sample(
    joint,
    y,
    SimpleAuxiliaryDistribution,
    push_pop_involution!,
    1000,
    (0, zeros(1)),  # start with zero-degree polynomial
)
plot_samples!(p, extrema(x)..., βs; color=:red, label="samples", maxsamples=20)
# Histogram of polynomial degree
hist(degrees; bins=((minimum(degrees) - 1):maximum(degrees)) .+ 0.5)

# Fresh plot
p = plot_data_and_truth(x, y, β)
βs, Rβs, degrees, accepteds = sample(
    joint,
    y,
    SimpleAuxiliaryDistribution,
    push_pop_involution!,
    1000,
    (10, zeros(11)),  # start with polynomial of degree 10
)
plot_samples!(p, extrema(x)..., βs; color=:red, label="samples", maxsamples=20)
# Histogram of polynomial degree
hist(degrees; bins=((minimum(degrees) - 1):maximum(degrees)) .+ 0.5)

######################################################################
#                          Some details
######################################################################

# Generate random polynomial coefficients (degree 5)
# Assume this is the "current state" when sampling
Rβ_old = randn(6)
# Create auxiliary distribution conditioned on current state:
aux_dist_old = SimpleAuxiliaryDistribution(joint, Rβ_old, y)
# Sample random auxiliary state
aux_old = rand(aux_dist_old)
# Apply involution
Rβ_new, aux_new, lad_jacobian = push_pop_involution!(copy(Rβ_old), aux_old)  # need to copy bc involution modifies vector in-place
# Apply involution again
Rβ_again, aux_again, _ = push_pop_involution!(copy(Rβ_new), aux_new)
# Back to old state:
Rβ_again == Rβ_old
isequal(aux_again, aux_old)  # `isequal(NaN, NaN) == true`