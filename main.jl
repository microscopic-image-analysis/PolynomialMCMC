using Distributions
using GLMakie
using LinearAlgebra
using Random
using StaticArrays
using Statistics

struct AuxiliaryDistribution{D,C}
    increase_dist::D
    coef_dist::C
end

function Base.rand(dist::AuxiliaryDistribution)
    increase = rand(dist.increase_dist)
    coef = if increase
        rand(dist.coef_dist)
    else
        NaN
    end
    (; increase, coef)
end

function Distributions.logpdf(
    dist::AuxiliaryDistribution, val::NamedTuple{(:increase, :coef)}
)
    increase = val.increase
    score = logpdf(dist.increase_dist, increase)
    if increase
        score += logpdf(dist.coef_dist, val.coef)
    end
    score
end

function involution!(coefs, aux::NamedTuple{(:increase, :coef)})
    logabsdet = 0.0
    if aux.increase
        coefs = push!(coefs, aux.coef)
        return coefs, (; increase=false, coef=NaN), logabsdet
    else
        coef = pop!(coefs)
        return coefs, (; increase=true, coef), logabsdet
    end
end

"""
    gen_data_and_true_coefs(n, degree, xmin, xmax, σ)

Generate data 
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

function plot_data_and_truth(x, y, β)
    x_plot = range(extrema(x)...; length=100)
    X_plot = design_matrix(x_plot, length(β) - 1)
    y_plot = X_plot * β
    lines(x_plot, y_plot; label="Truth")
    scatter!(x, y; label="Data")
    current_figure()
end

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
            lines!(p, x, y; color, alpha=0.2, label)
            label = nothing
            n_plotted += 1
        end
    end
    y_mean ./= length(βs)
    lines!(p, x, y_mean; color, label=meanlabel)
end

function posterior_Rβ(
    x, y, prior_precision, likelihood_precision, degree=size(prior_precision, 1)
)
    X = design_matrix(x, degree)
    QR = qr(X)
    posterior_Rβ(QR, y, prior_precision, likelihood_precision)
end

function posterior_Rβ(
    QR::GLMakie.LinearAlgebra.QRCompactWY, y, prior_precision, likelihood_precision
)
    Q = QR.Q
    # Q * Rβ = y + e,  e ~ Normal(0, σ)
    posterior_cov = posterior_covariance(prior_precision * I, likelihood_precision, Q)
    posterior_mean = posterior_cov * Matrix(Q)' * (likelihood_precision * I) * y
    posterior = MvNormal(posterior_mean, posterior_cov)
    posterior, QR.R
end

function sample_posteriorβ(x, y, degree, prior_precision, likelihood_precision, n)
    posterior, R = posterior_Rβ(x, y, prior_precision, likelihood_precision, degree)
    samples_Rβ = rand(posterior, n)
    samples_β = inv(R) * samples_Rβ
    reinterpret(reshape, SVector{degree + 1,Float64}, samples_β)
end

function log_posterior_Rβ_degree_nonorm_and_R(
    Rβ,
    x,
    y,
    prior_degree_dist::DiscreteUnivariateDistribution,
    prior_precision::Number,
    likelihood_precision::Number,
)
    degree = length(Rβ) - 1
    prior_Rβ_cov = inv(prior_precision * I)
    prior_Rβ_dist = MvNormal(zeros(degree + 1), prior_Rβ_cov)
    likelihood_cov = inv(likelihood_precision * I)
    QR = qr(design_matrix(x, degree))
    likelihood_dist = MvNormal(QR.Q * Rβ, likelihood_cov)

    logpdf(likelihood_dist, y) +
    logpdf(prior_degree_dist, degree) +
    logpdf(prior_Rβ_dist, Rβ),
    QR.R
end

function involutive_mcmc(
    x,
    y,
    prior_degree_dist,
    prior_precision::Number,
    likelihood_precision::Number,
    n::Int,
    sample_from_posterior,
    init_degree=rand(prior_degree_dist),
)
    βs = Vector{Float64}[]
    Rβs = Vector{Float64}[]
    degree = init_degree
    # directly sample initial Rβ from posterior conditional on degree
    Rβ_dist, R = posterior_Rβ(x, y, prior_precision, likelihood_precision, degree)
    Rβ = rand(Rβ_dist)
    β = inv(R) * Rβ
    push!(Rβs, copy(Rβ))
    push!(βs, β)
    for i in 2:n
        if mod(i, 2) == 0
            Rβ, R, accepted = involution_kernel!(
                Rβ,
                x,
                y,
                prior_degree_dist,
                prior_precision,
                likelihood_precision,
                sample_from_posterior,
            )
            degree = length(Rβ) - 1
        else
            Rβ_dist, R = posterior_Rβ(x, y, prior_precision, likelihood_precision, degree)
            Rβ = rand(Rβ_dist)
        end
        try
            β = inv(R) * Rβ
        catch e
            if e isa LAPACKException
                throw(BetterLAPACKException(i, R))
            end
            rethrow(e)
        end
        push!(Rβs, copy(Rβ))
        push!(βs, β)
    end
    βs, Rβs
end

struct BetterLAPACKException <: Exception
    i::Int
    R::Matrix{Float64}
end

function Base.showerror(io::IO, e::BetterLAPACKException)
    print(io, "BetterLAPACKException:\n")
    print(io, "Matrix R in iteration $(e.i) seems to be not invertible:\n")
    print(io, "$(e.R)")
end

function involution_kernel!(
    Rβ,
    x,
    y,
    prior_degree_dist,
    prior_precision,
    likelihood_precision,
    sample_from_posterior,
)
    log_p_old, R_old = log_posterior_Rβ_degree_nonorm_and_R(
        Rβ, x, y, prior_degree_dist, prior_precision, likelihood_precision
    )
    aux_dist_old = make_aux_dist(Rβ, x, y, prior_precision, likelihood_precision, sample_from_posterior)
    aux_old = rand(aux_dist_old)
    log_q_old = logpdf(aux_dist_old, aux_old)

    Rβ_new, aux_new, logabsdet = involution!(Rβ, aux_old)
    aux_dist_new = make_aux_dist(Rβ_new, x, y, prior_precision, likelihood_precision, sample_from_posterior)
    log_p_new, R_new = log_posterior_Rβ_degree_nonorm_and_R(
        Rβ_new, x, y, prior_degree_dist, prior_precision, likelihood_precision
    )
    log_q_new = logpdf(aux_dist_new, aux_new)

    a = exp(log_p_new + log_q_new - log_p_old - log_q_old + logabsdet)

    accepted = if rand() < a
        # accept
        Rβ = Rβ_new
        R = R_new
        true
    else
        # reject. Run involution again
        Rβ, _, _ = involution!(Rβ_new, aux_new)
        R = R_old
        false
    end
    @show log_p_new, log_q_new, log_p_old, log_q_old, a, accepted
    return Rβ, R, accepted
end

function make_aux_dist(Rβ, x, y, prior_precision, likelihood_precision, sample_from_posterior)
    degree = length(Rβ) - 1
    max_degree = length(y) - 1

    increase_dist = if degree == 0
        Bernoulli(1.0)
    elseif degree == max_degree
        Bernoulli(0.0)
    else
        Bernoulli(0.5)
    end
    coef_dist = if sample_from_posterior
        posterior_higher_degree = posterior_Rβ(
            x, y, prior_precision, likelihood_precision, degree + 1
        )[1]
        posterior_higher_degree_mean = mean(posterior_higher_degree)
        posterior_higher_degree_precision = invcov(posterior_higher_degree)
        posterior_cov = inv(posterior_higher_degree_precision[end, end])
        posterior_mean = @views posterior_higher_degree_mean[degree + 2] -
            posterior_cov * dot(
            posterior_higher_degree_precision[end, 1:(end - 1)],
            (Rβ - posterior_higher_degree_mean[1:(end - 1)]),
        )
        Normal(posterior_mean, sqrt(posterior_cov))
    else
        Normal(0.0, sqrt(inv(prior_precision)))
    end

    AuxiliaryDistribution(increase_dist, coef_dist)
end

function design_matrix(x, degree)
    degree == 0 ? ones(length(x), 1) : reduce(hcat, x .^ i for i in 0:degree)
end

function posterior_covariance(prior_precision, likelihood_precision, design_matrix)
    inv(prior_precision + design_matrix' * likelihood_precision * design_matrix)
end

function posterior_covariance(
    prior_precision, likelihood_precision::Number, design_matrix::LinearAlgebra.QRCompactWYQ
)
    posterior_covariance(prior_precision, likelihood_precision * I, design_matrix)
end

function posterior_covariance(
    prior_precision, likelihood_precision::UniformScaling, ::LinearAlgebra.QRCompactWYQ
)
    inv(prior_precision + likelihood_precision)
end