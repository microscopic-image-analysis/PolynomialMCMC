"""
This file contains implementations of bayesian polynomial regression.
All models assume the noise level is known.
Main components:

* `struct Joint` describes the probabilistic model for this task.
* `condition(joint::Joint, degree, y)` conditions the model on 
  observations `y` and a specific polynomial degree. 
  This can be done in closed form, therefore this function returns
  a probability distribution (`MvNormal`).
* `sample(joint::Joint, y, ...)` samples from the model given only
  observations `y` using involutive MCMC

See demo.jl for how to use.
"""

using Distributions
using LinearAlgebra

"""
    Joint

parameters of joint probability distribution for the random variables `y`, `Rβ` and `d`:
y ~ MvNormal(ŷ, `likelihood_variance`)
ŷ = Q * Rβ
Rβ ~ MvNormal(zeros(d + 1), `prior_variance`)
d ~ `prior_degree_dist`
"""
struct Joint{D}
    x::Vector{Float64}
    prior_degree_dist::D
    prior_variance::Float64
    likelihood_variance::Float64
end

"""
    logpdf(target, (Rβ, y))

Evaluate (non-normalized) log density of joint distribution at `Rβ`, `d`, `y`.
"""
function Distributions.logpdf(joint::Joint, (Rβ, y))
    # Polynomial degree:
    degree = length(Rβ) - 1
    # Estimate y given polynomial coefficients and x:
    # X = design_matrix(x, degree)
    # ŷ = X * β = QR * β = Q * Rβ
    X = design_matrix(joint.x, degree)
    Q, R = qr(X)  # X = Q*R
    ŷ = Q * Rβ

    # Prior for Rβ:
    # Assume independently normally distributed around zero:
    prior_coef_dist = MvNormal(zeros(degree + 1), joint.prior_variance * I)

    # Gaussian likelihood around ŷ:
    likelihood_dist = MvNormal(ŷ, joint.likelihood_variance * I)

    # Return total log density
    return logpdf(likelihood_dist, y) +  # p(y | ŷ)
           logpdf(joint.prior_degree_dist, degree) +  # p(degree)
           logpdf(prior_coef_dist, Rβ)  # p(β)
end

"""
    SimpleAuxiliaryDistribution

parameters of a distribution for auxiliary variables conditioned on current
non-auxiliary variables.
"""
struct SimpleAuxiliaryDistribution{D,C}
    increase_dist::D
    coef_dist::C
end

"""
    SimpleAuxiliaryDistribution(joint, Rβ, y)

Create auxiliary distribution conditioned on current state `Rβ` and observations `y`
"""
function SimpleAuxiliaryDistribution(joint::Joint, Rβ, y)
    # Distribution for auxiliary variable(s) given current state
    degree = length(Rβ) - 1
    max_degree = length(y) - 1

    increase_dist = if degree == 0
        # if currently constant polynomial, definitely increase degree
        Bernoulli(1.0)
    elseif degree == max_degree
        # if currently maximal possible degree, definitely decrease degree
        Bernoulli(0.0)
    else
        # else 50:50 chance
        Bernoulli(0.5)
    end

    # use prior for coefficients as proposal for new coefficient:
    coef_dist = Normal(0.0, sqrt(joint.prior_variance))

    return SimpleAuxiliaryDistribution(increase_dist, coef_dist)
end

"""
    rand(aux_dist)

Take a random sample from auxiliary distribution
"""
function Base.rand(dist::SimpleAuxiliaryDistribution)
    increase = rand(dist.increase_dist)
    coef = if increase
        rand(dist.coef_dist)
    else
        NaN
    end
    (; increase, coef)
end

"""
    logpdf(aux_dist, sample)

Evaluate log density of `aux_dist` at `sample`.
"""
function Distributions.logpdf(
    dist::SimpleAuxiliaryDistribution, val::NamedTuple{(:increase, :coef)}
)
    increase = val.increase
    score = logpdf(dist.increase_dist, increase)
    if increase
        score += logpdf(dist.coef_dist, val.coef)
    end
    score
end

"""
    push_pop_involution!(coefs, aux)

If `aux.increase`, push `aux.coef` to the end of `coefs`,
otherwise do the inverse.
One property of this function (and all involutions) is:
`coefs, aux == involution!(involution!(copy(coefs), aux)[1:2]...)`
"""
function push_pop_involution!(coefs, aux::NamedTuple{(:increase, :coef)})
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
    condition(joint, degree, y)

Condition `joint`` on `degree` and `y`
I.e. return the posterior distribution over polynomial coefficients Rβ
given observations `y` and a given `degree`
"""
function condition(joint::Joint, degree::Int, y)
    # assume linear-gaussian model, i.e.
    # gaussian prior and gaussian likelihood:
    # Rβ ~ MvNormal(0, σ1)
    # ŷ = Q * Rβ
    # y ~ MvNormal(ŷ, σ2) 
    # 
    # return posterior for Rβ which is again gaussian
    X = design_matrix(joint.x, degree)
    Q, R = qr(X)

    prior_precision = inv(joint.prior_variance)
    likelihood_precision = inv(joint.likelihood_variance)
    posterior_variance = inv(prior_precision + likelihood_precision)  # Bishop 2.117 
    posterior_mean = (posterior_variance * I) * Matrix(Q)' * (likelihood_precision * I) * y  # Bishop 2.116
    return MvNormal(posterior_mean, posterior_variance * I)
end

"""
    randprior(joint)

Perform ancestral sampling using prior distributions and return sample.
"""
function randprior(joint::Joint)
    degree = rand(joint.prior_degree_dist)
    # Prior for Rβ:
    # Assume independently normally distributed around zero:
    prior_coef_dist = MvNormal(zeros(degree + 1), joint.prior_variance * I)
    Rβ = rand(prior_coef_dist)
    return degree, Rβ
end

"""
    involutive_mc_step(Rβ, ...)

Perform a single involutive Metropolis-Hastings step.
"""
function involutive_mc_step(Rβ, target, y, aux_dist, involution)
    log_p_old = logpdf(target, (Rβ, y))  # logdensity of target at current Rβ
    aux_dist_old = aux_dist(target, Rβ, y)  # auxiliary distribution given current Rβ
    aux_old = rand(aux_dist_old)  # sample auxiliary state
    log_q_old = logpdf(aux_dist_old, aux_old)  # evaluate logpdf of auxiliary state

    Rβ, aux_new, lad_jacobian = involution(Rβ, aux_old)

    log_p_new = logpdf(target, (Rβ, y))  # logdensity of target at *proposed* Rβ
    aux_dist_new = aux_dist(target, Rβ, y)  # auxiliary distribution given *proposed* Rβ
    log_q_new = logpdf(aux_dist_new, aux_new)

    acceptance_probability = exp(
        log_p_new + log_q_new - log_p_old - log_q_old + lad_jacobian
    )

    accepted = rand() < acceptance_probability

    if !accepted
        # simply call involution again to return to old Rβ
        Rβ, _, _ = involution(Rβ, aux_new)
    end

    return accepted, Rβ
end

"""
    sample(...)

sample from `target` using involutive MCMC.
"""
function sample(
    target::Joint, y, aux_dist, involution, n_samples::Int, init_state=randprior(target)
)
    βs = Vector{Float64}[]
    Rβs = Vector{Float64}[]
    degrees = Int[]
    accepteds = Bool[]
    # sample initial state (from prior):
    degree, Rβ = init_state
    # Gibbs sampling:
    # every odd step sample from p(Rβ | degree)
    # every even step sample from full p(Rβ, degree)
    for i in 2:n_samples
        if mod(i, 2) == 0  # sample from full posterior
            accepted, Rβ = involutive_mc_step(Rβ, target, y, aux_dist, involution)
            degree = length(Rβ) - 1
            push!(accepteds, accepted)
        else  # sample β
            Rβ_dist = condition(target, degree, y)  # target conditioned on degree and y
            # can directly sample from this
            Rβ = rand(Rβ_dist)
        end
        Q, R = qr(design_matrix(target.x, degree))
        β = R \ Rβ
        push!(βs, β)
        push!(Rβs, copy(Rβ))
        push!(degrees, degree)
    end
    βs, Rβs, degrees, accepteds
end

"""
    design_matrix(x, degree)

return the [Vandermonde-matrix](https://en.wikipedia.org/wiki/Vandermonde_matrix) of x
"""
function design_matrix(x, degree)
    degree == 0 ? ones(length(x), 1) : reduce(hcat, x .^ i for i in 0:degree)
end