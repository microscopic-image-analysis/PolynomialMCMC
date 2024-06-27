using Distributions
using GLMakie
using StaticArrays
using LinearAlgebra

struct Parameters{T}
    points::Vector{SVector{3, T}}
end

Parameters(n::Int) = Parameters([2 * @SVector(rand(3)) .- 1 for _ in 1:n])
struct ParameterDist{D<:DiscreteUnivariateDistribution}
    n_points_dist::D
end

function Distributions.logpdf(d::ParameterDist, params::Parameters)
    points = params.points
    n_points = length(points)

    outside_box = mapreduce(|, points) do point
        any(abs.(point) .> 1)
    end

    if outside_box
        return -Inf
    else
        # independent uniform distribution in 2x2x2 cube
        logpdf_points = -log(8) * n_points
        return logpdf_points + logpdf(d.n_points_dist, n_points)
    end
end

function involutive_mc_step(
    params::Parameters{T}, param_dist::ParameterDist, involution, make_aux_dist
) where {T<:Real}
    log_p_old = logpdf(param_dist::ParameterDist, params)
    aux_dist_old = make_aux_dist(params)
    aux_old = rand(aux_dist_old)
    log_q_old = logpdf(aux_dist_old, aux_old)

    params, aux_new, lad_jacobian = involution(params, aux_old)

    log_p_new = logpdf(param_dist::ParameterDist, params)
    aux_dist_new = make_aux_dist(params)
    log_q_new = logpdf(aux_dist_new, aux_new)

    # println("Target acceptance ratio: $(exp(log_p_new - log_p_old))")

    acceptance_probability = exp(
        log_p_new + log_q_new - log_p_old - log_q_old + lad_jacobian
    )

    println("$(aux_old) q acc ratio: $(exp(log_q_new - log_q_old)), p acc ratio: $(exp(log_p_new - log_p_old)), J: $(exp(lad_jacobian)), a: $acceptance_probability")
    # println("$(d[aux_old.split]) q acc ratio: $(exp(log_q_new - log_q_old)), p acc ratio: $(exp(log_p_new - log_p_old)), log_p_new: $log_p_new,  log_p_old: $log_p_old, J: $(exp(lad_jacobian))")

    return if rand() < acceptance_probability
        true, aux_old
    else
        params, _, _ = involution(params, aux_new)
        false, aux_old
    end
end

function sample(params, param_dist, involution, make_aux_dist, n)
    accepted = Bool[]
    n_points = Int[]
    for i in 1:n
        a, _ = involutive_mc_step(params, param_dist, involution, make_aux_dist)
        push!(accepted, a)
        push!(n_points, length(params.points))
    end
    return accepted, n_points
end

struct SplitMergeAuxDist{T}
    state::Parameters{T}
end

function Base.rand(d::SplitMergeAuxDist{T}) where T
    points = d.state.points
    n_points = length(points)
    bandwidth = T(0.2)
    split = n_points < 2 ? true : rand(Bernoulli(T(0.5)))
    if split
        # point_split_idx = rand(Distributions.Categorical(point_weights))
        point_split_idx = rand(DiscreteUniform(1, n_points))
        insert_idx = rand(DiscreteUniform(1, n_points + 1))
        point_delta = rand(MvNormal(@SVector(zeros(3)), bandwidth * I)) 
        return (; split, point_split_idx, insert_idx, point_delta)
    else
        # m = T(1.000001) * maximum(point_weights)
        # w = m .- point_weights
        # w ./= sum(w)
        # point_merge_idx = rand(Distributions.Categorical(w))
        point_merge_idx = rand(DiscreteUniform(1, n_points))
        w = distance_weights(points, point_merge_idx, bandwidth)
        point_idx2 = rand(Distributions.Categorical(w))
        return (; split, point_merge_idx, point_idx2)
    end
end

function Distributions.logpdf(d::SplitMergeAuxDist{T}, aux) where {T}
    points = d.state.points
    n_points = length(points)
    bandwidth = T(0.2)
    score = n_points < 2 ? zero(T) : logpdf(Bernoulli(T(0.5)), aux.split)
    if aux.split
        # score += logpdf(Distributions.Categorical(point_weights), aux.point_split_idx)
        score += logpdf(DiscreteUniform(1, n_points), aux.point_split_idx)
        score += logpdf(DiscreteUniform(1, n_points + 1), aux.insert_idx)
        score += logpdf(MvNormal(@SVector(zeros(3)), bandwidth * I), aux.point_delta) 
    else
        # m = T(1.000001) * maximum(point_weights)
        # w = m .- point_weights
        # w ./= sum(w)
        # score += logpdf(Distributions.Categorical(w), aux.point_merge_idx)
        score += logpdf(DiscreteUniform(1, n_points), aux.point_merge_idx)
        w = distance_weights(points, aux.point_merge_idx, bandwidth)
        score += logpdf(Distributions.Categorical(w), aux.point_idx2)
    end
    score
end

function distance_weights(points::AbstractVector{<:AbstractVector{T}}, ref_idx, sigma) where {T}
    if length(points) == 2
        weights = ones(T, 2)
        weights[ref_idx] = zero(T)
        return weights
    else
        ref_point = points[ref_idx]
        inv_var = inv(sigma^2)
        w = map(points) do point
            exp(-sum((point .- ref_point) .^ 2) * inv_var)
        end
        w[ref_idx] = zero(T)
        w .*= inv(sum(w))
        return w
    end
end

function split_merge_involution_weighted(state::Parameters, aux)
    points = state.points
    if aux.split
        original_point = points[aux.point_split_idx]
        T = eltype(original_point)
        point1 = original_point + aux.point_delta
        point2 = original_point - aux.point_delta
        points[aux.point_split_idx] = point1
        insert!(points, aux.insert_idx, point2)
        point_merge_idx = if aux.insert_idx <= aux.point_split_idx
            aux.point_split_idx + 1
        else
            aux.point_split_idx
        end

        lad = log(T(8))  # log abs det of Jacobian

        return state, (; split=false, point_merge_idx, point_idx2=aux.insert_idx), lad
    else
        point1 = points[aux.point_merge_idx]
        point2 = points[aux.point_idx2]
        T = eltype(point1)
        point_delta = T(0.5) * (point1 - point2)
        point = point2 + point_delta
        points[aux.point_merge_idx] = point
        deleteat!(points, aux.point_idx2)
        point_split_idx = if aux.point_idx2 < aux.point_merge_idx
            aux.point_merge_idx - 1
        else
            aux.point_merge_idx
        end

        lad = -log(T(8))  # log abs det of Jacobian

        state,
        (;
            split=true, point_split_idx, insert_idx=aux.point_idx2, point_delta
        ),
        lad
    end
end

struct BirthDeathAuxDist{T, D<:Distribution}
    state::Parameters{T}
    point_dist::D
end

function Base.rand(
    d::BirthDeathAuxDist{T}
) where {T}
    points = d.state.points
    n_points = length(points)
    birth = n_points < 2 ? true : rand(Bernoulli(T(0.5)))
    if birth
        point_birth_idx = rand(DiscreteUniform(1, n_points + 1))
        point = rand(d.point_dist)
        return (; birth, point_birth_idx, point)
    else
        point_death_idx = rand(DiscreteUniform(1, n_points))
        return (; birth, point_death_idx)
    end
end

function Distributions.logpdf(d::BirthDeathAuxDist{T}, aux) where {T}
    points = d.state.points
    n_points = length(points)
    score = n_points < 2 ? zero(T) : logpdf(Bernoulli(T(0.5)), aux.birth)
    if aux.birth
        score += logpdf(DiscreteUniform(1, n_points + 1), aux.point_birth_idx)
        score += logpdf(d.point_dist, aux.point)
    else
        score += logpdf(DiscreteUniform(1, n_points), aux.point_death_idx)
    end
    score
end

function birth_death_involution(state::Parameters{T}, aux) where {T}
    points = state.points
    if aux.birth
        i = aux.point_birth_idx
        insert!(points, i, aux.point)
        return state, (; birth=false, point_death_idx=i), zero(T)
    else
        i = aux.point_death_idx
        point = popat!(points, i)
        return state, (; birth=true, point_birth_idx=i, point), zero(T)
    end
end