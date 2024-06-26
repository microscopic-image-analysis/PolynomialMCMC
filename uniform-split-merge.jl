using Distributions
using GLMakie
using StaticArrays
using LinearAlgebra

struct Parameters{T}
    points::Vector{SVector{3, T}}
end

Parameters(n::Int) = Parameters([2 * @SVector(rand(3)) .- 1 for _ in 1:n])

function logdensity(n_points_dist, params::Parameters)
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
        return logpdf_points + logpdf(n_points_dist, n_points)
    end
end

function split_merge_involution_weighted(points, aux)
    if aux.split
        original_point = points[aux.point_split_idx]
        T = eltype(original_point)
        point1 = original_point + aux.point_delta
        point2 = original_point - aux.point_delta
        points[aux.point_split_idx] = point1
        points = insert!(points, aux.insert_idx, point2)
        point_merge_idx = if aux.insert_idx <= aux.point_split_idx
            aux.point_split_idx + 1
        else
            aux.point_split_idx
        end

        lad = log(T(8))  # log abs det of Jacobian

        return points, (; split=false, point_merge_idx, point_idx2=aux.insert_idx), lad
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

        points,
        (;
            split=true, point_split_idx, insert_idx=aux.point_idx2, point_delta
        ),
        lad
    end
end

function sample_aux_split_merge(state::Parameters{T}) where {T}
    points = state.points
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
        w = distance_weights(state.points, point_merge_idx, bandwidth)
        point_idx2 = rand(Distributions.Categorical(w))
        return (; split, point_merge_idx, point_idx2)
    end
end

function logpdf_aux_split_merge(aux, state::Parameters{T}) where {T}
    points = state.points
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
        w = distance_weights(state.points, aux.point_merge_idx, bandwidth)
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

function sample_split_merge(
    params::Parameters{T}, n_points_prior_dist
) where {T<:Real}
    log_p_old = logdensity(n_points_prior_dist, params)
    aux_old = sample_aux_split_merge(params)
    log_q_old = logpdf_aux_split_merge(aux_old, params)

    _, aux_new, lad_jacobian = split_merge_involution_weighted(params.points, aux_old)

    log_p_new = logdensity(n_points_prior_dist, params)
    log_q_new = logpdf_aux_split_merge(aux_new, params)

    # println("Target acceptance ratio: $(exp(log_p_new - log_p_old))")

    acceptance_probability = exp(
        log_p_new + log_q_new - log_p_old - log_q_old + lad_jacobian
    )
    d = Dict(true => "Split", false => "Merge")

    println("$(d[aux_old.split]) q acc ratio: $(exp(log_q_new - log_q_old)), p acc ratio: $(exp(log_p_new - log_p_old)), J: $(exp(lad_jacobian)), a: $acceptance_probability")
    # println("$(d[aux_old.split]) q acc ratio: $(exp(log_q_new - log_q_old)), p acc ratio: $(exp(log_p_new - log_p_old)), log_p_new: $log_p_new,  log_p_old: $log_p_old, J: $(exp(lad_jacobian))")

    return if rand() < acceptance_probability
        # accept
        # if aux_old.split
        #     point1 = params.points[aux_old.point_split_idx]
        #     point2 = params.points[aux_old.insert_idx]
        #     println("Splitted point at index $(aux_old.point_split_idx) into $point1, $point2. P_acc = $acceptance_probability")
        # else
        #     point = params.points[aux_old.point_merge_idx]
        #     println("Merged point at index $(aux_old.point_merge_idx) with point at index $(aux_old.point_idx2): $point. P_acc = $acceptance_probability")
        # end
        true, aux_old.split
    else
        # if aux_old.split
        #     println("Rejected split of point at index $(aux_old.point_split_idx). P_acc = $acceptance_probability")
        # else
        #     println("Rejected merge of point at index $(aux_old.point_merge_idx) with point at index $(aux_old.point_idx2). P_acc = $acceptance_probability")
        # end
        # reject. Call involution again to go back to old state
        split_merge_involution_weighted(params.points, aux_new)
        false, aux_old.split
    end
end

function sample(params, n_points_prior_dist, n)
    accepted = Bool[]
    split = Bool[]
    n_points = Int[]
    for i in 1:n
        a, s = sample_split_merge(params, n_points_prior_dist)
        push!(accepted, a)
        push!(split, s)
        push!(n_points, length(params.points))
    end
    return accepted, split, n_points
end