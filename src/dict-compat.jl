# Base.Dict compatibility layer for benchmarking
#
# Provides query/query! overloads for Base.Dict{K,V} matching the
# GPUHashTables interface, so Dict can be used as a baseline in benchmarks.

"""
    query!(results::Vector{V}, found::Vector{Bool}, dict::Dict{K,V}, keys::Vector{K})

Batch query keys against a Base.Dict, writing results in-place.
"""
function query!(
    results::Vector{V},
    found::Vector{Bool},
    dict::Dict{K,V},
    keys::Vector{K}
) where {K,V}
    n = length(keys)
    @assert length(results) >= n "Results vector too small"
    @assert length(found) >= n "Found vector too small"

    for i in 1:n
        val = get(dict, keys[i], nothing)
        if isnothing(val)
            @inbounds found[i] = false
        else
            @inbounds found[i] = true
            @inbounds results[i] = val
        end
    end

    return nothing
end

"""
    query(dict::Dict{K,V}, keys::Vector{K}) -> (found::Vector{Bool}, results::Vector{V})

Batch query keys against a Base.Dict, allocating result vectors.
"""
function query(dict::Dict{K,V}, keys::Vector{K}) where {K,V}
    n = length(keys)
    results = Vector{V}(undef, n)
    found = Vector{Bool}(undef, n)
    query!(results, found, dict, keys)
    return (found, results)
end
